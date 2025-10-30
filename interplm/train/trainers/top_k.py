from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch as t

from interplm.sae.dictionary import (
    TopKSAE,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)
from interplm.train.trainers.base_trainer import SAETrainer, SAETrainerConfig
from interplm.train.trainers.common import get_lr_schedule, get_autocast_context
from interplm.utils import get_device


@dataclass
class TopKTrainerConfig(SAETrainerConfig):
    k: int = 10
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    trainer_name: str = "TopKTrainer"

    def __post_init__(self):
        if self.lr is None:
            scale = self.activation_dim * self.expansion_factor / (2**14)
            self.lr = 2e-4 / scale**0.5

    @classmethod
    def trainer_cls(cls) -> type["TopKTrainer"]:
        return TopKTrainer


class TopKTrainer(SAETrainer):
    """
    Top-K SAE training scheme.
    """

    def __init__(
        self,
        trainer_config: TopKTrainerConfig,
    ):
        super().__init__(
            trainer_config,
            logging_parameters=["effective_l0", "dead_features", "pre_norm_auxk_loss"],
        )

        # Training parameters
        self.lr = trainer_config.lr
        self.steps = trainer_config.steps
        self.decay_start = trainer_config.decay_start
        self.warmup_steps = trainer_config.warmup_steps
        self.grad_clip_norm = trainer_config.grad_clip_norm

        # Top-K parameters
        self.k = trainer_config.k
        self.threshold_beta = trainer_config.threshold_beta
        self.threshold_start_step = trainer_config.threshold_start_step

        # Initialise autoencoder
        self.ae = TopKSAE(
            activation_dim=trainer_config.activation_dim,
            dict_size=trainer_config.dictionary_size,
            k=trainer_config.k,
            normalize_to_sqrt_d=trainer_config.normalize_to_sqrt_d,
        )
        self.device = get_device()
        self.ae.to(self.device)

        self.auxk_alpha = trainer_config.auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = (
            trainer_config.activation_dim // 2
        )  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(
            self.ae.dict_size, dtype=t.long, device=self.device
        )
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(
            total_steps=self.steps,
            warmup_steps=self.warmup_steps,
            decay_start=self.decay_start,
        )

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    @classmethod
    def dictionary_cls(cls):
        return TopKSAE

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, top_acts_BK: t.Tensor):
        with get_autocast_context(self.device, enabled=False), t.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            B, K = active.shape
            assert len(active.shape) == 2
            assert min_activations.shape == (B,)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        # The SAE model handles normalization internally
        # Run the SAE
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)

        # Compute reconstruction error
        e = x - x_hat

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
            if self.auxk_alpha > 0
            else 0
        )

        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        if self.grad_clip_norm is not None:
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()
