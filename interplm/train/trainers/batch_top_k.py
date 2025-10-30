from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch as t

from interplm.sae.dictionary import (
    BatchTopKSAE,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)
from interplm.train.trainers.base_trainer import SAETrainer, SAETrainerConfig
from interplm.train.trainers.common import get_lr_schedule, get_autocast_context
from interplm.utils import get_device


@dataclass
class BatchTopKTrainerConfig(SAETrainerConfig):
    # Number of features to keep
    k: int = 10
    # Auxiliary loss weight
    auxk_alpha: float = 1 / 32
    # Beta for threshold update (how much to update the threshold each batch)
    threshold_beta: float = 0.999
    # Step at which to start updating the threshold
    threshold_start_step: int = 1000

    # Dead feature threshold
    dead_feature_threshold: int = 10_000_000

    def __post_init__(self):
        if self.lr is None:
            scale = self.activation_dim * self.expansion_factor / (2**14)
            self.lr = 2e-4 / scale**0.5

    def trainer_cls(self) -> type["BatchTopKTrainer"]:
        return BatchTopKTrainer


class BatchTopKTrainer(SAETrainer):
    def __init__(
        self,
        trainer_config: BatchTopKTrainerConfig,
    ):
        super().__init__(
            trainer_config,
            logging_parameters=[
                "training/learning_rate",
                "training/threshold", 
                "features/total_dead",
            ],
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

        self.ae = BatchTopKSAE(
            activation_dim=trainer_config.activation_dim,
            dict_size=trainer_config.expansion_factor * trainer_config.activation_dim,
            k=trainer_config.k,
            normalize_to_sqrt_d=trainer_config.normalize_to_sqrt_d,
        )

        self.device = get_device()
        self.ae.to(self.device)

        self.auxk_alpha = trainer_config.auxk_alpha
        self.dead_feature_threshold = trainer_config.dead_feature_threshold
        self.top_k_aux = (
            trainer_config.activation_dim // 2
        )  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(
            self.ae.dict_size, dtype=t.long, device=self.device
        )
        self.steps_since_active = t.zeros(
            self.ae.dict_size, dtype=t.long, device=self.device
        )
        # Use namespaced attribute names to match logging parameters
        setattr(self, "features/total_dead", -1)

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(
            self.steps,
            self.warmup_steps,
            decay_start=self.decay_start,
        )

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        """
        Compute the auxiliary loss for the batch top-k SAE.

        This loss encourages dead features to become active by:
        1. Identifying dead features (those that haven't fired in a long time)
        2. Taking the top-k dead features from the post-ReLU activations
        3. Computing how well these dead features can reconstruct the residual error
        4. Normalizing this reconstruction loss by the variance of the residual

        This helps prevent features from becoming permanently inactive by giving
        them an opportunity to learn useful patterns from the residual error.
        """
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        setattr(self, "features/total_dead", int(dead_features.sum()))

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            # Only look at activations of dead features, mask others to -inf
            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Get top-k dead feature activations
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Create sparse tensor with only the top-k dead activations
            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Try to reconstruct the residual using only dead features
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            # Remove pre_norm_auxk_loss tracking since auxk_loss is sufficient

            # Normalize by variance of residual to make loss scale-invariant
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    @property
    def threshold(self):
        return self.ae.threshold
        
    # Property accessors for namespaced logging parameters
    @property 
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def update_threshold(self, f: t.Tensor):
        with get_autocast_context(self.device, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        # The SAE model handles normalization internally
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_hat = self.ae.decode(f)

        # Compute reconstruction error
        e = x - x_hat  # Shape: (batch_size, d_input)

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        # Update steps since active
        self.steps_since_active += 1
        self.steps_since_active[did_fire] = 0

        # Compute reconstruction loss
        l2_loss = e.pow(2).sum(dim=-1).mean()
            
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        loss = l2_loss + self.auxk_alpha * auxk_loss

        # track a normalized l2 loss
        normalized_l2_loss = l2_loss / (x.size(1))

        if not logging:
            return loss
        else:
            # Return preorganized namespaced metrics
            loss_dict = {
                "loss/reconstruction": l2_loss.item(),
                "loss/auxiliary": auxk_loss.item(), 
                "loss/total": loss.item(),
                "performance/mse_normalized": normalized_l2_loss.item(),
            }

            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                loss_dict,
            )

    def update(self, step, x):
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        if self.grad_clip_norm is not None:
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    def get_per_dimension_mse(self, x):
        """
        Calculate per-dimension MSE for final analysis.
        
        Args:
            x: Input activations to evaluate
            
        Returns:
            torch.Tensor: Per-dimension MSE, shape (activation_dim,)
        """
        x = x.to(self.device)
        with t.no_grad():
            # Get reconstruction
            f, _, _ = self.ae.encode(x, return_active=True, use_threshold=True)
            x_hat = self.ae.decode(f)
            
            # Calculate per-dimension squared error, averaged across batch
            per_dim_mse = (x - x_hat).pow(2).mean(dim=0)
            
        return per_dim_mse

    @classmethod
    def dictionary_cls(cls):
        return BatchTopKSAE
