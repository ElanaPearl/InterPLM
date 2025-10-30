from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.autograd as autograd
from torch import nn

from interplm.sae.dictionary import (
    JumpReLUSAE,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)
from interplm.train.trainers.base_trainer import SAETrainer, SAETrainerConfig
from interplm.train.trainers.common import get_lr_schedule, get_sparsity_warmup_fn
from interplm.utils import get_device


@dataclass
class JumpReLUTrainerConfig(SAETrainerConfig):
    bandwidth: float = 0.001
    sparsity_penalty: float = 1.0
    sparsity_warmup_steps: Optional[int] = 2000
    target_l0: float = 20.0

    @classmethod
    def trainer_cls(cls) -> type["JumpReLUTrainer"]:
        return JumpReLUTrainer


class JumpReLUTrainer(SAETrainer):
    """
    Trains a JumpReLU autoencoder.

    Note does not use learning rate or sparsity scheduling as in the paper.
    """

    def __init__(
        self,
        trainer_config: JumpReLUTrainerConfig,
    ):
        super().__init__(
            trainer_config=trainer_config, logging_parameters=["dead_features"]
        )

        # Training parameters
        self.lr = trainer_config.lr
        self.steps = trainer_config.steps
        self.warmup_steps = trainer_config.warmup_steps
        self.decay_start = trainer_config.decay_start
        self.grad_clip_norm = trainer_config.grad_clip_norm

        # JumpReLU parameters
        self.bandwidth = trainer_config.bandwidth
        self.sparsity_coefficient = trainer_config.sparsity_penalty
        self.sparsity_warmup_steps = trainer_config.sparsity_warmup_steps
        self.target_l0 = trainer_config.target_l0

        self.device = get_device()

        self.ae = JumpReLUSAE(
            activation_dim=trainer_config.activation_dim,
            dict_size=trainer_config.dictionary_size,
            normalize_to_sqrt_d=trainer_config.normalize_to_sqrt_d,
        ).to(self.device)

        # Parameters from the paper
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.0, 0.999), eps=1e-8
        )

        lr_fn = get_lr_schedule(
            total_steps=self.steps,
            warmup_steps=self.warmup_steps,
            decay_start=self.decay_start,
            sparsity_warmup_steps=self.sparsity_warmup_steps,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            self.steps, self.sparsity_warmup_steps
        )

        # Purely for logging purposes
        self.dead_feature_threshold = 10_000_000
        self.num_tokens_since_fired = torch.zeros(
            self.ae.dict_size, dtype=torch.long, device=self.device
        )
        self.dead_features = -1

    def loss(self, x: torch.Tensor, step: int, logging=False, **_):
        # Note: We are using threshold, not log_threshold as in this notebook:
        # https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD#scrollTo=yP828a6uIlSO
        # I had poor results when using log_threshold and it would complicate the scale_biases() function

        # The SAE model handles normalization internally
        sparsity_scale = self.sparsity_warmup_fn(step)
        x = x.to(self.ae.encoder.weight.dtype)

        pre_jump = self.ae.encoder(x)
        f = JumpReLUFunction.apply(pre_jump, self.ae.threshold, self.bandwidth)

        active_indices = f.sum(0) > 0
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += x.size(0)
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (
            (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()
        )

        recon = self.ae.decode(f)

        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()

        sparsity_loss = (
            self.sparsity_coefficient
            * ((l0 / self.target_l0) - 1).pow(2)
            * sparsity_scale
        )
        loss = recon_loss + sparsity_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "recon", "f", "losses"])(
                x,
                recon,
                f,
                {
                    "l2_loss": recon_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # Remove gradient parallel to decoder directions
        # decoder.weight has shape (dict_size, activation_dim) for nn.Linear
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Renormalize decoder weights to unit norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
