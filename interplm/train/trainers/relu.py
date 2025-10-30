"""
Implements the standard SAE training scheme.
(Originally from https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/trainers/standard.py)
"""

from collections import namedtuple
from dataclasses import dataclass

import torch as t

from interplm.sae.dictionary import ReLUSAE
from interplm.train.trainers.base_trainer import SAETrainer, SAETrainerConfig
from interplm.train.trainers.common import (
    ConstrainedAdam,
    get_lr_schedule,
    get_sparsity_warmup_fn,
)
from interplm.utils import get_device


@dataclass
class ReLUTrainerConfig(SAETrainerConfig):
    l1_penalty: float | None = None
    l1_penalty_warmup_steps: int | None = None

    @classmethod
    def trainer_cls(cls) -> type["ReLUTrainer"]:
        return ReLUTrainer


class ReLUTrainer(SAETrainer):
    """
    ReLU SAE training implementation with L1 sparsity and neuron resampling.

    Implements training with:
    - L1 sparsity penalty
    - Learning rate warmup and decay
    - Dead neuron detection and resampling
    - L1 penalty warmup
    """

    def __init__(
        self,
        trainer_config: ReLUTrainerConfig,
    ):
        super().__init__(
            trainer_config=trainer_config,
            logging_parameters=["training/learning_rate", "training/l1_penalty"],
        )

        # Initialize autoencoder with rescaling if configured
        self.ae = ReLUSAE(
            activation_dim=trainer_config.activation_dim,
            dict_size=trainer_config.dictionary_size,
            normalize_to_sqrt_d=trainer_config.normalize_to_sqrt_d,
        )

        # Training parameters
        self.lr = trainer_config.lr
        self.steps = trainer_config.steps
        self.warmup_steps = trainer_config.warmup_steps
        self.decay_start = trainer_config.decay_start
        self.grad_clip_norm = trainer_config.grad_clip_norm

        self.device = get_device()
        self.ae.to(self.device)

        # Set sparsity warmup steps if it is None
        if trainer_config.l1_penalty_warmup_steps is None:
            trainer_config.l1_penalty_warmup_steps = int(self.steps * 0.05)

        # Resampling setup
        self.resample_steps = trainer_config.resample_steps
        self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(
            self.device
        )

        # Initialize optimizer with constrained decoder weights
        self.optimizer = ConstrainedAdam(
            params=self.ae.parameters(),
            constrained_params=self.ae.decoder.parameters(),
            lr=self.lr,
        )

        # Setup learning rate schedule with warmup and decay
        lr_fn = get_lr_schedule(
            total_steps=self.steps,
            warmup_steps=self.warmup_steps,
            decay_start=self.decay_start,
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # L1 penalty warmup setup
        self.l1_penalty = trainer_config.l1_penalty
        self.l1_penalty_warmup_steps = trainer_config.l1_penalty_warmup_steps
        self.l1_penalty_warmup_fn = get_sparsity_warmup_fn(
            self.steps, self.l1_penalty_warmup_steps
        )

        print(f"Training with config: {self.config}")

    @classmethod
    def dictionary_cls(cls):
        return ReLUSAE

    @property
    def current_lr(self):
        """Get current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def loss(self, x, step=None, logging=False, **kwargs):
        """
        Compute loss for current batch.

        Args:
            x: Input activations (unnormalized)
            logging: Whether to return extended logging information

        Returns:
            If logging=False: Combined loss value
            If logging=True: Named tuple with reconstruction details and losses
        """
        # The SAE model handles normalization internally
        x_hat, f = self.ae(x, output_features=True)

        # Compute reconstruction loss
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
            
        l1_loss = f.norm(p=1, dim=-1).mean()

        # Track inactive neurons
        deads = (f == 0).all(dim=0)
        self.steps_since_active[deads] += 1
        self.steps_since_active[~deads] = 0

        self.current_l1_penalty_scale = self.l1_penalty * self.l1_penalty_warmup_fn(
            step
        )
        loss = l2_loss + l1_loss * self.current_l1_penalty_scale

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "loss/reconstruction": l2_loss.item(),
                    "loss/sparsity": l1_loss.item(),
                    "loss/total": loss.item(),
                },
            )

    def update(self, step, x):
        """
        Perform single training step.

        Args:
            step: Current training step
            x: Batch of input activations
        """
        x = x.to(self.device)

        # Compute and apply gradients
        self.optimizer.zero_grad()
        loss = self.loss(x, step=step)
        loss.backward()

        if self.grad_clip_norm is not None:
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()

        # Check for dead neurons
        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(
                self.steps_since_active > self.resample_steps / 2, x
            )

        return loss.item()
    
    def resample_neurons(self, dead_neurons: t.Tensor, activations: t.Tensor):
        """
        Resample dead neurons by reinitializing them with samples from the data.
        
        Args:
            dead_neurons: Boolean tensor indicating which neurons are dead
            activations: Current batch of activations to sample from
        """
        if not dead_neurons.any():
            return
            
        n_dead = dead_neurons.sum().item()
        print(f"Resampling {n_dead} dead neurons")
        
        with t.no_grad():
            # Sample from current activations
            if activations.shape[0] >= n_dead:
                sampled_indices = t.randperm(activations.shape[0])[:n_dead]
                sampled_activations = activations[sampled_indices]
            else:
                # If we don't have enough samples, repeat some
                sampled_activations = activations[t.randint(0, activations.shape[0], (n_dead,))]
            
            # Reinitialize encoder weights for dead neurons
            # Use a small random perturbation of the sampled activations
            noise_scale = 0.01
            new_encoder_weights = sampled_activations + noise_scale * t.randn_like(sampled_activations)
            self.ae.encoder.weight.data[dead_neurons] = new_encoder_weights / new_encoder_weights.norm(dim=1, keepdim=True)
            
            # Reinitialize encoder bias for dead neurons
            self.ae.encoder.bias.data[dead_neurons] = 0.0
            
            # Reinitialize decoder weights for dead neurons
            # Set to be the transpose of encoder weights (standard initialization)
            self.ae.decoder.weight.data[:, dead_neurons] = self.ae.encoder.weight.data[dead_neurons].T
            
            # Reset tracking for resampled neurons
            self.steps_since_active[dead_neurons] = 0
