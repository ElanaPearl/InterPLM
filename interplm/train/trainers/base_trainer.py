from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch as t


@dataclass
class SAETrainerConfig(ABC):
    expansion_factor: int | None = None
    dictionary_size: int | None = None
    lr: float | None = None
    activation_dim: int | None = None
    epochs: float | None = None
    steps: int | None = None
    warmup_steps: int = None
    decay_start: int = None
    resample_steps: int | None = None
    normalize_to_sqrt_d: bool = False # previously apply_unit_normalization
    grad_clip_norm: float | None = 1.0  # Gradient clipping norm (None = no clipping)

    def set_and_validate_activation_dim(self, activation_dim: int):
        self.activation_dim = activation_dim

        # If neither expansion_factor or dictionary_size is set, raise error
        if self.expansion_factor is None and self.dictionary_size is None:
            raise ValueError("Either expansion_factor or dictionary_size must be set")
        # If only one is set, set the other according to the formula
        elif self.expansion_factor is not None and self.dictionary_size is None:
            self.dictionary_size = self.expansion_factor * self.activation_dim
        elif self.dictionary_size is not None and self.expansion_factor is None:
            self.expansion_factor = self.dictionary_size / self.activation_dim
        # If both are set, check if they are consistent
        else:
            if self.expansion_factor * self.activation_dim != self.dictionary_size:
                raise ValueError(
                    "expansion_factor and dictionary_size are inconsistent"
                )

    def validate_and_check_training_steps(self, dataloader_len: int):
        if self.epochs is None and self.steps is not None:
            self.epochs = self.steps / dataloader_len
        elif self.epochs is not None and self.steps is None:
            self.steps = self.epochs * dataloader_len
        elif self.epochs is not None and self.steps is not None:
            if abs(int(self.epochs * dataloader_len) - self.steps) > 1:
                raise ValueError(
                    f"Epochs ({self.epochs}) * dataloader_len ({dataloader_len}) = {int(self.epochs * dataloader_len)} is not equal to steps ({self.steps})"
                )
        else:
            print("Steps not specified, using full dataset")
            self.steps = dataloader_len
            self.epochs = 1

        if self.warmup_steps is None:
            self.warmup_steps = self.steps * 0.05
        if self.decay_start is None:
            self.decay_start = self.steps // 2
        # If decay_start is negative, it means we do not decay
        elif self.decay_start < 0:
            self.decay_start = self.steps + 1

    @abstractmethod
    def trainer_cls(self):
        raise NotImplementedError

    def build(self) -> "SAETrainer":
        trainer_cls = self.trainer_cls()
        return trainer_cls(self)


class SAETrainer:
    """
    Generic class for implementing SAE training algorithms.

    Base class that provides common functionality for SAE training implementations.
    Subclasses should implement the update method to define specific training behavior.

    """

    def __init__(
        self,
        trainer_config: SAETrainerConfig,
        logging_parameters: list[str] = [],
    ):
        self.config = trainer_config
        self.logging_parameters = logging_parameters

    def update(
        self,
        step: int,  # index of step in training
        activations: t.Tensor,  # shape [batch_size, d_submodule]
    ):
        """
        Update the model based on current step and activations.

        Args:
            step: Current training step number
            activations: Batch of input activations to process
        """
        pass  # implemented by subclasses

    def get_logging_parameters(self):
        """
        Collect all registered logging parameters from the trainer.
        
        Supports both regular attributes and namespaced parameters 
        (e.g., "training/learning_rate" maps to current_lr property).

        Returns:
            Dictionary mapping parameter names to their current values
        """
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                # Direct attribute access (for namespaced attributes)
                stats[param] = getattr(self, param)
            else:
                # Try to map namespaced parameter to property
                if param == "training/learning_rate" and hasattr(self, "current_lr"):
                    stats[param] = self.current_lr
                elif param == "training/threshold" and hasattr(self, "threshold"):
                    stats[param] = self.threshold
                elif param == "training/l1_penalty" and hasattr(self, "current_l1_penalty_scale"):
                    stats[param] = self.current_l1_penalty_scale
                else:
                    print(f"Warning: {param} not found in {self}")
        return stats

    @property
    def current_lr(self):
        """Get current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, save_dir: Path):
        """Save the model state and trainer configuration to a checkpoint."""
        save_dir = Path(save_dir)
        # save the model state
        t.save(self.ae.state_dict(), save_dir / "checkpoint.pt")

        # save optimizer and scheduler states
        optimizer_state = {
            "optimizer": (
                self.optimizer.state_dict() if hasattr(self, "optimizer") else None
            ),
            "scheduler": (
                self.scheduler.state_dict() if hasattr(self, "scheduler") else None
            ),
        }
        t.save(optimizer_state, save_dir / "optimizer.pt")

    def update_from_checkpoint(self, checkpoint_dir: Path):
        """Update the trainer from a checkpoint."""
        self.ae.load_state_dict(t.load(checkpoint_dir / "checkpoint.pt"))

        # First confirm that optimizer and scheduler exist
        if not (checkpoint_dir / "optimizer.pt").exists():
            raise FileNotFoundError(f"Optimizer state not found in {checkpoint_dir}")

        optimizer_state = t.load(checkpoint_dir / "optimizer.pt")
        self.optimizer.load_state_dict(optimizer_state["optimizer"])
        self.scheduler.load_state_dict(optimizer_state["scheduler"])
