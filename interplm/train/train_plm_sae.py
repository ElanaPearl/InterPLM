"""
Script for training SAE from a cache of PLM activations and loading the custom fidelity function.
Assumes specific dataset structure for activations, as created by interp/data_processing/embed_fasta.py.
"""

import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from interplm.sae.dictionary import AutoEncoder
from interplm.train.fidelity import get_loss_recovery_fn
from interplm.train.load_sharded_acts import LazyMultiDirectoryTokenDataset
from interplm.train.trainer import StandardTrainer
from interplm.train.training import train_run
from interplm.utils import get_device

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def train_SAE_on_PLM_embeds(
    # Data paths and sources
    plm_embd_dir: Path,
    eval_seq_path: Path | None = None,
    # Core model architecture
    expansion_factor: int = 8,
    # Training configuration
    batch_size: int = 32,
    steps: int = 1_000,
    seed: int = 0,
    # Optimization parameters
    lr: float = 1e-3,
    warmup_steps: int = 50,
    resample_steps: int = 0,  # 0 to disable
    # Regularization
    l1_penalty: float = 1e-1,
    l1_annealing_pct: float = 0.05,
    # Evaluation settings
    eval_batch_size: int = 128,
    eval_steps: int = 1_000,
    # Logging and checkpointing
    save_dir: str = "models",
    log_steps: int = 100,
    save_steps: int = 50,
    max_ckpts_to_keep: int = 3,
    # Weights & Biases configuration
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "test_logging",
    wandb_name: str = "SAE",
):
    """
    Train a Sparse Autoencoder (SAE) using cached activation data from a language model.

    Args:
        # Data paths and sources
        plm_embd_dir: Directory containing cached model embeddings
        eval_seq_path: Path to sequences for fidelity evaluation, if None, fidelity evaluation is disabled

        # Core model architecture
        expansion_factor: Factor by which to expand the dictionary size relative to input dimension

        # Training configuration
        batch_size: Number of samples per training batch
        steps: Total number of training steps
        seed: Random seed for reproducibility

        # Optimization parameters
        lr: Learning rate for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        resample_steps: Steps between dictionary resampling (0 to disable)

        # Regularization
        l1_penalty: Coefficient for L1 regularization
        l1_annealing_pct: Percentage of training during which to anneal L1 penalty

        # Evaluation settings
        eval_batch_size: Batch size for evaluation
        eval_steps: Frequency of evaluation steps

        # Logging and checkpointing
        save_dir: Directory to save model checkpoints and outputs
        log_steps: Frequency of logging
        save_steps: Frequency of saving checkpoints

        # Weights & Biases configuration
        use_wandb: Whether to use Weights & Biases logging
        wandb_entity: W&B username or team name
        wandb_project: W&B project name
        wandb_name: W&B run name
    """
    device = get_device()

    def collate_fn(batch):
        return torch.stack(batch).to(device)

    # Initialize dataset and dataloader
    acts_dataset = LazyMultiDirectoryTokenDataset(plm_embd_dir)

    # Determine layer from dataset metadata
    layer = acts_dataset.datasets[0]["layer"]
    plm_name = acts_dataset.datasets[0]["plm_name"]
    print(f"Using activations from layer {layer} of {plm_name}")

    dataloader = DataLoader(
        acts_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(f"Loaded dataset with {len(acts_dataset):,} tokens")

    # Configure resampling
    if resample_steps == 0:
        resample_steps = None

    # Setup trainer configuration
    trainer = StandardTrainer(
        activation_dim=acts_dataset.d_model,
        dict_size=acts_dataset.d_model * expansion_factor,
        warmup_steps=warmup_steps,
        resample_steps=resample_steps,
        lr=lr,
        l1_penalty=l1_penalty,
        l1_annealing_pct=l1_annealing_pct,
        seed=seed,
        wandb_name=wandb_name,
        layer=layer,
        plm_name=plm_name,
        device=device,
        steps=min(steps, len(dataloader)),
    )
    print(f"Training with config: {trainer.config}")

    # Initialize fidelity function if evaluation sequences provided
    if eval_seq_path is not None:
        fidelity_fn = get_loss_recovery_fn(
            esm_model_name=plm_name,
            layer_idx=int(layer),
            eval_seq_path=eval_seq_path,
            device=device,
            batch_size=eval_batch_size,
        )
    else:
        fidelity_fn = None

    # Train the SAE
    train_run(
        # Core training components
        data=dataloader,
        trainer=trainer,
        # Evaluation settings
        fidelity_fn=fidelity_fn,
        eval_steps=eval_steps,
        # Logging and checkpointing
        save_dir=save_dir,
        log_steps=log_steps,
        save_steps=save_steps,
        max_ckpts_to_keep=3,
        # Weights & Biases configuration
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        additional_wandb_args={
            "eval_seq_path": eval_seq_path,
            "eval_steps": eval_steps,
            "batch_size": batch_size,
            "save_dir": save_dir,
        },
    )


if __name__ == "__main__":
    from tap import tapify

    tapify(train_SAE_on_PLM_embeds)
