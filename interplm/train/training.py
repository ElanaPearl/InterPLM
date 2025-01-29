"""
General training script for training SAEs using the trainer
(Originally from https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/training.py)
"""

import json
import os

import torch as t
from tqdm import tqdm

import wandb
from interplm.sae.dictionary import AutoEncoder
from interplm.train.trainer import SAETrainer


def train_run(
    data,
    trainer: SAETrainer,
    fidelity_fn=None,  # This has to be defined in the script that calls this
    eval_steps=None,
    save_dir=None,
    log_steps=None,
    save_steps=None,
    max_ckpts_to_keep=3,
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    additional_wandb_args={},
):
    """
    Train a Sparse Autoencoder with configurable training parameters and logging.

    Args:
        data: Iterator providing activation data for training
        trainer: SAETrainer instance
        use_wandb: Enable Weights & Biases logging
        wandb_entity: W&B team/entity name
        wandb_project: W&B project name
        save_steps: Checkpoint saving frequency
        max_ckpts_to_keep: Maximum saved checkpoints to retain
        save_dir: Directory for saving checkpoints and configs
        log_steps: Logging frequency
        fidelity_fn: Function for computing model fidelity
        eval_steps: Evaluation frequency
        additional_wandb_args: Extra arguments for W&B logging

    The training loop:
    1. Initializes trainer and logging
    2. For each batch of activations:
        - Computes loss and updates model
        - Logs metrics if specified
        - Saves checkpoints if specified
        - Evaluates fidelity if specified
    3. Saves final model state
    """

    # Setup logging
    trainer_config = trainer.config
    steps = trainer.config["steps"]

    if log_steps is not None:
        if use_wandb:
            check_for_necessary_wandb_args(wandb_entity, wandb_project, log_steps)

            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=trainer_config.update(additional_wandb_args),
                name=trainer_config["wandb_name"],
            )
            # Process save_dir for W&B run
            if save_dir is not None:
                save_dir = save_dir.format(run=wandb.run.name)

    # Initialize save directory and export config
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        config = {"trainer": trainer.config}
        try:
            config["buffer"] = data.config
        except:
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        if save_steps is not None:
            os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
            saved_steps = set()

    # Training loop
    n_tokens_total = 0
    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            print("Stopped training because reached max specified steps")
            break

        # Logging metrics
        if log_steps is not None and step % log_steps == 0:
            log = {}
            with t.no_grad():
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # Calculate sparsity metrics
                n_nonzero_per_example = (f != 0).float().sum(dim=-1)
                l0 = n_nonzero_per_example.mean().item()
                l0_norm = (
                    n_nonzero_per_example / trainer_config["dict_size"]
                ).mean().item() * 100

                # Calculate variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log["frac_variance_explained"] = frac_variance_explained.item()

                # Check for NaN loss
                if losslog["loss"] != losslog["loss"]:
                    print("Oh no, NaN loss!!")
                    breakpoint()

                # Aggregate logging metrics
                log.update(losslog)
                log["l0"] = l0
                log["l0_pct_nonzero"] = l0_norm
                trainer_log = trainer.get_logging_parameters()
                trainer_log.update(trainer.get_extra_logging_parameters())
                for name, value in trainer_log.items():
                    log[name] = value

                # Run fidelity evaluation
                if fidelity_fn is not None and step % eval_steps == 0:
                    fidelity = fidelity_fn(sae_model=trainer.ae)
                    for k, v in fidelity.items():
                        log[k] = v

                # Log activation statistics
                log["act_mean"] = act.mean().item()
                log["act_std"] = act.std().item()
                log["reconstruction_mean"] = act_hat.mean().item()
                log["reconstruction_std"] = act_hat.std(dim=1).mean().item()
                log["tokens"] = n_tokens_total

            if use_wandb:
                wandb.log(log, step=step)

        # Save checkpoints
        if save_steps is not None and step % save_steps == 0:
            t.save(
                trainer.ae.state_dict(),
                os.path.join(save_dir, "checkpoints", f"ae_{step}.pt"),
            )
            saved_steps.add(step)

            # Maintain maximum number of checkpoints
            if len(saved_steps) > max_ckpts_to_keep:
                min_step = min(saved_steps)
                saved_steps.remove(min_step)
                os.remove(os.path.join(save_dir, "checkpoints", f"ae_{min_step}.pt"))

        # Update model
        trainer.update(step, act)
        n_tokens_total += act.shape[0]

    # Save final model state
    if save_dir is not None:
        t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))

    # Cleanup W&B
    if log_steps is not None and use_wandb:
        wandb.finish()


def check_for_necessary_wandb_args(wandb_entity, wandb_project, log_steps):
    """
    Validate required arguments for Weights & Biases logging.

    Args:
        wandb_entity: W&B team/entity name
        wandb_project: W&B project name
        log_steps: Logging frequency

    Raises:
        ValueError: If any required arguments are missing
    """
    necessary_args = {
        "wandb_entity": wandb_entity,
        "wandb_project": wandb_project,
        "log_steps": log_steps,
    }

    missing_args = [arg for arg, value in necessary_args.items() if not value]

    if missing_args:
        raise ValueError(
            "In order to log your run to wandb, you must specify the following arguments:\n"
            + "\n".join(f"* {arg}" for arg in missing_args)
        )


def check_for_optional_wandb_args(trainer_cfg):
    """
    Check for recommended optional Weights & Biases logging arguments.

    Args:
        trainer_cfg: Training configuration dictionary

    Prints a warning if recommended arguments are missing
    """
    optional_args = ["wandb_name", "layer", "plm_name"]
    missing_args = [arg for arg in optional_args if arg not in trainer_cfg]

    if missing_args:
        print(
            "Warning: You are missing the following optional arguments from trainer_cfg:\n"
            + "\n".join(f"* {arg}" for arg in missing_args)
            + "\nIt will still log your run, but these are useful for tracking purposes."
        )
