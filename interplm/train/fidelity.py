"""
Calculate the loss (cross entropy) fidelity metric for a Sparse Autoencoder (SAE) trained on ESM embeddings
1. Calculates original cross entropy and cross entropy after zero-ablation.
2. Creates a function to calculate cross entropy using SAE reconstructions.
3. During each evaluation step, uses step 2 to evaluate the current SAE model
   then compares to the original and zero-ablation cross entropy to calculate
   the loss recovered metric.
"""

import numpy as np
import torch
import torch.nn.functional as F
from esm import pretrained
from nnsight import NNsight
from tqdm import tqdm
from transformers import EsmForMaskedLM

from interplm.esm.embed import shuffle_individual_parameters
from interplm.sae.intervention import get_model_output


def CE_for_orig_and_zero_ablation(
    esm_model, nnsight_model, tokenized_batches, hidden_layer_idx, device
):
    """Calculate cross entropy for original and zero-ablated outputs."""
    orig_losses, zero_losses = [], []

    for batch_tokens, batch_attn_mask in tqdm(tokenized_batches):
        batch_tokens = batch_tokens.to(device)
        batch_attn_mask = batch_attn_mask.to(device)

        orig_logits, orig_hidden = get_model_output(
            esm_model, nnsight_model, batch_tokens, batch_attn_mask, hidden_layer_idx
        )

        zero_logits, _ = get_model_output(
            esm_model,
            nnsight_model,
            batch_tokens,
            batch_attn_mask,
            hidden_layer_idx,
            torch.zeros_like(orig_hidden),
        )

        orig_losses.extend(
            calculate_cross_entropy(orig_logits, batch_tokens, batch_attn_mask)
        )
        zero_losses.extend(
            calculate_cross_entropy(zero_logits, batch_tokens, batch_attn_mask)
        )

    return np.mean(orig_losses), np.mean(zero_losses)


def CE_from_sae_recon(
    esm_model, nnsight_model, tokenized_batches, hidden_layer_idx, sae_model, device
):
    """Calculate cross entropy using SAE reconstructions."""
    sae_losses = []

    for batch_tokens, batch_attn_mask in tokenized_batches:
        batch_tokens = batch_tokens.to(device)
        batch_attn_mask = batch_attn_mask.to(device)

        _, orig_hidden = get_model_output(
            esm_model, nnsight_model, batch_tokens, batch_attn_mask, hidden_layer_idx
        )

        reconstructions = sae_model(orig_hidden)
        sae_logits, _ = get_model_output(
            esm_model,
            nnsight_model,
            batch_tokens,
            batch_attn_mask,
            hidden_layer_idx,
            reconstructions,
        )

        sae_losses.extend(
            calculate_cross_entropy(sae_logits, batch_tokens, batch_attn_mask)
        )

    return np.mean(sae_losses)


def calculate_loss_recovered(ce_autoencoder, ce_identity, ce_zero_ablation):
    """
    Calculate the loss recovered metric for a Sparse Autoencoder (SAE).

    If the recovered loss is as good as  to the original loss as possible, the
    metric will be 100%. If the recovered loss is as bad as zero-ablating, the
    metric will be 0%.

    Parameters:
    ce_autoencoder (float): Cross-entropy loss when using the SAE's reconstructions
    ce_identity (float): Cross-entropy loss when using the identity function
    ce_zero_ablation (float): Cross-entropy loss when using the zero-ablation function

    Returns:
    float: The loss recovered metric as a percentage
    """

    numerator = ce_autoencoder - ce_identity
    denominator = ce_zero_ablation - ce_identity

    # Avoid division by zero
    if np.isclose(denominator, 0):
        return 0.0

    loss_recovered = 1 - (numerator / denominator)

    # Clip the result to be between 0 and 1
    loss_recovered = np.clip(loss_recovered, 0, 1)

    # Convert to percentage
    return loss_recovered * 100


def get_loss_recovery_fn(
    esm_model_name: str,
    layer_idx: int,
    eval_seq_path: str,
    device: str,
    batch_size: int = 8,
    corrupt: bool = False,
) -> callable:
    print("Prepping loss fidelity_fn")
    # Load the ESM model and alphabet
    _, alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    model = EsmForMaskedLM.from_pretrained(f"facebook/{esm_model_name}").to(device)

    if corrupt:
        model = shuffle_individual_parameters(model)

    batch_converter = alphabet.get_batch_converter()

    # Load evaluation sequences
    with open(eval_seq_path, "r") as f:
        eval_seqs = [line.strip() for line in f]

    # Prepare data in the format expected by the batch converter
    data = [(f"protein{i}", seq) for i, seq in enumerate(eval_seqs)]

    # Pre-tokenize and create batches
    tokenized_batches = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_mask = (batch_tokens != alphabet.padding_idx).to(int)
        tokenized_batches.append((batch_tokens, batch_mask))

    nnsight_model = NNsight(model, device=device)

    orig_loss, zero_loss = CE_for_orig_and_zero_ablation(
        esm_model=model,
        nnsight_model=nnsight_model,
        tokenized_batches=tokenized_batches,
        hidden_layer_idx=layer_idx,
        device=device,
    )

    print("Finished preparing loss fidelity_fn")

    def loss_recovery_fn(sae_model) -> dict:
        sae_loss = CE_from_sae_recon(
            esm_model=model,
            sae_model=sae_model,
            nnsight_model=nnsight_model,
            tokenized_batches=tokenized_batches,
            hidden_layer_idx=layer_idx,
            device=device,
        )

        loss_recovered = calculate_loss_recovered(
            ce_autoencoder=sae_loss, ce_identity=orig_loss, ce_zero_ablation=zero_loss
        )

        print(f"Orig {orig_loss}, Zero {zero_loss}, SAE {sae_loss}")

        return {"pct_loss_recovered": loss_recovered, "CE_w_sae_patching": sae_loss}

    return loss_recovery_fn


def calculate_cross_entropy(model_output, batch_tokens, batch_attn_mask):
    """Calculate cross entropy for each sequence in batch, excluding start/end tokens."""
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 1 : length - 1]
        seq_tokens = batch_tokens[j, 1 : length - 1]
        loss = F.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses
