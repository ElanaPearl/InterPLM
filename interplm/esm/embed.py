""" Functions for embedding protein sequences using ESM models """

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from esm import FastaBatchedDataset, pretrained
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from interplm.utils import get_device


def get_model_converter_alphabet(
    esm_model_name: str,
    corrupt: bool = False,
    truncation_seq_length: int = 1022,
    device: str | None = None,
):
    """
    Initialize ESM model, batch converter, and alphabet for protein sequence processing.

    Args:
        esm_model_name: Name of the ESM model to load
        corrupt: If True, randomly shuffle model parameters. Defaults to False.
        truncation_seq_length: Maximum sequence length before truncation. Defaults to 1022.

    Returns:
        tuple: (model, batch_converter, alphabet)
            - model: Loaded ESM model
            - batch_converter: Function to convert sequences to model inputs
            - alphabet: ESM alphabet object for token conversion
    """
    if device is None:
        device = get_device()

    _, alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    model = EsmForMaskedLM.from_pretrained(f"facebook/{esm_model_name}").to(device)
    model.eval()

    if corrupt:
        model = shuffle_individual_parameters(model)

    batch_converter = alphabet.get_batch_converter(truncation_seq_length)

    return model, batch_converter, alphabet


def shuffle_individual_parameters(model, seed=42):
    """
    Randomly shuffle all parameters within a model while preserving their shapes.
    Used for creating controlled corrupted model baselines.

    Args:
        model: PyTorch model to shuffle
        seed: Random seed for reproducibility.

    Returns:
        Model with randomly shuffled parameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for param in model.parameters():
        original_shape = param.data.shape
        flat_param = param.data.view(-1)
        shuffled_indices = torch.randperm(flat_param.nelement())
        shuffled_param = flat_param[shuffled_indices]
        param.data = shuffled_param.view(original_shape)

    return model


def embed_list_of_prot_seqs(
    protein_seq_list: List[str],
    esm_model_name: str,
    layer: int,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = None,
    device: torch.device = None,
    corrupt: bool = False,
) -> List[np.ndarray]:
    """
    Generate ESM embeddings for a list of protein sequences in batches.

    Args:
        protein_seq_list: List of protein sequences to embed
        esm_model_name: Name of the ESM model to use
        layer: Which transformer layer to extract embeddings from
        toks_per_batch Maximum tokens per batch. Defaults to 4096.
        truncation_seq_length: Maximum sequence length before truncation.
        device: Device to run computations on. Defaults to None.
        corrupt: If True, use corrupted model parameters. Defaults to False.

    Returns:
        List of embedding arrays, one per input sequence
    """
    if device is None:
        device = get_device()

    # Load ESM model
    model, batch_converter, alphabet = get_model_converter_alphabet(
        esm_model_name, corrupt, truncation_seq_length, device=device
    )
    # Create FastaBatchedDataset
    labels = [f"protein_{i}" for i in range(len(protein_seq_list))]
    dataset = FastaBatchedDataset(labels, protein_seq_list)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_converter,
        batch_sampler=batches,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Processing {len(dataset):,} sequences")
    total_tokens = 0
    all_embeddings = [None] * len(protein_seq_list)  # Pre-allocate list

    for labels, strs, toks in tqdm(data_loader, desc="Processing batches"):
        toks = toks.to(device)

        with torch.no_grad():
            results = model(
                toks,
                attention_mask=(toks != alphabet.padding_idx),
                output_hidden_states=True,
            )
            embeddings = results.hidden_states[layer]

        # Remove padding and special tokens, and store in the correct position
        for i, (label, seq) in enumerate(zip(labels, strs)):
            seq_len = len(seq)
            # Extract original index from label
            seq_idx = int(label.split("_")[1])
            all_embeddings[seq_idx] = embeddings[i, 1 : seq_len + 1]

        total_tokens += embeddings.shape[0] * embeddings.shape[1]

    print(f"Processed {total_tokens:,} tokens in total")

    # Verify that all sequences have been processed
    assert all(
        emb is not None for emb in all_embeddings
    ), "Some sequences were not processed"

    return all_embeddings


def embed_single_sequence(
    sequence: str, model_name: str, layer: int, device: torch.device = None
) -> torch.Tensor:
    """
    Embed a single protein sequence using ESM model.

    This method is optimized for quick, individual sequence processing, making it
    ideal for interactive applications like dashboards. Unlike batch processing
    methods, it doesn't use FastaBatchedDataset or complex data loading,
    making it more suitable for concurrent user queries.

    Args:
        sequence: Protein sequence string to embed
        model_name: Name of the ESM model to use
        layer: Which transformer layer to extract embeddings from
        device: Computation device.

    Returns:
        Embedding tensor for the sequence, with shape
            (sequence_length, embedding_dimension)
    """
    if device is None:
        device = get_device()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
    model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}")
    model = model.to(device)
    model.eval()

    # Tokenize sequence
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get embeddings from specified layer
        embeddings = outputs.hidden_states[layer]
        # Remove batch dimension and special tokens
        embeddings = embeddings[0, 1:-1]

    return embeddings
