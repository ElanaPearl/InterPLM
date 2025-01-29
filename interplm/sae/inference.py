from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from interplm.sae.dictionary import AutoEncoder
from interplm.utils import get_device


def load_model(
    model_path: Union[str, Path], device: Optional[str] = None
) -> AutoEncoder:
    """
    Load a pretrained AutoEncoder model in inference mode.

    :param model_path: Path to the saved model state dictionary
    :param device: Target device for model computation ('cpu', 'cuda', etc.).
                  If None, automatically determines the best available device
    :return: Loaded AutoEncoder model in eval mode with gradients disabled

    This function:
    1. Loads the model state from disk
    2. Reconstructs the AutoEncoder architecture based on the saved weights
    3. Moves the model to the specified device
    4. Sets the model to evaluation mode
    5. Disables gradient computation for all parameters
    """
    if device is None:
        device = get_device()

    # Load state dict to the target device
    state_dict = torch.load(
        model_path, map_location=torch.device(device), weights_only=True
    )

    # Extract architecture dimensions from the encoder weights
    dict_size, activation_dim = state_dict["encoder.weight"].shape

    # Initialize and configure the model
    autoencoder = AutoEncoder(activation_dim, dict_size)
    autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)
    autoencoder.eval()

    # Disable gradient computation for inference
    for param in autoencoder.parameters():
        param.requires_grad = False

    return autoencoder


def encode_subset_of_feats(
    sae: AutoEncoder, esm_embds: Tensor, feat_list: List[int]
) -> Tensor:
    """
    Encode a batch of embeddings using a subset of features from the SAE.
    This is hacky but it lets you only use a subset of features for inference.

    :param sae: Trained Sparse AutoEncoder model
    :param chunk: Input tensor of embeddings to encode
    :param feat_list: List of feature indices to use from the encoder
    :return: Encoded features for the specified subset
    """
    with torch.no_grad():
        features = torch.nn.ReLU()(
            ((esm_embds - sae.bias) @ sae.encoder.weight[feat_list, :].T)
            + sae.encoder.bias[feat_list]
        )
    return features


def get_sae_feats_in_batches(
    sae: AutoEncoder,
    device: str,
    esm_embds: np.ndarray,
    chunk_size: int,
    feat_list: Optional[List[int]] = None,
) -> Tensor:
    """
    Process large embedding arrays in chunks to generate SAE features.

    :param sae: Trained Sparse AutoEncoder model
    :param device: Device to perform computations on ('cpu', 'cuda', etc.)
    :param esm_embds: NumPy array of ESM embeddings to process
    :param chunk_size: Number of embeddings to process in each batch
    :param feat_list: List of feature indices to encode. If None, uses all features
    :return: Tensor containing encoded features for all input embeddings

    This function:
    1. Defaults to using all features if feat_list is None
    2. Processes embeddings in batches to manage memory usage
    3. Shows progress using tqdm
    4. Concatenates all processed batches into a single tensor
    """
    # Use all features if none specified
    if feat_list is None:
        feat_list = list(range(sae.dict_size))

    # Convert input to tensor on specified device
    esm_embds = (
        esm_embds.to(device)
        if torch.is_tensor(esm_embds)
        else torch.tensor(esm_embds, device=device)
    )

    all_features = []

    # Process in chunks with progress bar
    for i in range(0, len(esm_embds), chunk_size):
        chunk = esm_embds[i : i + chunk_size]
        features = encode_subset_of_feats(sae, chunk, feat_list)
        all_features.append(features)

    # Combine all processed chunks
    all_features = torch.vstack(all_features)
    return all_features


def split_up_feature_list(total_features, max_feature_chunk_size: int = 2560):
    feature_chunk_size = min(max_feature_chunk_size, total_features)
    num_chunks = int(np.ceil(total_features / feature_chunk_size))

    return np.array_split(range(total_features), num_chunks)
