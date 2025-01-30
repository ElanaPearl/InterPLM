"""
Compares each feature to each concept across all proteins in an evaluation set. Calculates
metrics for each shard individualy (tp, fp, tp_per_domain) and saves the results to disk.
These metrics need to be combined across all shards to get the final metrics for the evaluation set.

Because the number of comparisons can become very large, and both the feature activations
and the concept labels are sparse, we use sparse matrix operations to calculate the metrics
and this really speeds things up.

However, the neuron activations are actually quite dense so running the sparse calculations
on neurons disguising as SAE features via identity SAEs is quite slow. To address that, we
use a dense implementation for the neuron activations that can be set via the is_sparse flag.
"""

import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from interplm.concept.uniprotkb_concept_constants import is_aa_level_concept
from interplm.sae.dictionary import AutoEncoder
from interplm.sae.inference import get_sae_feats_in_batches, load_model


def count_unique_nonzero_sparse(
    matrix: Union[np.ndarray, sparse.spmatrix]
) -> List[int]:
    """
    Count unique non-zero values in each column of a sparse matrix.

    Args:
        matrix: Input matrix, either as a NumPy array or a SciPy sparse matrix.
               Will be converted to sparse CSC format if not already sparse.

    Returns:
        List of integers where each element represents the count of unique
        non-zero values in the corresponding column.
    """
    # Convert input to CSC (Compressed Sparse Column) format if not already sparse
    if not sparse.issparse(matrix):
        matrix = sparse.csc_matrix(matrix)
    else:
        # Ensure matrix is in CSC format for efficient column access
        matrix = matrix.tocsc()

    unique_counts = []
    # Iterate through each column
    for i in range(matrix.shape[1]):
        # Extract the current column
        col = matrix.getcol(i)
        # Count unique values:
        # 1. Get the non-zero data values in the column
        # 2. Convert to set to get unique values
        # 3. Subtract 1 if 0 is present in the data
        # Note: col.data contains only explicitly stored values
        unique_counts.append(len(set(col.data)) - (0 in col.data))

    return unique_counts


def calc_metrics_sparse(
    sae_feats_sparse: sparse.spmatrix,
    per_token_labels_sparse: sparse.spmatrix,
    threshold_percents: List[float],
    is_aa_level_concept_list: List[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate various metrics for sparse feature matrices across different thresholds.

    Args:
        sae_feats_sparse: Sparse matrix of features (samples x features)
        per_token_labels_sparse: Sparse matrix of labels (samples x concepts)
        threshold_percents: List of threshold values to evaluate
        is_aa_level_concept_list: Boolean flags indicating if each concept is AA-level

    Returns:
        Tuple containing:
        - tp: True positives array (concepts x features x thresholds)
        - fp: False positives array (concepts x features x thresholds)
        - tp_per_domain: True positives per domain array (concepts x features x thresholds)
    """
    # Get dimensions from input matrices
    _, n_features = sae_feats_sparse.shape
    n_concepts = per_token_labels_sparse.shape[1]
    n_thresholds = len(threshold_percents)
    per_feat_adjusted_thresholds = threshold_percents

    # Initialize arrays to store results
    tp = np.zeros((n_concepts, n_features, n_thresholds))
    fp = np.zeros((n_concepts, n_features, n_thresholds))
    tp_per_domain = np.zeros((n_concepts, n_features, n_thresholds))

    # Convert matrices to appropriate sparse formats for efficient operations
    sae_feats_sparse = sae_feats_sparse.tocsr()
    per_token_labels_sparse = per_token_labels_sparse.tocsc()

    # Iterate through each threshold
    for threshold_idx in range(n_thresholds):
        threshold = per_feat_adjusted_thresholds[threshold_idx]

        # Binarize features based on threshold
        sae_feats_binarized = sae_feats_sparse.copy()
        sae_feats_binarized.data = (sae_feats_binarized.data > threshold).astype(int)
        sae_feats_binarized.eliminate_zeros()

        # Calculate metrics for each concept
        for concept_idx in range(n_concepts):
            concept_labels = per_token_labels_sparse[:, concept_idx]

            # Calculate true positives
            tp_sparse = sae_feats_binarized.multiply(concept_labels > 0)
            tp[concept_idx, :, threshold_idx] = np.asarray(
                tp_sparse.sum(axis=0)
            ).ravel()

            # Calculate false positives
            fp_sparse = (
                sae_feats_binarized.multiply(concept_labels != 0).multiply(-1)
                + sae_feats_binarized
            )
            fp[concept_idx, :, threshold_idx] = np.asarray(
                fp_sparse.sum(axis=0)
            ).ravel()

            # Calculate domain-specific metrics for non-AA-level concepts
            if not is_aa_level_concept_list[concept_idx]:
                tp_per_domain[concept_idx, :, threshold_idx] = (
                    count_unique_nonzero_sparse(
                        sae_feats_binarized.multiply(concept_labels)
                    )
                )

    return tp, fp, tp_per_domain


def count_unique_nonzero_dense(matrix: torch.Tensor) -> List[int]:
    """
    Count unique non-zero values in each column of a dense matrix.

    Args:
        matrix: Dense PyTorch tensor to analyze

    Returns:
        List of counts of unique non-zero values for each column
    """
    # Initialize list to store counts
    unique_counts = []

    # Iterate through each column
    for col in range(matrix.shape[1]):
        # Get unique values in the column
        unique_values = torch.unique(matrix[:, col])
        # Count how many unique values are non-zero
        count = torch.sum(unique_values != 0).item()
        unique_counts.append(count)

    return unique_counts


def calc_metrics_dense(
    sae_feats: torch.Tensor,
    per_token_labels_sparse: Union[np.ndarray, sparse.spmatrix],
    threshold_percents: List[float],
    is_aa_level_concept: List[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate metrics for dense feature matrices

    Args:
        sae_feats: Dense tensor of features from SAE
        per_token_labels_sparse: Label matrix in sparse format
        threshold_percents: List of threshold values to evaluate
        is_aa_level_concept: Boolean flags indicating if each concept is AA-level

    Returns:
        Tuple of numpy arrays (tp, fp, tp_per_domain) containing metrics
    """
    # Set up GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert labels to dense tensor and move to appropriate device
    per_token_labels = torch.tensor(
        per_token_labels_sparse.astype(np.float32), device=device
    )

    # Get dimensions from input tensors
    _, n_features = sae_feats.shape
    n_concepts = per_token_labels.shape[1]
    n_thresholds = len(threshold_percents)

    # Convert thresholds to tensor and move to device
    per_feat_adjusted_thresholds = torch.tensor(
        threshold_percents, dtype=torch.float32, device=device
    )

    # Initialize result tensors on device
    tp = torch.zeros((n_concepts, n_features, n_thresholds), device=device)
    fp = torch.zeros((n_concepts, n_features, n_thresholds), device=device)
    tp_per_domain = torch.zeros((n_concepts, n_features, n_thresholds), device=device)

    # Calculate metrics for each threshold
    for threshold_idx in range(n_thresholds):
        threshold = per_feat_adjusted_thresholds[threshold_idx]

        # Binarize features based on threshold
        sae_feats_binarized = (sae_feats > threshold).float()

        # Calculate metrics for each concept
        for concept_idx in range(n_concepts):
            concept_labels = per_token_labels[:, concept_idx].unsqueeze(1)

            # Calculate true positives and false positives
            tp[concept_idx, :, threshold_idx] = (
                sae_feats_binarized * (concept_labels > 0)
            ).sum(dim=0)
            fp[concept_idx, :, threshold_idx] = (
                sae_feats_binarized * (concept_labels != 0) * (-1) + sae_feats_binarized
            ).sum(dim=0)

            # Calculate domain-specific metrics for non-AA-level concepts
            if not is_aa_level_concept[concept_idx]:
                tp_per_domain[concept_idx, :, threshold_idx] = torch.tensor(
                    count_unique_nonzero_dense(sae_feats_binarized * concept_labels),
                    device=device,
                )

    # Convert results back to numpy arrays on CPU
    return tp.cpu().numpy(), fp.cpu().numpy(), tp_per_domain.cpu().numpy()


def process_shard(
    sae: AutoEncoder,
    device: torch.device,
    esm_embeddings_pt_path: str,
    per_token_labels: Union[np.ndarray, sparse.spmatrix],
    threshold_percents: List[float],
    is_aa_concept_list: List[bool],
    feat_chunk_max: int = 512,
    is_sparse: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a shard of data by splitting it into manageable chunks for feature calculation.

    Args:
        sae: Normalized SAE model
        device: PyTorch device to use for computation
        esm_embeddings_pt_path: Path to ESM embeddings file
        per_token_labels: Label matrix
        threshold_percents: List of threshold values to evaluate
        is_aa_concept_list: Boolean flags indicating if each concept is AA-level
        feat_chunk_max: Maximum chunk size for feature processing
        is_sparse: Whether to use sparse matrix operations

    Returns:
        Tuple of arrays (tp, fp, tp_per_domain) containing calculated metrics
    """
    # Load embeddings to specified device
    esm_acts = torch.load(
        esm_embeddings_pt_path, map_location=device, weights_only=True
    )

    # Calculate chunking parameters
    feature_chunk_size = min(feat_chunk_max, sae.dict_size)
    total_features = sae.dict_size
    num_chunks = int(np.ceil(total_features / feature_chunk_size))
    print(f"Calculating over {total_features} features in {num_chunks} chunks")

    # Initialize result arrays
    n_concepts = per_token_labels.shape[1]
    n_thresholds = len(threshold_percents)
    n_features = sae.dict_size
    tp = np.zeros((n_concepts, n_features, n_thresholds))
    fp = np.zeros((n_concepts, n_features, n_thresholds))
    tp_per_domain = np.zeros((n_concepts, n_features, n_thresholds))

    # Convert labels to appropriate format
    per_token_labels = (
        sparse.csr_matrix(per_token_labels) if is_sparse else per_token_labels.toarray()
    )

    # Process each chunk of features
    for feature_list in tqdm(np.array_split(range(total_features), num_chunks)):
        # Get SAE features for current chunk
        sae_feats = get_sae_feats_in_batches(
            sae=sae,
            device=device,
            esm_embds=esm_acts,
            chunk_size=1024,
            feat_list=feature_list,
        )

        # Calculate metrics using either sparse or dense implementation
        if is_sparse:
            sae_feats_sparse = sparse.csr_matrix(sae_feats.cpu().numpy())
            metrics = calc_metrics_sparse(
                sae_feats_sparse,
                per_token_labels,
                threshold_percents,
                is_aa_concept_list,
            )
        else:
            metrics = calc_metrics_dense(
                sae_feats, per_token_labels, threshold_percents, is_aa_concept_list
            )

        # Update results arrays with computed metrics
        tp_subset, fp_subset, tp_per_domain_subset = metrics
        tp[:, feature_list] = tp_subset
        fp[:, feature_list] = fp_subset
        tp_per_domain[:, feature_list] = tp_per_domain_subset

    return (tp, fp, tp_per_domain)


def analyze_concepts(
    sae_dir: Path,
    esm_embds_dir: Path = Path("../../data/processed/embeddings"),
    eval_set_dir: Path = Path("../../data/processed/valid"),
    output_dir: Path = "concept_results",
    threshold_percents: List[float] = [0, 0.15, 0.5, 0.6, 0.8],
    shard: int | None = None,
    is_sparse: bool = True,
):
    """
    Analyzes concepts in protein sequences using a Sparse Autoencoder (SAE) model.

    Args:
        sae_dir (Path): Directory containing the normalized SAE model file 'ae_normalized.pt'
        esm_embds_dir (Path, optional): Directory containing ESM embeddings.
        eval_set_dir (Path, optional): Directory containing validation dataset and metadata.
        output_dir (Path, optional): Directory where results will be saved.
        threshold_percents (List[float], optional): List of threshold values for concept detection.
        shard (int | None): Specific shard number to process. Must exist in evaluation set.
        is_sparse (bool, optional): Whether to use sparse matrix operations.

    Returns:
        None: Results are saved to disk as NPZ file with following arrays:
            - tp: True positives counts
            - fp: False positives counts
            - tp_per_domain: True positives counts per domain

    Raises:
        ValueError: If normalized SAE model is not found in sae_dir
        ValueError: If specified shard is not in the evaluation set
    """

    # Load evaluation set metadata from JSON file
    with open(eval_set_dir / "metadata.json", "r") as f:
        eval_set_metadata = json.load(f)

    # Verify that the normalized SAE model exists
    if not (sae_dir / "ae_normalized.pt").exists():
        raise ValueError(f"Normalized SAE model not found in {sae_dir}")

    # Validate that the specified shard exists in the evaluation set
    if shard not in eval_set_metadata["shard_source"]:
        raise ValueError(f"Shard {shard} is not in this evaluation set")

    # Load concept names and identify amino acid level concepts
    concept_names = load_concept_names(eval_set_dir / "aa_concepts_columns.txt")
    is_aa_concept_list = [
        is_aa_level_concept(concept_name) for concept_name in concept_names
    ]

    # Load and process labels for the specified shard
    per_token_labels = sparse.load_npz(eval_set_metadata["path_to_shards"][str(shard)])
    per_token_labels = per_token_labels[
        :, eval_set_metadata["indices_of_concepts_to_keep"]
    ]

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the normalized SAE model
    sae = load_model(model_path=sae_dir / "ae_normalized.pt", device=device)

    # Process the shard and get results (true positives, false positives, and true positives per domain)
    (tp, fp, tp_per_domain) = process_shard(
        sae,
        device,
        esm_embds_dir / f"shard_{shard}.pt",
        per_token_labels,
        threshold_percents,
        is_aa_concept_list,
        feat_chunk_max=250,
        is_sparse=is_sparse,
    )

    # Create output directory if it doesn't exist and save results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"shard_{shard}_counts.npz",
        tp=tp,
        fp=fp,
        tp_per_domain=tp_per_domain,
    )


def analyze_all_shards_in_set(
    sae_dir: Path,
    esm_embds_dir: Path,
    eval_set_dir: Path,
    output_dir: Path = "concept_results",
    threshold_percents: List[float] = [0, 0.15, 0.5, 0.6, 0.8],
    is_sparse: bool = True,
):
    """Wrapper to scan calculate metrics across all shards in an evaluation set.

    Args:
        sae_dir (Path): Directory containing the normalized SAE model file 'ae_normalized.pt'
        esm_embds_dir (Path): Directory containing ESM embeddings
        eval_set_dir (Path): Directory containing validation dataset and metadata
        output_dir (Path, optional): Directory where results will be saved.
        threshold_percents (List[float], optional): List of threshold values for concept detection.
        is_sparse (bool, optional): Whether to use sparse matrix operations.

    Returns:
        None: Results for each shard are saved to disk in the output_dir

    Raises:
        FileNotFoundError: If metadata.json is not found in eval_set_dir
        ValueError: If any individual shard analysis fails (inherited from analyze_concepts)
    """
    # Load list of shards to evaluate from metadata
    with open(eval_set_dir / "metadata.json", "r") as f:
        shards_to_eval = json.load(f)["shard_source"]
        print(f"Analyzing set {eval_set_dir.stem} with {shards_to_eval} shards")

    # Process each shard sequentially
    for shard in shards_to_eval:
        analyze_concepts(
            sae_dir,
            esm_embds_dir,
            eval_set_dir,
            output_dir,
            threshold_percents,
            shard,
            is_sparse,
        )


def load_concept_names(concept_name_path: Path) -> List[str]:
    """Load concept names from a file."""
    with open(concept_name_path, "r") as f:
        return f.read().split("\n")


if __name__ == "__main__":
    from tap import tapify

    tapify(analyze_all_shards_in_set)

    # Note: If you want to split this up and run each shard individually,
    # you can do so by instead calling:
    # tapify(analyze_concepts)
