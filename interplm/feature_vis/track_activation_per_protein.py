"""
Organizes proteins based on their maximum activation value for each feature. Both finds
proteins that have the higest activation value for each feature and finds proteins where
the maximum activation value *within that protein* is in a pre-specified quantile range.

Additionally tracks the average percent of proteins that are active for each feature, and
when a feature is active in a protein, what percent of the protein is it active in.
"""

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interplm.sae.dictionary import AutoEncoder
from interplm.sae.inference import get_sae_feats_in_batches, split_up_feature_list
from interplm.utils import get_device


class PerProteinActivationTracker:
    """
    Tracks and analyzes feature activations across proteins.

    This class maintains various statistics about how features activate across proteins:
    - Top N proteins per feature by maximum activation
    - Top N proteins per feature by percentage of activation
    - Proteins grouped by activation quantile ranges
    - Overall activation statistics across the protein dataset
    """

    def __init__(
        self,
        num_features: int,
        n_top: int,
        lower_quantile_thresholds: list = [
            (0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
        ],
    ):
        """
        Args:
            num_features: Total number of features to track
            n_top: Number of top proteins to track per feature
            lower_quantile_thresholds: List of tuples defining activation quantile ranges
        """
        self.num_features = num_features
        self.n_top = n_top

        # Initialize min-heaps to track top activations
        # Each heap stores tuples of (activation_value, protein_id)
        # Track by maximum activation
        self.max_heap = [[] for _ in range(num_features)]
        # Track by activation percentage
        self.pct_heap = [[] for _ in range(num_features)]

        # Initialize quantile tracking
        self.lower_quantile_thresholds = lower_quantile_thresholds
        self.lower_quantile_lists = [
            {thresh: set() for thresh in lower_quantile_thresholds}
            for _ in range(num_features)
        ]

        # Initialize global statistics
        self.total_proteins = 0
        self.proteins_with_activation = np.zeros(num_features)
        self.total_activation_percentage = np.zeros(num_features)

    def update(
        self, feature_activations: np.ndarray, protein_id: str, feature_ids: List
    ):
        """
        Update tracker with new protein activation data.

        Args:
            feature_activations: 2D array of activation values (amino_acids × features)
            protein_id: Identifier for the current protein
            feature_ids: List of feature indices being processed
        """
        if feature_ids is None:
            feature_ids = list(range(feature_activations.shape[1]))

        self.total_proteins += 1

        # Calculate per-feature statistics
        max_activations = feature_activations.max(
            axis=0
        )  # Maximum activation per feature
        nonzero_counts = (feature_activations > 0).sum(
            axis=0
        )  # Count of non-zero activations
        # Percentage of amino acids activated
        pct_nonzero = nonzero_counts / feature_activations.shape[0]

        # Update global statistics
        self.proteins_with_activation[feature_ids] += (nonzero_counts > 0).astype(int)
        self.total_activation_percentage[feature_ids] += pct_nonzero

        # Process each feature
        for i, feature_id in enumerate(feature_ids):
            max_activation = max_activations[i]
            pct_activation = pct_nonzero[i]

            if max_activation > 0:
                # Update top-N heaps if activation is significant
                if len(self.max_heap[feature_id]) < self.n_top:
                    heapq.heappush(
                        self.max_heap[feature_id], (max_activation, protein_id)
                    )
                    heapq.heappush(
                        self.pct_heap[feature_id], (pct_activation, protein_id)
                    )
                else:
                    # Replace lowest value if current activation is higher
                    if max_activation > self.max_heap[feature_id][0][0]:
                        heapq.heapreplace(
                            self.max_heap[feature_id], (max_activation, protein_id)
                        )
                    if pct_activation > self.pct_heap[feature_id][0][0]:
                        heapq.heapreplace(
                            self.pct_heap[feature_id], (pct_activation, protein_id)
                        )

                # Assign to appropriate quantile range
                for start_threshold, end_threshold in self.lower_quantile_lists[
                    feature_id
                ]:
                    if (
                        max_activation > start_threshold
                        and max_activation <= end_threshold
                    ):
                        self.lower_quantile_lists[feature_id][
                            (start_threshold, end_threshold)
                        ].add(protein_id)
                        break

            # Track proteins with zero activation (up to 1000 per feature)
            elif (0.0, 0.0) in self.lower_quantile_lists[feature_id] and len(
                self.lower_quantile_lists[feature_id][(0.0, 0.0)]
            ) < 1_000:
                self.lower_quantile_lists[feature_id][(0.0, 0.0)].add(protein_id)

    def get_results(self) -> Dict[str, Dict[int, List[str]]]:
        """
        Compile and return all tracking results.

        Returns:
            Dictionary containing:
            - 'max': Top proteins by maximum activation
            - 'lower_quantile': Proteins grouped by activation quantiles
            - 'pct': Top proteins by activation percentage
            - 'pct_proteins_with_activation': Percentage of proteins showing any activation
            - 'avg_pct_activated_when_present': Average activation percentage when feature is present
        """
        # Sort and convert max activation heaps to lists
        max_result = {
            i: [p for _, p in sorted(self.max_heap[i], reverse=True)]
            for i in range(self.num_features)
        }

        # Process quantile results (randomly sample 10 proteins if more are present)
        lower_quantile_results = {
            feat: {quantile: [] for quantile in self.lower_quantile_thresholds}
            for feat in range(self.num_features)
        }
        for feat in range(self.num_features):
            for quantile, quantile_res in self.lower_quantile_lists[feat].items():
                n_res = len(quantile_res)
                quantile_res = list(quantile_res)
                if n_res > 10:
                    quantile_res = np.random.choice(quantile_res, 10, replace=False)
                lower_quantile_results[feat][quantile] = quantile_res

        # Sort and convert percentage activation heaps to lists
        pct_result = {
            i: [p for _, p in sorted(self.pct_heap[i], reverse=True)]
            for i in range(self.num_features)
        }

        # Calculate global statistics
        pct_proteins_with_activation = (
            self.proteins_with_activation / self.total_proteins
        ) * 100
        avg_pct_activated_when_present = np.divide(
            self.total_activation_percentage,
            self.proteins_with_activation,
            out=np.zeros_like(self.total_activation_percentage, dtype=float),
            where=self.proteins_with_activation != 0,
        )

        return {
            "max": max_result,
            "lower_quantile": lower_quantile_results,
            "pct": pct_result,
            "pct_proteins_with_activation": pct_proteins_with_activation.tolist(),
            "avg_pct_activated_when_present": avg_pct_activated_when_present.tolist(),
        }


def find_max_examples_per_feat(
    sae: AutoEncoder,
    esm_embds_dir: Path,
    aa_metadata_dir: Path,
    n_shards: int,
    protein_id_col: str = "Entry",
):
    """
    Find proteins that maximally activate each feature in the sparse autoencoder.

    Processes protein embeddings in shards to handle large datasets efficiently.

    Args:
        sae: Trained sparse autoencoder model
        esm_embds_dir: Directory containing protein embedding shards
        aa_metadata_dir: Directory containing protein metadata
        n_shards: Number of data shards to process
        protein_id_col: Column name containing protein IDs in metadata

    Returns:
        Dictionary containing activation analysis results from PerProteinActivationTracker
    """
    total_features = sae.dict_size
    total_proteins = 0
    device = get_device()

    # Initialize tracker for all features
    tracker = PerProteinActivationTracker(total_features, n_top=20)

    # Process each shard of data
    for shard in range(n_shards):
        # Load embeddings and metadata for current shard
        esm_acts = torch.load(
            esm_embds_dir / f"shard_{shard}.pt", map_location=device, weights_only=True
        )
        uniprot_id_per_aa = pd.read_csv(
            aa_metadata_dir / f"shard_{shard}" / "aa_metadata.csv"
        )[protein_id_col]

        total_prot_in_shard = uniprot_id_per_aa.nunique()
        total_proteins += total_prot_in_shard

        # Map amino acid indices to protein IDs
        prot_id_to_idx = defaultdict(list)
        for i, prot_id in enumerate(uniprot_id_per_aa):
            prot_id_to_idx[prot_id].append(i)

        # Process features in chunks to manage memory
        for feature_list in tqdm(
            split_up_feature_list(total_features=total_features),
            desc=f"Finding max in feature chunks, uniprot shard {shard}",
        ):

            # Get SAE features for current chunk
            sae_feats = (
                get_sae_feats_in_batches(
                    sae=sae,
                    device=device,
                    esm_embds=esm_acts,
                    chunk_size=25_000,
                    feat_list=feature_list,
                )
                .cpu()
                .numpy()
            )

            # Update tracker with each protein's activations
            for prot_id, prot_idx in prot_id_to_idx.items():
                tracker.update(
                    sae_feats[prot_idx], protein_id=prot_id, feature_ids=feature_list
                )

    return tracker.get_results()
