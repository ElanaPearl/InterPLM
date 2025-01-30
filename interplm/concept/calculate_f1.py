"""
Calculate and combine metrics across evaluation shards for concept analysis.
This script processes evaluation results from multiple shards and combines them
to generate overall metrics like precision, recall, and F1 scores.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tap import tapify
from tqdm import tqdm

from interplm.concept.compare_activations_to_concepts import (
    is_aa_level_concept,
    load_concept_names,
)


def load_metadata(eval_set_dir: Path) -> Dict[str, Any]:
    """Load and return evaluation set metadata from JSON file."""
    with open(eval_set_dir / "metadata.json", "r") as f:
        return json.load(f)


def calculate_metrics(
    tp: np.ndarray,
    fp: np.ndarray,
    tp_per_domain: np.ndarray,
    positive_labels: np.ndarray,
    positive_labels_per_domain: np.ndarray,
    concept_names: List[str],
    threshold_percents: List[float],
    is_aa_concept_list: List[bool],
) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1 scores for each concept-feature-threshold combination.

    Args:
        tp: True positives array (concepts x features x thresholds)
        fp: False positives array
        tp_per_domain: True positives per domain array
        positive_labels: Total positive labels per concept
        positive_labels_per_domain: Total positive labels per domain per concept
        concept_names: List of concept names
        threshold_percents: List of threshold percentages
        is_aa_concept_list: Boolean list indicating if each concept is AA-level

    Returns:
        DataFrame containing calculated metrics for each combination
    """
    results = []

    for concept_idx, concept in enumerate(concept_names):
        for feature in range(tp.shape[1]):
            for threshold_idx, threshold_pct in enumerate(threshold_percents):
                # Skip if no true positives
                if tp[concept_idx, feature, threshold_idx] == 0:
                    continue

                # Calculate tp, fp, precision and recall
                curr_tp = tp[concept_idx, feature, threshold_idx]
                curr_fp = fp[concept_idx, feature, threshold_idx]
                precision = curr_tp / (curr_tp + curr_fp)
                recall = (
                    curr_tp / positive_labels[concept_idx]
                    if positive_labels[concept_idx] > 0
                    else 0
                )

                # Calculate recall per domain for domain-level concepts or just
                # use recall if AA-level concept
                if is_aa_concept_list[concept_idx]:
                    recall_per_domain = recall
                else:
                    recall_per_domain = (
                        tp_per_domain[concept_idx, feature, threshold_idx]
                        / positive_labels_per_domain[concept_idx]
                        if positive_labels_per_domain[concept_idx] > 0
                        else 0
                    )

                # Calculate F1 scores
                f1 = calculate_f1(precision, recall)
                f1_per_domain = calculate_f1(precision, recall_per_domain)

                results.append(
                    {
                        "concept": concept,
                        "feature": feature,
                        "threshold_pct": threshold_pct,
                        "precision": precision,
                        "recall": recall,
                        "recall_per_domain": recall_per_domain,
                        "f1": f1,
                        "f1_per_domain": f1_per_domain,
                        "tp": curr_tp,
                        "fp": curr_fp,
                        "tp_per_domain": tp_per_domain[
                            concept_idx, feature, threshold_idx
                        ],
                        "is_aa_level_concept": is_aa_concept_list[concept_idx],
                    }
                )

    return pd.DataFrame(results)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    return (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )


def combine_metrics_across_shards(
    eval_res_dir: Path,
    eval_set_dir: Path,
    threshold_percents: List[float] = [0, 0.15, 0.5, 0.6, 0.8],
) -> None:
    """
    Combine metrics across multiple evaluation shards and save results.

    Args:
        eval_res_dir: Directory containing evaluation results
        eval_set_dir: Directory containing evaluation set data
        threshold_percents: List of threshold percentages to evaluate
        shards_to_eval: Optional list of specific shards to evaluate
    """
    # Load concept information
    concept_names = load_concept_names(eval_set_dir / "aa_concepts_columns.txt")
    is_aa_concept_list = [is_aa_level_concept(name) for name in concept_names]

    # Load metadata and get positive label counts
    metadata = load_metadata(eval_set_dir)
    positive_labels = np.array(metadata["n_positive_aa_per_concept"])
    positive_labels_per_domain = np.array(metadata["n_positive_domains_per_concept"])

    # Use all shards if none specified
    shards_to_eval = metadata["shard_source"]

    # Initialize total counts
    tp_total = None
    fp_total = None
    tp_per_domain_total = None

    # Combine counts from all shards
    for shard in tqdm(shards_to_eval, desc="Combining shard counts"):
        shard_data = np.load(eval_res_dir / f"shard_{shard}_counts.npz")

        # For the first shard, initialize total counts with correct shapes
        if tp_total is None:
            tp_total = np.zeros(shard_data["tp"].shape)
            fp_total = np.zeros(shard_data["fp"].shape)
            tp_per_domain_total = np.zeros(shard_data["tp_per_domain"].shape)

        tp_total += shard_data["tp"]
        fp_total += shard_data["fp"]
        tp_per_domain_total += shard_data["tp_per_domain"]

    # Calculate and save metrics
    print("Calculating F1 scores...")
    metrics_df = calculate_metrics(
        tp_total,
        fp_total,
        tp_per_domain_total,
        np.array(positive_labels).sum(axis=0),
        np.array(positive_labels_per_domain).sum(axis=0),
        concept_names,
        threshold_percents,
        is_aa_concept_list,
    )

    output_path = eval_res_dir / "concept_f1_scores.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    tapify(combine_metrics_across_shards)
