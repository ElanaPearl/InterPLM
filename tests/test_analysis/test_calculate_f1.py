"""
Tests for F1/precision/recall metric calculations.

Tests the aggregation logic and domain-level vs AA-level distinction.
"""

import pytest
import numpy as np
import pandas as pd

from interplm.analysis.concepts.calculate_f1 import (
    calculate_f1,
    calculate_metrics,
)


def test_calculate_metrics_from_counts():
    """Test the full calculate_metrics function with known TP/FP counts."""
    # Create test data:
    # 2 concepts, 3 features, 2 thresholds
    n_concepts = 2
    n_features = 3
    n_thresholds = 2

    tp = np.zeros((n_concepts, n_features, n_thresholds))
    fp = np.zeros((n_concepts, n_features, n_thresholds))
    tp_per_domain = np.zeros((n_concepts, n_features, n_thresholds))

    # Concept 0, Feature 0, Threshold 0: TP=10, FP=5
    tp[0, 0, 0] = 10
    fp[0, 0, 0] = 5
    tp_per_domain[0, 0, 0] = 3  # 3 unique domains

    # Concept 1, Feature 1, Threshold 1: TP=20, FP=10
    tp[1, 1, 1] = 20
    fp[1, 1, 1] = 10
    tp_per_domain[1, 1, 1] = 5

    # Positive labels
    positive_labels = np.array([100, 200])  # Concept 0 has 100, Concept 1 has 200
    positive_labels_per_domain = np.array([10, 20])

    concept_names = ['concept_0', 'concept_1']
    threshold_percents = [0.5, 0.7]
    is_aa_concept_list = [False, True]  # Concept 0 is domain-level

    df = calculate_metrics(
        tp, fp, tp_per_domain,
        positive_labels, positive_labels_per_domain,
        concept_names, threshold_percents, is_aa_concept_list
    )

    # Check that we got results for the non-zero TP cases
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"

    # Check concept 0, feature 0, threshold 0
    row0 = df[(df['concept'] == 'concept_0') & (df['feature'] == 0) & (df['threshold_pct'] == 0.5)]
    assert len(row0) == 1
    row0 = row0.iloc[0]

    # Precision = TP / (TP + FP) = 10 / 15 = 0.666...
    expected_precision = 10 / 15
    assert abs(row0['precision'] - expected_precision) < 1e-6

    # Recall = TP / positive_labels = 10 / 100 = 0.1
    expected_recall = 10 / 100
    assert abs(row0['recall'] - expected_recall) < 1e-6

    # Recall per domain = tp_per_domain / positive_labels_per_domain = 3 / 10 = 0.3
    expected_recall_per_domain = 3 / 10
    assert abs(row0['recall_per_domain'] - expected_recall_per_domain) < 1e-6

    # F1 = 2 * P * R / (P + R)
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert abs(row0['f1'] - expected_f1) < 1e-6


def test_calculate_metrics_filters_zero_tp():
    """Test that rows with TP=0 are filtered out."""
    tp = np.zeros((1, 2, 1))
    fp = np.zeros((1, 2, 1))
    tp_per_domain = np.zeros((1, 2, 1))

    # Only feature 0 has TP > 0
    tp[0, 0, 0] = 5
    fp[0, 0, 0] = 2

    positive_labels = np.array([100])
    positive_labels_per_domain = np.array([10])
    concept_names = ['concept_0']
    threshold_percents = [0.5]
    is_aa_concept_list = [True]

    df = calculate_metrics(
        tp, fp, tp_per_domain,
        positive_labels, positive_labels_per_domain,
        concept_names, threshold_percents, is_aa_concept_list
    )

    # Should only have 1 row (feature 0), not feature 1
    assert len(df) == 1
    assert df.iloc[0]['feature'] == 0


def test_calculate_metrics_aa_level_vs_domain_level():
    """Test that AA-level and domain-level concepts have different recall calculations."""
    tp = np.zeros((2, 1, 1))
    fp = np.zeros((2, 1, 1))
    tp_per_domain = np.zeros((2, 1, 1))

    # Both concepts have same TP/FP
    tp[0, 0, 0] = 10
    tp[1, 0, 0] = 10
    fp[0, 0, 0] = 5
    fp[1, 0, 0] = 5

    # But different tp_per_domain
    tp_per_domain[0, 0, 0] = 3  # Domain-level: 3 domains
    tp_per_domain[1, 0, 0] = 0  # AA-level: not used

    positive_labels = np.array([100, 100])
    positive_labels_per_domain = np.array([10, 10])
    concept_names = ['domain_concept', 'aa_concept']
    threshold_percents = [0.5]
    is_aa_concept_list = [False, True]  # First is domain-level, second is AA-level

    df = calculate_metrics(
        tp, fp, tp_per_domain,
        positive_labels, positive_labels_per_domain,
        concept_names, threshold_percents, is_aa_concept_list
    )

    # Both should have same precision and recall (AA-level)
    domain_row = df[df['concept'] == 'domain_concept'].iloc[0]
    aa_row = df[df['concept'] == 'aa_concept'].iloc[0]

    assert domain_row['precision'] == aa_row['precision']
    assert domain_row['recall'] == aa_row['recall']

    # But domain-level should have different recall_per_domain
    expected_recall_per_domain_domain = 3 / 10
    expected_recall_per_domain_aa = 10 / 100  # Same as recall for AA-level

    assert abs(domain_row['recall_per_domain'] - expected_recall_per_domain_domain) < 1e-6
    assert abs(aa_row['recall_per_domain'] - expected_recall_per_domain_aa) < 1e-6


def test_calculate_metrics_zero_positive_labels():
    """Test handling when positive_labels is zero (should give recall=0)."""
    tp = np.array([[[5]]])
    fp = np.array([[[2]]])
    tp_per_domain = np.array([[[0]]])

    positive_labels = np.array([0])  # No positive labels
    positive_labels_per_domain = np.array([0])
    concept_names = ['test']
    threshold_percents = [0.5]
    is_aa_concept_list = [True]

    df = calculate_metrics(
        tp, fp, tp_per_domain,
        positive_labels, positive_labels_per_domain,
        concept_names, threshold_percents, is_aa_concept_list
    )

    row = df.iloc[0]

    # Recall should be 0 when no positive labels exist
    assert row['recall'] == 0.0
