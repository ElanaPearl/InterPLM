"""
Tests for concept-feature matching (compare_activations.py).

Tests the critical TP/FP/FN counting logic, including domain-level concepts.
"""

import pytest
import numpy as np
import torch
from scipy import sparse

from interplm.analysis.concepts.compare_activations import (
    calc_metrics_sparse,
    calc_metrics_dense,
    count_unique_nonzero_sparse,
    count_unique_nonzero_dense,
)


def test_calc_metrics_sparse_simple_case():
    """Test sparse metrics with a simple known case."""
    # 3 samples (proteins/tokens), 2 features, 1 concept
    # Activations:
    #   Sample 0: feature 0=0.8, feature 1=0.2
    #   Sample 1: feature 0=0.3, feature 1=0.9
    #   Sample 2: feature 0=0.0, feature 1=0.0
    activations = sparse.csr_matrix(np.array([
        [0.8, 0.2],
        [0.3, 0.9],
        [0.0, 0.0],
    ]))

    # Labels (concept 0):
    #   Sample 0: has concept (1)
    #   Sample 1: has concept (1)
    #   Sample 2: no concept (0)
    labels = sparse.csr_matrix(np.array([
        [1.0],
        [1.0],
        [0.0],
    ]))

    thresholds = [0.5]
    is_aa_level = [True]

    tp, fp, tp_per_domain = calc_metrics_sparse(
        activations, labels, thresholds, is_aa_level
    )

    # At threshold 0.5:
    # Feature 0: activates at samples 0 (0.8 > 0.5), both have concept -> TP=1, FP=0
    # Feature 1: activates at sample 1 (0.9 > 0.5), has concept -> TP=1, FP=0

    assert tp.shape == (1, 2, 1)  # (concepts, features, thresholds)
    assert tp[0, 0, 0] == 1, f"Feature 0 TP should be 1, got {tp[0, 0, 0]}"
    assert tp[0, 1, 0] == 1, f"Feature 1 TP should be 1, got {tp[0, 1, 0]}"
    assert fp[0, 0, 0] == 0, f"Feature 0 FP should be 0, got {fp[0, 0, 0]}"
    assert fp[0, 1, 0] == 0, f"Feature 1 FP should be 0, got {fp[0, 1, 0]}"


def test_calc_metrics_sparse_with_false_positives():
    """Test that false positives are counted correctly."""
    # 4 samples, 1 feature, 1 concept
    activations = sparse.csr_matrix(np.array([
        [0.8],  # Sample 0: feature active
        [0.9],  # Sample 1: feature active
        [0.3],  # Sample 2: feature inactive (< 0.5)
        [0.7],  # Sample 3: feature active
    ]))

    # Only samples 0 and 2 have the concept
    labels = sparse.csr_matrix(np.array([
        [1.0],  # Sample 0: has concept
        [0.0],  # Sample 1: no concept
        [1.0],  # Sample 2: has concept
        [0.0],  # Sample 3: no concept
    ]))

    thresholds = [0.5]
    is_aa_level = [True]

    tp, fp, _ = calc_metrics_sparse(activations, labels, thresholds, is_aa_level)

    # Feature 0 at threshold 0.5:
    # - Activates at samples 0, 1, 3 (values > 0.5)
    # - Sample 0: has concept -> TP
    # - Sample 1: no concept -> FP
    # - Sample 3: no concept -> FP
    # TP=1, FP=2

    assert tp[0, 0, 0] == 1, f"TP should be 1, got {tp[0, 0, 0]}"
    assert fp[0, 0, 0] == 2, f"FP should be 2, got {fp[0, 0, 0]}"


def test_calc_metrics_sparse_multiple_thresholds():
    """Test that metrics change correctly with different thresholds."""
    activations = sparse.csr_matrix(np.array([
        [0.3],
        [0.6],
        [0.9],
    ]))

    labels = sparse.csr_matrix(np.array([
        [1.0],
        [1.0],
        [1.0],
    ]))

    thresholds = [0.2, 0.5, 0.8]
    is_aa_level = [True]

    tp, fp, _ = calc_metrics_sparse(activations, labels, thresholds, is_aa_level)

    # Threshold 0.2: all 3 samples activate -> TP=3
    # Threshold 0.5: samples 1,2 activate -> TP=2
    # Threshold 0.8: sample 2 activates -> TP=1

    assert tp[0, 0, 0] == 3, f"TP at threshold 0.2 should be 3, got {tp[0, 0, 0]}"
    assert tp[0, 0, 1] == 2, f"TP at threshold 0.5 should be 2, got {tp[0, 0, 1]}"
    assert tp[0, 0, 2] == 1, f"TP at threshold 0.8 should be 1, got {tp[0, 0, 2]}"


def test_calc_metrics_dense_matches_sparse():
    """Test that dense and sparse implementations give same results."""
    # Create test data
    activations_dense = torch.tensor([
        [0.8, 0.2, 0.5],
        [0.3, 0.9, 0.7],
        [0.6, 0.1, 0.4],
    ], dtype=torch.float32)

    labels_dense = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Convert to sparse
    activations_sparse = sparse.csr_matrix(activations_dense.numpy())
    labels_sparse = sparse.csr_matrix(labels_dense)

    thresholds = [0.5, 0.7]
    is_aa_level = [True, True]

    # Calculate with both methods
    tp_sparse, fp_sparse, tpd_sparse = calc_metrics_sparse(
        activations_sparse, labels_sparse, thresholds, is_aa_level
    )

    tp_dense, fp_dense, tpd_dense = calc_metrics_dense(
        activations_dense, labels_dense, thresholds, is_aa_level
    )

    # Should match
    np.testing.assert_array_almost_equal(tp_sparse, tp_dense, decimal=5)
    np.testing.assert_array_almost_equal(fp_sparse, fp_dense, decimal=5)
    np.testing.assert_array_almost_equal(tpd_sparse, tpd_dense, decimal=5)


def test_count_unique_nonzero_sparse():
    """Test counting unique non-zero values in sparse matrix columns."""
    # Create a matrix where each column has different unique non-zero values
    # Column 0: values [1.0, 1.0, 2.0] -> 2 unique non-zero values
    # Column 1: values [3.0, 3.0, 3.0] -> 1 unique non-zero value
    # Column 2: values [0.0, 4.0, 5.0] -> 2 unique non-zero values

    matrix = sparse.csr_matrix(np.array([
        [1.0, 3.0, 0.0],
        [1.0, 3.0, 4.0],
        [2.0, 3.0, 5.0],
    ]))

    counts = count_unique_nonzero_sparse(matrix)

    assert len(counts) == 3
    assert counts[0] == 2, f"Column 0 should have 2 unique values, got {counts[0]}"
    assert counts[1] == 1, f"Column 1 should have 1 unique value, got {counts[1]}"
    assert counts[2] == 2, f"Column 2 should have 2 unique values, got {counts[2]}"


def test_count_unique_nonzero_dense():
    """Test counting unique non-zero values in dense tensor columns."""
    matrix = torch.tensor([
        [1.0, 3.0, 0.0],
        [1.0, 3.0, 4.0],
        [2.0, 3.0, 5.0],
    ])

    counts = count_unique_nonzero_dense(matrix)

    assert len(counts) == 3
    assert counts[0] == 2, f"Column 0 should have 2 unique values, got {counts[0]}"
    assert counts[1] == 1, f"Column 1 should have 1 unique value, got {counts[1]}"
    assert counts[2] == 2, f"Column 2 should have 2 unique values, got {counts[2]}"


def test_domain_level_tp_counting(domain_level_data):
    """Test domain-level TP counting with the fixture."""
    activations_sparse, labels_sparse, expected = domain_level_data

    thresholds = [0.5]
    # Concept 0 is domain-level, concept 1 is AA-level
    is_aa_level = [False, True]

    tp, fp, tp_per_domain = calc_metrics_sparse(
        activations_sparse, labels_sparse, thresholds, is_aa_level
    )

    # Check domain-level metrics for concept 0
    # From fixture: concept_0_feature_0 should have tp_per_domain=2
    concept_0_feature_0_tpd = tp_per_domain[0, 0, 0]
    expected_tpd = expected['concept_0_feature_0']['tp_per_domain']

    assert concept_0_feature_0_tpd == expected_tpd, \
        f"Concept 0, Feature 0: expected tp_per_domain={expected_tpd}, got {concept_0_feature_0_tpd}"

    # Check AA-level TP
    concept_0_feature_0_tp = tp[0, 0, 0]
    expected_tp = expected['concept_0_feature_0']['tp']

    assert concept_0_feature_0_tp == expected_tp, \
        f"Concept 0, Feature 0: expected tp={expected_tp}, got {concept_0_feature_0_tp}"


def test_domain_level_with_aa_level_concept(domain_level_data):
    """Test that AA-level concepts have tp_per_domain == tp."""
    activations_sparse, labels_sparse, expected = domain_level_data

    thresholds = [0.5]
    is_aa_level = [False, True]  # Concept 1 is AA-level

    tp, fp, tp_per_domain = calc_metrics_sparse(
        activations_sparse, labels_sparse, thresholds, is_aa_level
    )

    # For AA-level concept (concept 1), tp_per_domain should equal tp
    # Because domain-level counting is only done for non-AA-level concepts
    # Actually, looking at the code, for AA-level concepts tp_per_domain stays 0

    # Check concept 1 (AA-level)
    concept_1_feature_2_tp = tp[1, 2, 0]
    concept_1_feature_2_tpd = tp_per_domain[1, 2, 0]

    expected_tp = expected['concept_1_feature_2']['tp']
    assert concept_1_feature_2_tp == expected_tp, \
        f"AA-level concept should have tp={expected_tp}, got {concept_1_feature_2_tp}"

    # For AA-level concepts, tp_per_domain should be 0 (not calculated)
    assert concept_1_feature_2_tpd == 0, \
        f"AA-level concept should have tp_per_domain=0, got {concept_1_feature_2_tpd}"


def test_empty_activations():
    """Test handling of empty activation matrices."""
    activations = sparse.csr_matrix((5, 3))  # All zeros
    labels = sparse.csr_matrix(np.array([
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
    ]))

    thresholds = [0.5]
    is_aa_level = [True]

    tp, fp, tp_per_domain = calc_metrics_sparse(
        activations, labels, thresholds, is_aa_level
    )

    # No activations, so all metrics should be 0
    assert np.all(tp == 0), "TP should be all zeros for empty activations"
    assert np.all(fp == 0), "FP should be all zeros for empty activations"


def test_activation_exactly_at_threshold():
    """Test that activations exactly equal to threshold are NOT counted (uses > not >=)."""
    # Critical edge case: threshold comparison uses strictly greater (>)
    activations = sparse.csr_matrix(np.array([
        [0.5],  # Exactly at threshold - NOT counted
        [0.6],  # Above threshold - counted
        [0.4],  # Below threshold - NOT counted
    ]))

    labels = sparse.csr_matrix(np.array([
        [1.0],
        [1.0],
        [1.0],
    ]))

    thresholds = [0.5]
    is_aa_level = [True]

    tp, fp, _ = calc_metrics_sparse(activations, labels, thresholds, is_aa_level)

    # Only 0.6 is counted (activation > threshold, not >=)
    assert tp[0, 0, 0] == 1, f"Expected TP=1 (only 0.6 > 0.5), got {tp[0, 0, 0]}"
    assert fp[0, 0, 0] == 0, "No false positives"


def test_all_false_positives():
    """Test when feature activates but concept never exists anywhere."""
    activations = sparse.csr_matrix(np.array([
        [0.8],
        [0.9],
        [0.7],
    ]))

    # No concept anywhere (all zeros)
    labels = sparse.csr_matrix(np.array([
        [0.0],
        [0.0],
        [0.0],
    ]))

    thresholds = [0.5]
    is_aa_level = [True]

    tp, fp, _ = calc_metrics_sparse(activations, labels, thresholds, is_aa_level)

    # All activations are FP
    assert tp[0, 0, 0] == 0, "TP should be 0 when concept never exists"
    assert fp[0, 0, 0] == 3, f"FP should be 3, got {fp[0, 0, 0]}"


def test_concept_exists_but_feature_never_activates():
    """Test when concept exists but feature never activates there (all FN)."""
    # Feature activates at wrong positions
    activations = sparse.csr_matrix(np.array([
        [0.8],  # Position 0: no concept
        [0.9],  # Position 1: no concept
        [0.0],  # Position 2: has concept but no activation
        [0.0],  # Position 3: has concept but no activation
    ]))

    labels = sparse.csr_matrix(np.array([
        [0.0],
        [0.0],
        [1.0],  # Concept at position 2
        [1.0],  # Concept at position 3
    ]))

    thresholds = [0.5]
    is_aa_level = [True]

    tp, fp, _ = calc_metrics_sparse(activations, labels, thresholds, is_aa_level)

    # TP = 0 (feature never activates where concept exists)
    # FP = 2 (feature activates where concept doesn't exist)
    assert tp[0, 0, 0] == 0, "TP should be 0 when feature doesn't activate where concept exists"
    assert fp[0, 0, 0] == 2, f"FP should be 2, got {fp[0, 0, 0]}"


def test_non_contiguous_domain_regions():
    """Test that non-contiguous regions with same domain ID are counted as one domain."""
    # Domain 1 appears at positions 0-2 and again at 5-7 (split by other positions)
    activations = sparse.csr_matrix(np.array([
        [0.8],  # Domain 1, region 1
        [0.8],
        [0.0],
        [0.0],
        [0.0],
        [0.8],  # Domain 1, region 2 (same domain, different location)
        [0.8],
    ]))

    labels = sparse.csr_matrix(np.array([
        [1.0],  # Domain 1, region 1
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],  # Domain 1, region 2
        [1.0],
    ]))

    thresholds = [0.5]
    is_aa_level = [False]  # Domain-level

    tp, fp, tp_per_domain = calc_metrics_sparse(
        activations, labels, thresholds, is_aa_level
    )

    # TP = 4 positions (2 from each region)
    # tp_per_domain = 1 (same domain_id=1, even though split)
    assert tp[0, 0, 0] == 4, f"Expected TP=4, got {tp[0, 0, 0]}"
    assert tp_per_domain[0, 0, 0] == 1, \
        f"Non-contiguous regions with same domain ID should count as 1 domain, got {tp_per_domain[0, 0, 0]}"
