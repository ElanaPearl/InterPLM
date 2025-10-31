"""
Fixtures specific to analysis tests.

These are only available to tests in the test_analysis/ directory.
For shared fixtures, see the root tests/conftest.py.
"""

import pytest
from scipy import sparse
from typing import Dict, Tuple


@pytest.fixture
def domain_level_data() -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict]:
    """Test data for domain-level concept matching.

    Creates 3 proteins (90 AA total) with domain structure:
    - Protein 0 (35 AA): domains 1 and 3 have concept, domain 2 does not
    - Protein 1 (35 AA): domain 4 has concept
    - Protein 2 (20 AA): domain 5 for AA-level concept test

    Label matrix values:
    - Domain-level concepts: value = domain_id (1, 2, 3...) if concept present, 0 otherwise
    - AA-level concepts: value = 1 if concept present, 0 otherwise
    - 0 (or missing from sparse matrix) = concept not present

    Returns:
        (activations, labels, expected_metrics) where metrics are at threshold=0.5
    """
    n_tokens = 90
    n_features = 3
    n_concepts = 2

    # Feature activations
    activations_data = []
    activations_rows = []
    activations_cols = []

    # Feature 0: activates in domains 1, 3, and non-domain regions
    for pos in [0, 1, 2, 12, 13, 30, 31, 32]:  # Positions 12-13 have no concept (FP)
        activations_rows.append(pos)
        activations_cols.append(0)
        activations_data.append(0.8)

    # Feature 1: activates in domain 4
    for pos in range(40, 46):
        activations_rows.append(pos)
        activations_cols.append(1)
        activations_data.append(0.9)

    # Feature 2: activates in domain 5 (AA-level concept)
    for pos in range(75, 81):
        activations_rows.append(pos)
        activations_cols.append(2)
        activations_data.append(0.7)

    activations_sparse = sparse.csr_matrix(
        (activations_data, (activations_rows, activations_cols)),
        shape=(n_tokens, n_features)
    )

    # Concept labels
    labels_data = []
    labels_rows = []
    labels_cols = []

    # Concept 0 (domain-level): exists in domains 1, 3, 4
    # Domain 1 (positions 0-9)
    for pos in range(0, 10):
        labels_rows.append(pos)
        labels_cols.append(0)
        labels_data.append(1.0)

    # Domain 3 (positions 30-34)
    for pos in range(30, 35):
        labels_rows.append(pos)
        labels_cols.append(0)
        labels_data.append(3.0)

    # Domain 4 (positions 40-59)
    for pos in range(40, 60):
        labels_rows.append(pos)
        labels_cols.append(0)
        labels_data.append(4.0)

    # Concept 1 (AA-level): exists at positions 75-87
    for pos in range(75, 88):
        labels_rows.append(pos)
        labels_cols.append(1)
        labels_data.append(1.0)

    labels_sparse = sparse.csr_matrix(
        (labels_data, (labels_rows, labels_cols)),
        shape=(n_tokens, n_concepts)
    )

    # Expected metrics at threshold=0.5
    expected_metrics = {
        'concept_0_feature_0': {
            'tp': 6,  # Positions 0-2 (domain 1) + 30-32 (domain 3)
            'fp': 2,  # Positions 12-13 (no concept)
            'tp_per_domain': 2,  # Unique domains: 1 and 3
        },
        'concept_0_feature_1': {
            'tp': 6,  # Positions 40-45 (all in domain 4)
            'fp': 0,
            'tp_per_domain': 1,  # Unique domain: 4
        },
        'concept_1_feature_2': {
            'tp': 6,  # Positions 75-80 (AA-level)
            'fp': 0,
            'tp_per_domain': 0,  # Not calculated for AA-level
        },
    }

    return activations_sparse, labels_sparse, expected_metrics
