"""
Tests for protein embedders.

Tests that embedders work correctly with different PLMs.
"""

import pytest
import torch
import numpy as np

from interplm.embedders import get_embedder, ESM
from .conftest import create_mock_fasta


# Skip these tests if models aren't available (they require downloads)
pytestmark = pytest.mark.slow


# ============================================================================
# EMBEDDER CONFIGURATIONS
# To add a new embedder: add an entry here with type, model_name, layers, expected_dim
# ============================================================================

EMBEDDER_CONFIGS = {
    'esm_8m': {
        'type': 'esm',
        'model_name': 'facebook/esm2_t6_8M_UR50D',
        'layers': [1, 3],  # Layers to test
        'expected_dim': 320,
    },
    # Add more embedder configurations here
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def available_embedders():
    """Check which embedders are available (runs once per test session).

    Skips all tests if no embedders can be loaded.
    """
    available = {}
    for name, config in EMBEDDER_CONFIGS.items():
        try:
            embedder = get_embedder(config['type'], model_name=config['model_name'])
            available[name] = {
                'embedder': embedder,
                'config': config
            }
            print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Skipping {name}: {e}")

    if not available:
        pytest.skip("No embedder models available")

    return available


@pytest.fixture
def test_sequences():
    """Standard test sequences of varying lengths."""
    return [
        "MKTAYIAKQR",           # 10 residues
        "ARNDCQEGHILKMFPSTWY",  # 19 residues (all amino acids)
        "GLVM"                  # 4 residues
    ]


@pytest.fixture
def single_test_sequence():
    """Single test sequence for simple tests."""
    return "MKTAYIAKQRGGVVPGDDVSKAVTMG"  # 26 residues


# ============================================================================
# SINGLE SEQUENCE EMBEDDING
# ============================================================================

def test_single_sequence_embeddings_are_deterministic(available_embedders):
    """Test that embedding the same sequence twice produces identical results."""
    sequence = "MKTAYIAKQR"

    for name, info in available_embedders.items():
        embedder = info['embedder']
        config = info['config']
        layer = config['layers'][0]

        embeddings1 = embedder.embed_single_sequence(sequence, layer)
        embeddings2 = embedder.embed_single_sequence(sequence, layer)

        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5,
            err_msg=f"{name}: Embeddings should be deterministic")


def test_single_sequence_produces_expected_values(available_embedders):
    """Test that embedding a specific sequence produces expected reference values.

    This is a regression test to catch if model outputs change unexpectedly.
    Only checks a subset of values (first/last positions, few dimensions) as a smoke test.

    To generate reference values for new embedders:
        python tests/test_embedders/generate_reference_values.py
    """
    # Use a short, simple sequence for fast testing
    sequence = "MKTA"

    # Reference values for each embedder (generated from actual model outputs)
    # Generated using: python tests/test_embedders/generate_reference_values.py
    # Format: {embedder_name: {layer: {position: [dim_indices, expected_values]}}}
    REFERENCE_VALUES = {
        'esm_8m': {
            1: {
                'pos_0_dims_0_3': np.array([-0.33677340,  0.35784656, -0.06361206], dtype=np.float32),
                'pos_last_dims_0_3': np.array([0.32096285, 0.01496235, 0.48928824], dtype=np.float32),
            },
            3: {
                'pos_0_dims_0_3': np.array([ 0.6774302, -0.05016875, -1.1119854], dtype=np.float32),
                'pos_last_dims_0_3': np.array([ 0.23141728, -0.34035656, -0.76864666], dtype=np.float32),
            },
        },
        # Add reference values for other embedders here
    }

    for name, info in available_embedders.items():
        embedder = info['embedder']
        config = info['config']

        # Skip if no reference values defined yet
        if name not in REFERENCE_VALUES:
            pytest.skip(f"No reference values defined for {name} - run test once to generate")

        for layer in config['layers']:
            embeddings = embedder.embed_single_sequence(sequence, layer)

            # Skip if no reference for this layer
            if layer not in REFERENCE_VALUES[name]:
                continue

            layer_refs = REFERENCE_VALUES[name][layer]

            # Check first position, first 3 dimensions
            if 'pos_0_dims_0_3' in layer_refs:
                actual = embeddings[0, :3]
                expected = layer_refs['pos_0_dims_0_3']
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-3, atol=1e-4,
                    err_msg=f"{name} layer {layer}: First position values don't match reference"
                )

            # Check last position, first 3 dimensions
            if 'pos_last_dims_0_3' in layer_refs:
                actual = embeddings[-1, :3]
                expected = layer_refs['pos_last_dims_0_3']
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-3, atol=1e-4,
                    err_msg=f"{name} layer {layer}: Last position values don't match reference"
                )


# ============================================================================
# BOUNDARY TRACKING
# ============================================================================

def test_boundary_tracking_matches_sequence_lengths(available_embedders, test_sequences):
    """Test that extract_embeddings_with_boundaries returns correct token boundaries for each protein."""
    for name, info in available_embedders.items():
        embedder = info['embedder']
        config = info['config']
        layer = config['layers'][0]

        # Extract with boundaries
        result = embedder.extract_embeddings_with_boundaries(
            test_sequences, layer, batch_size=2
        )

        assert 'embeddings' in result, f"{name}: Missing 'embeddings' key"
        assert 'boundaries' in result, f"{name}: Missing 'boundaries' key"

        boundaries = result['boundaries']
        assert len(boundaries) == len(test_sequences), \
            f"{name}: Expected {len(test_sequences)} boundaries, got {len(boundaries)}"

        # Verify boundaries match sequence lengths
        for i, (start, end) in enumerate(boundaries):
            expected_length = len(test_sequences[i])
            actual_length = end - start
            assert actual_length == expected_length, \
                f"{name} sequence {i}: Boundary length {actual_length} doesn't match sequence length {expected_length}"

        # Verify boundaries are contiguous
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] == boundaries[i+1][0], \
                f"{name}: Boundaries are not contiguous at position {i}"


# ============================================================================
# FASTA FILE PROCESSING
# ============================================================================

def test_fasta_file_saves_to_disk_when_output_path_provided(available_embedders, temp_data_dir):
    """Test that embed_fasta_file saves embeddings to disk and can be loaded with correct shape."""
    # Create test FASTA
    fasta_path = temp_data_dir / "test.fasta"
    sequences = [("protein1", "MKTAYIAKQR")]  # 10 residues
    create_mock_fasta(sequences, fasta_path)

    expected_tokens = sum(len(seq) for _, seq in sequences)  # 10

    for name, info in available_embedders.items():
        embedder = info['embedder']
        config = info['config']
        layer = config['layers'][0]

        # Save to disk
        output_path = temp_data_dir / f"{name}_embeddings.pt"
        result = embedder.embed_fasta_file(fasta_path, layer, output_path=output_path, batch_size=2)

        # Should return None when saving
        assert result is None, f"{name}: Should return None when saving to file"

        # Check file was created
        assert output_path.exists(), f"{name}: Output file not created at {output_path}"

        # Verify can load and has correct full shape
        loaded = torch.load(output_path)
        expected_shape = (expected_tokens, config['expected_dim'])
        assert loaded.shape == expected_shape, \
            f"{name}: Loaded embeddings have shape {loaded.shape}, expected {expected_shape}"

        # Verify dtype is float
        assert loaded.dtype in [torch.float16, torch.float32, torch.float64], \
            f"{name}: Unexpected dtype {loaded.dtype}"