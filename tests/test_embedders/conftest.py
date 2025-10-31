"""
Fixtures and utilities specific to embedder tests.

These are only available to tests in the test_embedders/ directory.
For shared fixtures, see the root tests/conftest.py.
"""

import pytest
from pathlib import Path
from typing import List, Tuple


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for embeddings and annotations.

    Only used in test_embedders/ tests.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


def create_mock_fasta(sequences: List[Tuple[str, str]], output_path: Path) -> None:
    """Create a mock FASTA file with the given sequences.

    Args:
        sequences: List of (header, sequence) tuples
        output_path: Path where FASTA file should be written
    """
    with open(output_path, 'w') as f:
        for header, seq in sequences:
            f.write(f">{header}\n")
            f.write(f"{seq}\n")
