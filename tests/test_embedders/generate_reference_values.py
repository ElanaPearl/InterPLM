#!/usr/bin/env python
"""
Helper script to generate reference values for test_single_sequence_produces_expected_values.

Run this once to generate actual embedding values, then copy the output into
REFERENCE_VALUES in test_embedders.py.

Usage:
    python tests/test_embedders/generate_reference_values.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interplm.embedders import get_embedder

# This matches EMBEDDER_CONFIGS in test_embedders.py
EMBEDDER_CONFIGS = {
    'esm_8m': {
        'type': 'esm',
        'model_name': 'facebook/esm2_t6_8M_UR50D',
        'layers': [1, 3],
    },
    # Add more embedders here as needed
}

# Test sequence (must match test file)
TEST_SEQUENCE = "MKTA"


def generate_reference_values():
    """Generate reference values for all configured embedders."""
    print("Generating reference values...")
    print("=" * 60)
    print()

    reference_dict = {}

    for name, config in EMBEDDER_CONFIGS.items():
        print(f"Processing {name}...")

        try:
            embedder = get_embedder(config['type'], model_name=config['model_name'])
            reference_dict[name] = {}

            for layer in config['layers']:
                print(f"  Layer {layer}...")
                embeddings = embedder.embed_single_sequence(TEST_SEQUENCE, layer)

                # Extract first and last position, first 3 dimensions
                pos_0_vals = embeddings[0, :3]
                pos_last_vals = embeddings[-1, :3]

                reference_dict[name][layer] = {
                    'pos_0_dims_0_3': pos_0_vals,
                    'pos_last_dims_0_3': pos_last_vals,
                }

                print(f"    Position 0: {pos_0_vals}")
                print(f"    Position {len(TEST_SEQUENCE)-1}: {pos_last_vals}")

            print(f"✓ {name} complete")

        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
            continue

        print()

    # Now print formatted output to copy into test file
    print()
    print("=" * 60)
    print("Copy the following into REFERENCE_VALUES in test_embedders.py:")
    print("=" * 60)
    print()
    print("REFERENCE_VALUES = {")

    for embedder_name, layers_dict in reference_dict.items():
        print(f"    '{embedder_name}': {{")

        for layer, positions_dict in layers_dict.items():
            print(f"        {layer}: {{")

            for pos_key, values in positions_dict.items():
                # Format as numpy array
                values_str = np.array2string(values, precision=4, separator=', ', suppress_small=True)
                print(f"            '{pos_key}': np.array({values_str}, dtype=np.float32),")

            print(f"        }},")

        print(f"    }},")

    print("}")


if __name__ == "__main__":
    generate_reference_values()
