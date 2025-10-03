#!/usr/bin/env python
"""Create a dashboard for visualizing trained SAE features.

This script creates a dashboard cache from a trained SAE model, preparing it
for interactive visualization in the InterPLM dashboard.

Run this after training an SAE (e.g., with train_basic_sae.py) and collecting
feature activations (with collect_feature_activations.py).
"""

import os
from pathlib import Path
from typing import Optional, List
from interplm.dashboard.dashboard_cache import DashboardCache
from interplm.dashboard.protein_metadata import UniProtMetadata
from interplm.sae.dictionary import ReLUSAE


def create_dashboard(
    sae_path: Path,
    embeddings_dir: Path,
    layer: int,
    metadata_path: Path = Path("data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz"),
    dashboard_name: str = "walkthrough",
    model_name: str = "esm",
    model_type: str = "esm2_t6_8M_UR50D",
    shard_range: Optional[List[int]] = None,
    concept_enrichment_path: Optional[str] = None,
    uniprot_id_col: str = "Entry",
    protein_name_col: str = "Protein names",
    sequence_col: str = "Sequence",
):
    """
    Create a dashboard cache for visualizing SAE features.

    Args:
        sae_path: Path to trained SAE model (e.g., models/walkthrough_model/layer_4/ae.pt)
        embeddings_dir: Directory containing analysis embeddings
        layer: Layer number
        metadata_path: Path to protein metadata file
        dashboard_name: Name for the dashboard cache
        model_name: Embedder model name
        model_type: Embedder model type
        shard_range: Shard range [start, end] (inclusive) to search (default: auto-detect all shards)
        concept_enrichment_path: Optional path to concept enrichment results
        uniprot_id_col: Column name for UniProt IDs in metadata
        protein_name_col: Column name for protein names in metadata
        sequence_col: Column name for sequences in metadata
    """
    interplm_data = os.environ.get("INTERPLM_DATA", "data")

    # sae_path = Path(sae_path)
    # embeddings_dir = Path(embeddings_dir)
    # metadata_path = Path(metadata_path)

    # Check if normalized version exists, use it instead
    sae_normalized_path = sae_path.parent / "ae_normalized.pt"
    if sae_normalized_path.exists():
        sae_path = sae_normalized_path

    # Dashboard configuration
    cache_dir = Path(interplm_data) / "dashboard_cache" / dashboard_name
    layer_name = f"layer_{layer}"

    print("=" * 60)
    print("Creating Dashboard for Walkthrough SAE")
    print("=" * 60)
    print(f"SAE model: {sae_path}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Cache directory: {cache_dir}")
    print()

    # Check paths exist
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE model not found at {sae_path}. Train an SAE first (e.g., with train_basic_sae.py)!")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    # Create protein metadata from original source file
    print("Loading protein metadata...")
    protein_metadata = UniProtMetadata(
        metadata_path=metadata_path,
        uniprot_id_col=uniprot_id_col,
        protein_name_col=protein_name_col,
        sequence_col=sequence_col
    )

    # Create dashboard cache
    print(f"Creating dashboard cache at: {cache_dir}")
    cache = DashboardCache.create_dashboard_cache(
        cache_dir=cache_dir,
        model_name=model_name,
        model_type=model_type,
        protein_metadata=protein_metadata,
        overwrite=True  # Overwrite if exists
    )

    # Handle shard range
    if shard_range is None:
        # Use all shards available in the embeddings directory
        shards_to_search = []
        for shard_file in sorted(embeddings_dir.glob("shard_*.pt")):
            shard_num = int(shard_file.stem.split("_")[1])
            shards_to_search.append(shard_num)
        if not shards_to_search:
            print("\nWarning: No shard files found in embeddings directory")
            shards_to_search = [0]
    else:
        shards_to_search = list(range(shard_range[0], shard_range[1] + 1))

    # Handle concept enrichment path
    if concept_enrichment_path is not None:
        concept_enrichment_path = Path(concept_enrichment_path)
        if not concept_enrichment_path.exists():
            print(f"\nWarning: Specified concept enrichment path not found: {concept_enrichment_path}")
            concept_enrichment_path = None
    else:
        concept_enrichment_path = None

    # Add layer with SAE
    print(f"Adding layer: {layer_name}")
    cache.add_layer(
        layer_name=layer_name,
        sae_cls=ReLUSAE,
        sae_path=sae_path,
        feature_stats_dir=sae_path.parent,  # Pre-computed stats from collect_feature_activations.py
        aa_embeds_dir=embeddings_dir,  # For random feature sampling
        shards_to_search=shards_to_search,
        concept_enrichment_path=concept_enrichment_path,  # Optional concept analysis
        overwrite=True
    )

    print()
    print("=" * 60)
    print("✅ Dashboard created successfully!")
    print("=" * 60)
    print()
    print("To view the dashboard, run:")
    print(f"  streamlit run interplm/dashboard/app.py -- --cache_dir {cache_dir}")
    print()


if __name__ == "__main__":
    from tap import tapify
    tapify(create_dashboard)
