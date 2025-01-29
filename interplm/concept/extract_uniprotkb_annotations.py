"""
Script for processing UniProtKB protein data into a format suitable for machine learning.
This script:
1. Filters and shards UniProtKB annotation data
2. Expands protein features to amino acid level
3. Converts data into sparse matrix format for efficient storage

Usage:
    python extract_uniprotkb_annotations.py --input_uniprot_path ../../data/uniprotkb.tsv.gz
                             --output_dir ../../data/processed/
                             --n_shards 5
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from tap import tapify

from interplm.concept.uniprotkb_concept_constants import (
    aa_map,
    binary_meta_cols,
    categorical_concepts,
    paired_binary_cols,
)
from interplm.concept.uniprotkb_parsing_utils import (
    process_binary_feature,
    process_categorical_feature,
    process_interaction_feature,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_sequence_features(row: pd.Series) -> pd.Series:
    """Add amino acid and position features to a protein sequence row."""
    split_sequence = list(row["Sequence"])
    row["amino_acid"] = split_sequence
    row["local_index"] = list(range(len(split_sequence)))
    return row


def analyze_categorical_features(
    df: pd.DataFrame, category: str, category_name: str, separator_name: str = "note"
) -> Tuple[int, pd.Series, List[int], pd.Series]:
    """
    Analyze categorical features to find common categories and their statistics.

    Returns:
        Tuple of (number of proteins, occurrences per protein, lengths, note counts)
    """
    non_na = df[category].dropna()
    n_per_prot = non_na.apply(lambda x: x.count(category_name)).value_counts()

    all_notes = []
    all_lengths = []

    for row in non_na:
        entries = row.split(category_name)[1:]
        for entry in entries:
            # Extract note
            note_start = entry.find(separator_name)
            note = entry[note_start + len(separator_name) + 1 :]
            note_end = note.find(";")
            note = note[:note_end].strip('"')

            if "/evidence" not in note:
                all_notes.append(note)

            try:
                # Extract length
                location = entry[1:]
                end_of_loc = location.find(";")
                location = location[:end_of_loc] if end_of_loc != -1 else location

                if ":" in location or "?" in location:
                    continue

                if ".." in location:
                    start, end = location.split("..")
                    start = int(start.strip("<"))
                    end = int(end.strip(">"))
                    all_lengths.append(end - start)
                else:
                    all_lengths.append(1)

            except Exception as e:
                logger.warning(f"Error analyzing feature length: {e}")

    return (
        len(non_na),
        n_per_prot.sort_values(ascending=False),
        all_lengths,
        pd.Series(all_notes).value_counts(),
    )


def expand_features(
    df: pd.DataFrame,
    categorical_column_options: Dict[str, List[str]],
    binary_cols: List[str],
    interaction_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expand all protein features to amino acid level.
    """
    new_columns = defaultdict(list)

    # Process categorical features
    for col, category_options in categorical_column_options.items():
        # Initialize current index for each category
        current_index = {cat: 1 for cat in category_options}
        logger.info(f"Processing categorical column: {col}")

        # Handle case where this shard has no data for this column
        if df[col].isnull().all():
            for category_option in category_options:
                new_columns[f"{col}_{category_option}"] = df["Length"].apply(
                    lambda x: [False] * x
                )

        else:
            # Each column has a short name separator (e.g. ACT_SITE for Active Site column)
            col_name = df[col].dropna().iloc[0].split(" ")[0]

            # For each row, extract all sub-categories and instead of just using 0,1 to indicate
            # presence, use a counter to keep track of the number of times a sub-category appears
            # and every amino acid within one occurance gets the same index so that later, if a
            # concept appears multiple times, we can distinguish the various instances / domains.
            for _, row in df.iterrows():
                results, current_index = process_categorical_feature(
                    row[col], col_name, category_options, row["Length"], current_index
                )
                for category_option, result in zip(category_options, results):
                    new_columns[f"{col}_{category_option}"].append(result)

    # Process binary features
    for col in binary_cols:
        current_index = 1
        logger.info(f"Processing binary column: {col}")

        col_name = df[col].dropna().iloc[0].split(" ")[0]
        for _, row in df.iterrows():
            result, current_index = process_binary_feature(
                row[col], col_name, row["Length"], current_index
            )
            new_columns[f"{col}_binary"].append(result)

    # Process interaction features
    for col in interaction_cols:
        logger.info(f"Processing interaction column: {col}")

        col_name = df[col].dropna().iloc[0].split(" ")[0]
        for _, row in df.iterrows():
            indices, _ = process_interaction_feature(row[col], col_name, row["Length"])
            new_columns[f"{col}_binary"].append(indices)

    logger.info("Combining expanded features...")
    return pd.concat([df, pd.DataFrame(new_columns)], axis=1), list(new_columns.keys())


def one_hot_encode(
    df: pd.DataFrame, column: str, mapping: Dict[str, str], include_other: bool = True
) -> pd.DataFrame:
    """
    Efficiently one-hot encode a column with an optional 'Other' category.
    """
    categories = list(mapping.keys())
    if include_other:
        categories.append("Other")

    encoding_series = pd.Categorical(
        df[column].apply(lambda x: x if x in mapping else "Other"),
        categories=categories,
    )

    return pd.concat([df, pd.get_dummies(encoding_series, prefix=column)], axis=1)


def filter_and_shard(
    input_path: Path,
    output_dir: Path,
    n_shards: int,
    min_required_instances: int = 100,
    min_protein_length: int = 1022,
) -> Dict[str, List[str]]:
    """
    Clean UniProt data and split into shards.
    First filters out proteins without structural annotations and those that are too long.
    Then for each categorical concept type, counts the number of occurances on each sub-category
    and filters out those with less than min_required_instances. Finally splits the data into
    n_shards and saves each shard to disk.
    """
    logger.info("Loading and cleaning UniProt data...")
    df = pd.read_csv(input_path, sep="\t")

    # Clean and shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[df["Length"] <= min_protein_length]
    df = df[df["AlphaFoldDB"].notnull()]
    df = df.drop_duplicates(subset=["Sequence"], keep="first")

    # Count occurances of categorical features and filter
    categorical_options = {}
    for col_name, col_shortname, col_separator in categorical_concepts:
        # For each categorical feature, count the number of instances of each sub-category
        _, _, _, notes = analyze_categorical_features(
            df, col_name, col_shortname, col_separator
        )
        notes = notes[notes >= min_required_instances]

        # Create a list of most common sub-categories to keep (and a catch-all for the rest)
        categorical_options[col_name] = [c for c in notes.keys() if c != ""] + ["any"]

    # Split into shards
    logger.info(f"Splitting data into {n_shards} shards...")
    shard_size = len(df) // n_shards
    for i in range(0, len(df), shard_size):
        shard_id = i // shard_size
        df_shard = df.iloc[i : min(i + shard_size, len(df))].reset_index(drop=True)

        shard_dir = output_dir / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        df_shard.to_csv(shard_dir / "protein_data.tsv", sep="\t", index=False)

    return categorical_options


def process_shard(
    shard_id: int,
    input_path: Path,
    output_dir: Path,
    categorical_options: Dict[str, List[str]],
    binary_cols: List[str],
    interaction_cols: List[str],
):
    """
    Process a single shard of UniProt data and converts a protein-level tsv into
    an amino acid-level sparse matrix with labels for each concept type.
    """
    output_metadata = output_dir / f"shard_{shard_id}" / "aa_metadata.csv"
    output_sparse = output_dir / f"shard_{shard_id}" / "aa_concepts.npz"

    if output_sparse.exists():
        logger.info(f"Shard {shard_id} already processed, skipping...")
        return

    logger.info(f"Processing shard {shard_id}...")
    df = pd.read_csv(input_path, sep="\t")

    # Add sequence features
    df = df.apply(add_sequence_features, axis=1)

    # Expand features
    df, new_cols = expand_features(
        df=df,
        categorical_column_options=categorical_options,
        binary_cols=binary_cols,
        interaction_cols=interaction_cols,
    )

    # Explode to amino acid level
    cols_to_expand = ["amino_acid", "local_index"] + new_cols
    df = df[["Entry"] + cols_to_expand].explode(cols_to_expand).reset_index(drop=True)

    # Clean up some of the concept names to make them more readable
    df.columns = [re.sub(r"_binary", "", col) for col in df.columns]
    df.columns = [re.sub(r" \[FT\]", "", col) for col in df.columns]

    # One-hot encode amino acids
    df = one_hot_encode(df, "amino_acid", aa_map, include_other=True)

    # Save metadata (Entry, amino acid, local index)
    metadata = df.iloc[:, :3]
    metadata.to_csv(output_metadata, index=False)

    # Save concept columns if they don't exist yet
    concept_col_file = output_dir / "uniprotkb_aa_concepts_columns.txt"
    if not concept_col_file.exists():
        columns = df.columns[3:]  # Skip metadata columns
        concept_col_file.write_text("\n".join(columns))

    # Convert features to sparse matrix
    feature_matrix = sparse.csr_matrix(df.iloc[:, 3:].astype(np.uint32))
    sparse.save_npz(output_sparse, feature_matrix)

    logger.info(f"Processed shard {shard_id}: {len(df):,} amino acids")


def main(
    input_uniprot_path: Path,
    output_dir: Path = Path("../../data/processed/"),
    n_shards: int = 5,
    min_required_instances: int = 100,
):
    """
    Process UniProt protein data into a machine learning-ready format.

    Args:
        input_uniprot_path: Path to input UniProt TSV file
        output_dir: Directory to save processed data
        n_shards: Number of shards to split data into
        min_required_instances: Minimum number of instances required for a
            concept sub-type to be included.
    """
    logger.info("Starting UniProt data processing...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean data and split into shards
    categorical_options = filter_and_shard(
        input_path=input_uniprot_path,
        output_dir=output_dir,
        n_shards=n_shards,
        min_required_instances=min_required_instances,
    )

    # Process each shard
    for shard in range(n_shards):
        process_shard(
            shard_id=shard,
            input_path=output_dir / f"shard_{shard}" / "protein_data.tsv",
            output_dir=output_dir,
            categorical_options=categorical_options,
            binary_cols=binary_meta_cols,
            interaction_cols=paired_binary_cols,
        )

    logger.info("UniProt data processing complete!")


if __name__ == "__main__":
    tapify(main)
