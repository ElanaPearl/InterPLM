import re
from logging import getLogger
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = getLogger(__name__)


def process_binary_feature(
    column_data: str, column_name: str, seq_len: int, current_index: int
) -> Tuple[List[bool], int]:
    """
    Process binary features from UniProt annotations (i.e. Helix, Turn)
    """
    indices = [False] * seq_len
    if pd.isna(column_data):
        return indices, current_index
    entries = column_data.split(f"{column_name} ")[1:]
    for entry in entries:
        index = entry.split(";")[0]
        try:
            if ":" in index or "?" in index:
                continue
            if ".." in index:
                start, end = index.split("..")
                start = start.strip("<")
                end = end.strip(">")
                start, end = int(start), int(end)
                for i in range(start - 1, min(end, seq_len)):
                    indices[i] = current_index
            else:
                idx = int(index) - 1
                if 0 <= idx < seq_len:
                    indices[idx] = current_index

            current_index += 1
        except Exception as e:
            print(f"Error processing binary column {column_name}: {e}")
            print(f"Column data: {column_data}, Index: {index}")
    return indices, current_index


def process_interaction_feature(
    column_data: str, column_name: str, seq_len: int
) -> Tuple[List[bool], List[Optional[int]]]:
    """
    Process interaction features (i.e. disulfide bonds) from UniProt annotations.
    """
    indices = [False] * seq_len
    pairs = [None] * seq_len
    if pd.isna(column_data):
        return indices, pairs
    n_pairs = 0
    entries = column_data.split(f"{column_name} ")[1:]
    for entry in entries:
        index = entry.split(";")[0]
        try:
            if ":" in index or "?" in index:
                continue
            if ".." in index:
                start, end = index.split("..")
                start = int(start.strip("<"))
                end = int(end.strip(">"))

                if 0 <= start - 1 < seq_len:
                    indices[start - 1] = True
                    pairs[start - 1] = n_pairs
                if 0 <= end - 1 < seq_len:
                    indices[end - 1] = True
                    pairs[end - 1] = n_pairs
                n_pairs += 1
            else:
                idx = int(index) - 1
                if 0 <= idx < seq_len:
                    indices[idx] = True
        except Exception as e:
            print(f"Error processing interaction column {column_name}: {e}")
            print(f"Column data: {column_data}, Index: {index}")
    return indices, pairs


def process_categorical_feature(
    column_data: str,
    column_name: str,
    category_options: Set[str],
    seq_len: int,
    current_index: Dict[str, int],
) -> tuple[list, dict]:
    """
    Process categorical features (i.e. Domain has multiple sub-categories) from UniProt annotations.

    Args:
        column_data: Raw feature data from UniProt
        column_name: Name of the feature column
        category_options: Set of valid category names
        seq_len: Length of the protein sequence
        current_index: Dictionary tracking current index for each category

    Returns:
        Tuple of (list of category indices vectors, updated current_index)
    """
    category_indices = {
        category_name: [False] * seq_len for category_name in category_options
    }
    if pd.isna(column_data):
        return [
            category_indices[category] for category in category_options
        ], current_index

    entries = column_data.split(f"{column_name} ")[1:]
    for entry in entries:
        positions_in_entry = entry.split(";")[0]
        entry_category = re.search(r'/note="([^"]+)"', entry)
        if entry_category:
            entry_category = entry_category.group(1).split(";")[0]

            if entry_category not in category_options:
                # skip this one
                continue

            indices_in_entry = []
            try:
                # Skip the undefined cases
                if ":" in positions_in_entry or "?" in positions_in_entry:
                    continue

                # Case where the position is a range
                if ".." in positions_in_entry:
                    start, end = positions_in_entry.split("..")
                    start = start.strip("<")
                    end = end.strip(">")
                    indices_in_entry = range(int(start), int(end) + 1)
                else:
                    indices_in_entry = [int(positions_in_entry)]
                for index in indices_in_entry:
                    if 0 <= index - 1 < seq_len:
                        if entry_category in category_options:
                            category_indices[entry_category][index - 1] = current_index[
                                entry_category
                            ]
                        category_indices["any"][index - 1] = current_index["any"]

                current_index[entry_category] += 1
                current_index["any"] += 1

            except Exception as e:
                print(
                    f"Error processing binary column index {positions_in_entry} with seq length {seq_len}"
                )
    return [category_indices[category] for category in category_options], current_index


def analyze_categorical_features(
    df: pd.DataFrame, category: str, category_name: str, separator_name: str = "note"
) -> Tuple[int, pd.Series, List[int], pd.Series]:
    """
    Helper function for analyzing categorical features to find common categories and their statistics.

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
