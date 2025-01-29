from pathlib import Path

import pandas as pd

# While we do find these useful to examine, we don't want to report them in the final
# metrics as they do not necessarily represent biological concepts (as each amino acid
# is just a single token).
concept_types_to_ignore = ["amino_acid"]


def identify_top_feature_per_concept(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the top feature per concept based on the maximum F1 score.

    Args:
        df: DataFrame containing F1 scores for each feature and concept

    Returns:
        DataFrame containing the top feature per concept
    """
    # Get indices of feature concept pairs for best feature per concept
    df = df[
        ~df["concept"].str.contains(
            "|".join(concept_types_to_ignore), case=False, na=False
        )
    ]
    top_feat_per_concept = df.sort_values(
        by=["f1_per_domain", "f1"], ascending=False
    ).drop_duplicates("concept")
    return top_feat_per_concept[["feature", "concept"]]


def identify_all_top_pairings(
    df: pd.DataFrame, top_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify all feature-concept pairs above a threshold F1 score.

    Args:
        df: DataFrame containing F1 scores for each feature and concept
        top_threshold: Minimum F1 score threshold for considering a pairing

    Returns:
        DataFrame containing all feature-concept pairs above threshold
    """
    df = df[
        ~df["concept"].str.contains(
            "|".join(concept_types_to_ignore), case=False, na=False
        )
    ]
    top_feat_concept_pairs = (
        df[df["f1_per_domain"] > top_threshold]
        .sort_values(["f1_per_domain", "f1"], ascending=False)
        .drop_duplicates(subset=["feature", "concept"], keep="first")
    )
    return top_feat_concept_pairs


def find_top_heldout_feat_per_concept(
    df_valid: pd.DataFrame, df_test: pd.DataFrame
) -> pd.Series:
    """
    Calculate the best F1 score per concept based on the held-out test set.

    Args:
        df_valid: DataFrame containing F1 scores for each feature and concept in the validation set
        df_test: DataFrame containing F1 scores for each feature and concept in the test set

    Returns:
        Series containing best F1 scores per concept in test set
    """
    top_feat_per_concept_valid = identify_top_feature_per_concept(df_valid)

    # Merge test set with validation top pairs to get matching feature-concept pairs
    matched_pairs = pd.merge(
        df_test, top_feat_per_concept_valid, on=["feature", "concept"], how="inner"
    )

    return matched_pairs.sort_values(
        ["f1_per_domain", "f1"], ascending=False
    ).drop_duplicates(subset="concept", keep="first")


def find_all_top_heldout_feats(
    df_valid: pd.DataFrame, df_test: pd.DataFrame, top_threshold: float = 0.5
) -> int:
    """
    Calculate the number of top feature-concept pairs in the held-out test set.

    Args:
        df_valid: DataFrame containing F1 scores for each feature and concept in the validation set
        df_test: DataFrame containing F1 scores for each feature and concept in the test set
        top_threshold: Minimum F1 score threshold for considering a pairing

    Returns:
        Number of feature-concept pairs above threshold in test set
    """
    top_feat_per_concept_valid = identify_all_top_pairings(df_valid, top_threshold)

    # Merge test set with validation top pairs to get matching feature-concept pairs
    matched_pairs = pd.merge(
        df_test,
        top_feat_per_concept_valid[["concept", "feature"]],
        on=["feature", "concept"],
        how="inner",
    )

    matched_pairs = matched_pairs[matched_pairs["f1_per_domain"] > top_threshold]
    matched_pairs = matched_pairs.sort_values(
        ["f1_per_domain", "f1"], ascending=False
    ).drop_duplicates(subset=["feature", "concept"], keep="first")
    return matched_pairs


def report_metrics(
    valid_path: Path, test_path: Path, top_threshold: float = 0.5
) -> None:
    """
    Report the best F1 scores per concept in the held-out test set.

    Args:
        valid_path: Path to validation F1 scores
        test_path: Path to test F1 scores
        top_threshold: Minimum F1 score threshold for considering a pairing
    """
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    top_feat_per_concept_path = test_path.parent / "heldout_top_pairings.csv"
    all_top_feats_path = test_path.parent / "heldout_all_top_pairings.csv"

    top_feat_per_concept = find_top_heldout_feat_per_concept(df_valid, df_test)
    top_feat_per_concept.to_csv(top_feat_per_concept_path, index=True, header=True)

    all_top_feats = find_all_top_heldout_feats(df_valid, df_test, top_threshold)
    all_top_feats.to_csv(all_top_feats_path, index=False, header=True)

    print(
        f"Compared {df_valid['feature'].nunique():,} features to {len(top_feat_per_concept):,} concepts"
    )
    print(
        f"Saved best pairings per concept to {top_feat_per_concept_path} and all top pairings to {all_top_feats_path}"
    )
    print(
        f"Average best F1 per concept in test set: {top_feat_per_concept['f1_per_domain'].mean():.3f}"
    )
    print(f"Number of top pairings in test set: {len(all_top_feats)}")


if __name__ == "__main__":
    from tap import tapify

    tapify(report_metrics)
