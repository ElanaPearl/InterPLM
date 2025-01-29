from pathlib import Path

import pandas as pd
import torch

from interplm.esm.embed import embed_list_of_prot_seqs


def df_of_prot_seqs_to_pt(
    protein_df,
    esm_name: str,
    output_path: str,
    layer: int,
    seq_col: str,
    toks_per_batch: int = 4096,
    corrupt: bool = False,
):
    """
    Compute ESM embeddings for protein sequences in a DataFrame and save to disk.

    Args:
        protein_df: DataFrame containing protein sequences
        esm_name: Name of ESM model to use
        output_path: Where to save the output tensor
        layer: Which transformer layer to extract embeddings from
        seq_col: Name of column containing sequences
        toks_per_batch: Maximum tokens per batch
        corrupt: Whether to use corrupted model parameters

    Returns:
        Tensor containing concatenated embeddings for all sequences
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Extract protein sequences from the DataFrame
    protein_seq_list = protein_df[seq_col].tolist()
    # Call embed_list_of_prot_seqs
    embeddings = embed_list_of_prot_seqs(
        protein_seq_list=protein_seq_list,
        esm_model_name=esm_name,
        layer=layer,
        toks_per_batch=toks_per_batch,
        device=device,
        corrupt=corrupt,
    )
    # Convert list of embeddings to a single tensor
    all_embeddings = torch.cat([emb for emb in embeddings], dim=0)
    # Save the tensor
    torch.save(all_embeddings, output_path)
    print(f"Embedding computation complete. Saved to {output_path}")
    return all_embeddings


def embed_uniprotkb_shard(
    input_file: Path,
    output_file: Path,
    layer: int,
    esm_name: str = "esm2_t6_8M_UR50D",
    corrupt: bool = False,
):
    """
    Embed protein sequences from a UniProtKB shard file.

    Args:
        input_file: Path to input TSV file
        output_file: Path to save output embeddings
        layer: Which transformer layer to extract embeddings from
        esm_name: Name of ESM model to use
        corrupt: Whether to use corrupted model parameters
    """
    if output_file.exists():
        print(f"{output_file} already exists, skipping...")
        return
    # make output directory if it doesn't exist
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file, sep="\t").set_index("Entry")
    print(f"Embedding {len(df)} proteins from Uniprot database using {esm_name}...")
    df_of_prot_seqs_to_pt(
        protein_df=df,
        esm_name=esm_name,
        output_path=output_file,
        layer=layer,
        seq_col="Sequence",
        corrupt=corrupt,
    )


def embed_all_shards(
    input_dir: Path,
    output_dir: Path,
    layer: int,
    esm_name: str = "esm2_t6_8M_UR50D",
    corrupt: bool = False,
):
    """
    Process and embed all UniProtKB shards in a directory.

    Args:
        input_dir: Directory containing shard folders
        output_dir: Directory to save embedded outputs
        layer: Which transformer layer to extract embeddings from
        esm_name: Name of ESM model to use
        corrupt: Whether to use corrupted model parameters
    """
    for input_file in input_dir.glob("shard_*"):
        shard = input_file.stem.split("_")[-1]
        output_file = output_dir / f"shard_{shard}.pt"
        embed_uniprotkb_shard(
            input_file=input_file / "protein_data.tsv",
            output_file=output_file,
            layer=layer,
            esm_name=esm_name,
            corrupt=corrupt,
        )


if __name__ == "__main__":
    from tap import tapify

    tapify(embed_all_shards)
