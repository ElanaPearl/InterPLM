"""Compute InterPLM SAE feature activations for three paper examples.

Downloads the UniProt record and AlphaFold structure for each protein in examples.json, embeds the
sequence with ESM-2-8M, encodes with the pretrained layer-4 InterPLM SAE, and saves
per-residue activations of the chosen feature.

Run from the repo root (needs the `interplm` env, internet access and a GPU is optional):
    python examples/highlights/run_highlights.py

Then render on any machine with PyMOL (from this directory):
    pymol -cq render.py

Outputs per protein, in this directory:
    {acc}.json            UniProt record (sequence + Swiss-Prot features)
    AF-{acc}.pdb          AlphaFold structure
    {acc}_acts.npy        (L, 10240) SAE activations for every feature
    {acc}_f{feat}.csv     residue, aa, activation of the highlighted feature
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from interplm.embedders.esm import ESM
from interplm.sae.inference import load_sae_from_hf

LAYER = 4
HERE = Path(__file__).parent

# Three features (ESM-2-8M layer 4), three proteins each; see examples.json
EXAMPLES = [
    (acc, ex["feature"], ex["desc"])
    for ex in json.load(open(HERE / "examples.json"))
    for acc in ex["proteins"]
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    # AlphaFold returns 403 without a browser-like User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    dest.write_bytes(urllib.request.urlopen(req).read())


def alphafold_pdb_url(acc: str) -> str:
    req = urllib.request.Request(
        f"https://alphafold.ebi.ac.uk/api/prediction/{acc}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    return json.load(urllib.request.urlopen(req))[0]["pdbUrl"]


def main() -> None:
    sae = load_sae_from_hf("esm2-8m", LAYER)
    sae_device = next(sae.parameters()).device
    embedder = ESM("facebook/esm2_t6_8M_UR50D")

    for acc, feat, desc in EXAMPLES:
        fetch(f"https://rest.uniprot.org/uniprotkb/{acc}.json", HERE / f"{acc}.json")
        fetch(alphafold_pdb_url(acc), HERE / f"AF-{acc}.pdb")

        seq = json.load(open(HERE / f"{acc}.json"))["sequence"]["value"]
        emb = embedder.extract_embeddings([seq], LAYER)  # (L, 320), CLS/EOS removed
        with torch.no_grad():
            acts = sae.encode(emb.to(sae_device)).cpu().numpy()  # (L, 10240)
        np.save(HERE / f"{acc}_acts.npy", acts)

        v = acts[:, feat]
        with open(HERE / f"{acc}_f{feat}.csv", "w") as f:
            f.write("residue,aa,activation\n")
            for i, (aa, a) in enumerate(zip(seq, v), start=1):
                f.write(f"{i},{aa},{a:.4f}\n")

        top = np.argsort(v)[::-1][:10] + 1
        print(f"{acc} f/{feat}: {desc}")
        print(f"  len={len(seq)} max={v.max():.3f} top residues={sorted(top.tolist())}")


if __name__ == "__main__":
    main()
