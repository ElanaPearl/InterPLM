"""Render SAE feature activations on AlphaFold structures with PyMOL.
Run from this directory after run_highlights.py: pymol -cq render.py"""
import json
from pathlib import Path

import numpy as np
from pymol import cmd

here = Path.cwd()
STICK_TYPES = {"Active site", "Binding site", "Motif"}  # Swiss-Prot features shown as sticks

for ex in json.load(open(here / "examples.json")):
    feat = ex["feature"]
    for acc in ex["proteins"]:
        acts = np.load(here / f"{acc}_acts.npy")[:, feat]
        peak = acts.max()
        acts = acts / peak if peak > 0 else acts
        cmd.reset()
        cmd.delete("all")
        cmd.bg_color("white")
        cmd.load(str(here / f"AF-{acc}.pdb"), acc)
        cmd.alter(acc, "b=0.0")
        for i, v in enumerate(acts, start=1):
            cmd.alter(f"{acc} and resi {i}", f"b={v:.4f}")
        cmd.show_as("cartoon", acc)
        cmd.spectrum("b", "grey85_red", acc, minimum=0, maximum=1)
        cmd.set("cartoon_fancy_helices", 1)
        cmd.set("ray_opaque_background", 1)
        sel = [
            f"{f['location']['start']['value']}-{f['location']['end']['value']}"
            for f in json.load(open(here / f"{acc}.json"))["features"]
            if f["type"] in STICK_TYPES
        ]
        if sel:
            cmd.select("swissprot", f"{acc} and resi {'+'.join(sel)}")
            cmd.show("sticks", "swissprot and not hydro")
            cmd.util.cnc("swissprot")
            cmd.color("marine", "swissprot and elem C")
        cmd.orient(acc)
        name = f"{ex['name']}_{acc}"
        cmd.png(str(here / f"{name}.png"), 1600, 1200, dpi=200, ray=1)
        cmd.save(str(here / f"{name}.pse"))
        print(f"wrote {name} (peak activation {peak:.3f})")
