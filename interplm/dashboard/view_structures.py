import os
import tempfile
from typing import Dict, List

import numpy as np
import py3Dmol
import requests
from Bio.PDB import PDBIO, PDBList, PDBParser

from interplm.constants import PDB_DIR

aa_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def parse_pdb_line(line):
    match = line.startswith("ATOM") and (line[13:15] == "CA")
    if match:
        residue_num = line[22:26].strip()
        residue_3 = line[17:20].strip()
        x, y, z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
        return {
            "residue_num": int(residue_num) - 1,
            "residue_letter": aa_3to1.get(residue_3, "X"),
            "coords": (float(x), float(y), float(z)),
        }
    return None


def get_single_chain_pdb_structure(pdb_id: str, chain_id: str):
    pdb_file = PDBList().retrieve_pdb_file(
        pdb_id, file_format="pdb", pdir="pdbs", overwrite=True
    )
    structure = PDBParser().get_structure("tmp", pdb_file)
    structure = structure[0][chain_id]
    return structure


def get_single_chain_afdb_structure(uniprot_id: str):
    # Ensure the PDB_DIR exists
    os.makedirs(PDB_DIR, exist_ok=True)

    # Define the file path for this specific PDB
    pdb_file_path = os.path.join(PDB_DIR, f"AF-{uniprot_id}-F1-model_v4.pdb")

    # Check if the file already exists
    if not os.path.exists(pdb_file_path):
        # If it doesn't exist, download it
        afdb_path = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

        # Download the structure using requests
        try:
            response = requests.get(afdb_path)
            response.raise_for_status()  # This will not raise an exception now

            # Save the structure to the PDB_DIR
            with open(pdb_file_path, "w") as pdb_file:
                pdb_file.write(response.text)
        except requests.RequestException:
            # If there's any error in the request, return None
            return None

        # Save the structure to the PDB_DIR
        with open(pdb_file_path, "w") as pdb_file:
            pdb_file.write(response.text)

    # Parse the structure from the file
    parser = PDBParser()
    structure = parser.get_structure("tmp", pdb_file_path)
    return structure[0]


def get_pdb_info_as_string_from_afdb(uniprot_id: str):
    # Define the file path for this specific PDB
    pdb_file_path = os.path.join(PDB_DIR, f"AF-{uniprot_id}-F1-model_v4.pdb")

    # Check if the file already exists
    if not os.path.exists(pdb_file_path):
        # If it doesn't exist, download it
        afdb_path = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

        # Download the structure using requests
        try:
            response = requests.get(afdb_path)
            response.raise_for_status()  # This will not raise an exception now

            # Save the structure to the PDB_DIR
            with open(pdb_file_path, "w") as pdb_file:
                pdb_file.write(response.text)
        except requests.RequestException:
            # If there's any error in the request, return None
            print("Error downloading PDB file")
            return None

        pdb_text = response.text.split("\n")
    else:
        pdb_text = open(pdb_file_path, "r").readlines()
    residue_info = {}
    for line in pdb_text:
        res = parse_pdb_line(line)
        if res is not None:
            residue_info[res["residue_num"]] = res["coords"]
    return residue_info


def structure_to_seq(structure) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_pdb_file:
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_pdb_file.name)
        temp_pdb_file.seek(0)
        pdb_data = temp_pdb_file.read()
    return pdb_data


def default_colormap_fn(value):
    if value > 0:
        return "magenta"
    else:
        return "cyan"


def view_single_protein(
    pdb_id: str | None = None,
    chain_id: str | None = None,
    uniprot_id: str | None = None,
    values_to_color: List[float] | None = None,
    colormap_fn: callable = default_colormap_fn,
    default_color: str = "white",
    residues_to_highlight: List[int] | None = None,
    highlight_color: str = "magenta",
    pymol_params: Dict = {"width": 400, "height": 400},
) -> str:
    """
    Visualize a single protein structure with improved color assignment.
    """
    if uniprot_id is not None:
        pdb_struct = get_single_chain_afdb_structure(uniprot_id)
        chain_id = "A"
    elif pdb_id is not None and chain_id is not None:
        pdb_struct = get_single_chain_pdb_structure(pdb_id, chain_id)
    else:
        raise ValueError("Either pdb_id and chain_id or uniprot_id must be provided.")

    pdb_data = structure_to_seq(pdb_struct)
    residues = pdb_struct.get_residues()

    view = py3Dmol.view(**pymol_params)
    # view.setBackgroundColor(
    #    #"#0e1117"
    # )  # This is the streamlit dark theme background color
    view.addModel(pdb_data, "pdb")

    view.setStyle({"cartoon": {"color": default_color}})

    if values_to_color is None:
        values_to_color = [0] * len(residues)

    for res_id_in_seq, (residue, value) in enumerate(zip(residues, values_to_color)):
        color = colormap_fn(value)

        opacity = 0.95

        res_id_in_pdb = residue.id[1]
        residue_type = residue.get_resname()

        view.setStyle(
            {"chain": chain_id, "resi": res_id_in_pdb},
            {"cartoon": {"color": color, "opacity": opacity}},
        )

        if residues_to_highlight and res_id_in_seq in residues_to_highlight:
            view.addStyle(
                {"chain": chain_id, "resi": res_id_in_pdb},
                {"stick": {"color": highlight_color}},
            )
            view.addLabel(
                f"{residue_type}",  # {res_id_in_pdb}",
                {
                    "fontOpacity": 1,
                    "backgroundOpacity": 0.0,
                    "fontSize": 20,
                    "fontColor": highlight_color,
                },
                {"chain": chain_id, "resi": res_id_in_pdb},
            )

    view.zoomTo()
    return view._make_html()
