import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("INTERPLM_DATA", Path(__file__).parent.parent / "data"))
TRAINED_MODELS_DIR = Path(__file__).parent.parent / "models"
PDB_DIR = Path(os.environ.get("INTERPLM_PDB", DATA_DIR / "pdb_files"))

DASHBOARD_CACHE_DIR = DATA_DIR / "dashboard_cache"
DASHBOARD_CACHE = Path(
    os.environ.get("DASHBOARD_CACHE", DASHBOARD_CACHE_DIR / "dashboard_cache_650M.pkl")
)
