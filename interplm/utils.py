from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def get_device() -> str:
    """Get the best available device for PyTorch operations.

    Note: MPS (Apple Silicon GPU) is intentionally disabled as it has been
    found to be unreliable.

    Returns:
        "cuda" if NVIDIA GPU available, else "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    # MPS disabled - unreliable
    # elif torch.backends.mps.is_available():
    #     return "mps"
    else:
        return "cpu"


def _convert_paths_to_str(obj: Any) -> Any:
    """Recursively convert Path objects to strings in a nested structure."""
    if isinstance(obj, dict):
        return {k: _convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_str(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif is_dataclass(obj):
        return _convert_paths_to_str(asdict(obj))
    return obj

def convert_arrays_to_lists(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists in a nested structure."""
    if isinstance(obj, dict):
        return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def convert_numpy_ints(obj: Any) -> Any:
    """Recursively convert numpy ints to ints in a nested structure."""
    if isinstance(obj, dict):
        return {k: convert_numpy_ints(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_ints(v) for v in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    return obj

