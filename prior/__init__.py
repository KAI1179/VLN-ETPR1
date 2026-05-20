"""Prior project with Habitat."""

__version__ = "0.1.0"

from pathlib import Path

# Get the workspace root (parent of the 'prior' package directory)
_PACKAGE_ROOT = Path(__file__).parent
_WORKSPACE_ROOT = _PACKAGE_ROOT.parent

DATA_DIR = _WORKSPACE_ROOT / "data"
"""Data directory for the project."""

MP3D_DIR = DATA_DIR / "scene_datasets" / "mp3d"
"""MP3D dataset directory."""

R2R_DIR = DATA_DIR / "datasets" / "R2R_VLNCE_v1-3_preprocessed"
"""R2R dataset directory."""

RxR_DIR = DATA_DIR / "datasets" / "RxR_VLNCE_v0"
"""RxR dataset directory."""

VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
"""Directory to save visualizations."""

__all__ = [
    "DATA_DIR",
    "MP3D_DIR",
    "R2R_DIR",
    "RxR_DIR",
    "_WORKSPACE_ROOT",
]
