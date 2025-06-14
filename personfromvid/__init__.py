"""Person From Vid - AI-powered video frame extraction and pose categorization.

This package provides tools for analyzing video files to extract and categorize
high-quality frames containing people in specific poses and head orientations.
"""

import os
from pathlib import Path

def _get_version() -> str:
    """Get version from pyproject.toml file."""
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        # Get the project root directory (parent of personfromvid package)
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            return pyproject["project"]["version"]
    except Exception:
        pass
    
    # Fallback version if we can't read from pyproject.toml
    return "1.0.1"

__version__ = _get_version()
__author__ = "Person From Vid Project"
__description__ = "Extract and categorize high-quality frames containing people in specific poses from video files"

# Public API exports
from .data.config import Config, get_default_config, load_config
from .utils.exceptions import PersonFromVidError
from .utils.logging import setup_logging, get_logger

__all__ = [
    "Config",
    "get_default_config",
    "load_config",
    "PersonFromVidError",
    "setup_logging",
    "get_logger",
    "__version__",
    "__author__",
    "__description__",
]
