"""Person From Vid - AI-powered video frame extraction and pose categorization.

This package provides tools for analyzing video files to extract and categorize
high-quality frames containing people in specific poses and head orientations.
"""

__version__ = "1.0.0"
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
