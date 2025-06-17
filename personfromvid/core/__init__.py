"""Core processing infrastructure for Person From Vid.

This module provides the main pipeline orchestrator, state management,
and temporary directory management components.
"""

from .frame_extractor import FrameExtractor
from .pipeline import PipelineStatus, ProcessingPipeline, ProcessingResult
from .state_manager import StateManager
from .temp_manager import TempManager
from .video_processor import VideoProcessor

__all__ = [
    # Pipeline orchestration
    "ProcessingPipeline",
    "ProcessingResult",
    "PipelineStatus",
    # State management
    "StateManager",
    # Temporary file management
    "TempManager",
    # Video processing
    "VideoProcessor",
    "FrameExtractor",
]
