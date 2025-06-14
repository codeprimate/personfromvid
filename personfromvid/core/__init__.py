"""Core processing infrastructure for Person From Vid.

This module provides the main pipeline orchestrator, state management,
and temporary directory management components.
"""

from .pipeline import ProcessingPipeline, ProcessingResult, PipelineStatus
from .state_manager import StateManager
from .temp_manager import TempManager
from .video_processor import VideoProcessor
from .frame_extractor import FrameExtractor

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
