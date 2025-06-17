"""Data models and structures for Person From Vid.

This module provides data classes for configuration, pipeline state,
frame metadata, and AI model outputs.
"""

from .config import (
    Config,
    DeviceType,
    FrameExtractionConfig,
    HeadAngleConfig,
    JpegConfig,
    LoggingConfig,
    LogLevel,
    ModelConfig,
    ModelType,
    OutputConfig,
    OutputImageConfig,
    PngConfig,
    PoseClassificationConfig,
    ProcessingConfig,
    QualityConfig,
    StorageConfig,
    get_default_config,
    load_config,
)
from .context import ProcessingContext
from .detection_results import (
    FaceDetection,
    HeadPoseResult,
    PoseDetection,
    ProcessingTimings,
    QualityMetrics,
)
from .frame_data import (
    FrameData,
    ImageProperties,
    ProcessingStepInfo,
    SelectionInfo,
    SourceInfo,
)
from .pipeline_state import (
    PipelineState,
    PipelineStatus,
    ProcessingResult,
    StepProgress,
    VideoMetadata,
)

__all__ = [
    # Configuration
    "Config",
    "ModelConfig",
    "FrameExtractionConfig",
    "QualityConfig",
    "PoseClassificationConfig",
    "HeadAngleConfig",
    "OutputConfig",
    "OutputImageConfig",
    "PngConfig",
    "JpegConfig",
    "StorageConfig",
    "ProcessingConfig",
    "LoggingConfig",
    "ModelType",
    "LogLevel",
    "DeviceType",
    "get_default_config",
    "load_config",
    # Detection results
    "FaceDetection",
    "PoseDetection",
    "HeadPoseResult",
    "QualityMetrics",
    "ProcessingTimings",
    # Frame data
    "FrameData",
    "SourceInfo",
    "ImageProperties",
    "SelectionInfo",
    "ProcessingStepInfo",
    # Pipeline state
    "PipelineState",
    "VideoMetadata",
    "StepProgress",
    "ProcessingResult",
    "PipelineStatus",
    # Processing context
    "ProcessingContext",
]
