"""Data models and structures for Person From Vid.

This module provides data classes for configuration, pipeline state,
frame metadata, and AI model outputs.
"""

from .config import (
    Config,
    ModelConfig,
    FrameExtractionConfig,
    QualityConfig,
    PoseClassificationConfig,
    HeadAngleConfig,
    OutputConfig,
    OutputImageConfig,
    PngConfig,
    JpegConfig,
    StorageConfig,
    ProcessingConfig,
    LoggingConfig,
    ModelType,
    LogLevel,
    DeviceType,
    get_default_config,
    load_config,
)

from .detection_results import (
    FaceDetection,
    PoseDetection,
    HeadPoseResult,
    QualityMetrics,
    ProcessingTimings,
)

from .frame_data import (
    FrameData,
    SourceInfo,
    ImageProperties,
    SelectionInfo,
    ProcessingStepInfo,
)

from .pipeline_state import (
    PipelineState,
    VideoMetadata,
    StepProgress,
    ProcessingResult,
    PipelineStatus,
)

from .context import ProcessingContext

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
