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
from .constants import QualityMethod
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
from .person import (
    BodyUnknown,
    FaceUnknown,
    Person,
    PersonQuality,
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
    # Constants
    "QualityMethod",
    # Detection results
    "FaceDetection",
    "PoseDetection",
    "HeadPoseResult",
    "QualityMetrics",
    "ProcessingTimings",
    # Person model
    "Person",
    "PersonQuality",
    "FaceUnknown",
    "BodyUnknown",
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
