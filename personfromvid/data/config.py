"""Configuration management for Person From Vid.

This module defines configuration classes and default settings for the video processing
pipeline, with support for environment variable overrides and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict
from platformdirs import user_cache_dir

# Configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_BATCH_SIZE = 1
DEFAULT_JPEG_QUALITY = 95


class ModelType(str, Enum):
    """Supported AI model types."""

    FACE_DETECTION = "face_detection"
    POSE_ESTIMATION = "pose_estimation"
    HEAD_POSE_ESTIMATION = "head_pose_estimation"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DeviceType(str, Enum):
    """Supported computation devices."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class ModelConfig(BaseModel):
    """Configuration for AI models."""

    face_detection_model: str = Field(
        default="yolov8s-face", description="Face detection model name"
    )
    pose_estimation_model: str = Field(
        default="yolov8s-pose", description="Pose estimation model name"
    )
    head_pose_model: str = Field(
        default="sixdrepnet", description="Head pose estimation model name"
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO, description="Computation device preference"
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=1,
        le=64,
        description="Batch size for model inference",
    )
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    )

    @field_serializer("device")
    def serialize_device(self, value) -> str:
        """Serialize DeviceType enum to string for JSON."""
        if isinstance(value, DeviceType):
            return value.value
        return str(value)


class FrameExtractionConfig(BaseModel):
    """Configuration for frame extraction."""

    temporal_sampling_interval: float = Field(
        default=0.25,
        ge=0.1,
        le=2.0,
        description="Interval in seconds for temporal frame sampling",
    )
    enable_keyframe_detection: bool = Field(
        default=True, description="Enable I-frame keyframe detection"
    )
    enable_temporal_sampling: bool = Field(
        default=True, description="Enable temporal frame sampling"
    )
    max_frames_per_video: Optional[int] = Field(
        default=None,
        ge=10,
        description="Maximum frames to extract per video (None for unlimited)",
    )
    deduplication_enabled: bool = Field(
        default=True, description="Enable frame deduplication"
    )


class QualityConfig(BaseModel):
    """Configuration for image quality assessment."""

    blur_threshold: float = Field(
        default=100.0,
        ge=10.0,
        description="Minimum blur threshold (Laplacian variance)",
    )
    brightness_min: float = Field(
        default=30.0, ge=0.0, le=255.0, description="Minimum acceptable brightness"
    )
    brightness_max: float = Field(
        default=225.0, ge=0.0, le=255.0, description="Maximum acceptable brightness"
    )
    contrast_min: float = Field(
        default=20.0, ge=0.0, description="Minimum acceptable contrast"
    )
    enable_multiple_metrics: bool = Field(
        default=True, description="Use multiple quality metrics"
    )


class PoseClassificationConfig(BaseModel):
    """Configuration for pose classification."""

    standing_hip_knee_angle_min: float = Field(
        default=160.0,
        ge=120.0,
        le=180.0,
        description="Minimum hip-knee angle for standing classification",
    )
    sitting_hip_knee_angle_min: float = Field(
        default=80.0,
        ge=45.0,
        le=120.0,
        description="Minimum hip-knee angle for sitting classification",
    )
    sitting_hip_knee_angle_max: float = Field(
        default=120.0,
        ge=80.0,
        le=160.0,
        description="Maximum hip-knee angle for sitting classification",
    )
    squatting_hip_knee_angle_max: float = Field(
        default=90.0,
        ge=45.0,
        le=120.0,
        description="Maximum hip-knee angle for squatting classification",
    )
    closeup_face_area_threshold: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Minimum face area ratio for closeup detection",
    )


class HeadAngleConfig(BaseModel):
    """Configuration for head angle classification."""

    yaw_threshold_degrees: float = Field(
        default=22.5,
        ge=10.0,
        le=45.0,
        description="Yaw angle threshold for direction classification",
    )
    pitch_threshold_degrees: float = Field(
        default=22.5,
        ge=10.0,
        le=45.0,
        description="Pitch angle threshold for direction classification",
    )
    max_roll_degrees: float = Field(
        default=30.0, ge=15.0, le=60.0, description="Maximum acceptable roll angle"
    )
    profile_yaw_threshold: float = Field(
        default=67.5,
        ge=45.0,
        le=90.0,
        description="Yaw threshold for profile classification",
    )


class CloseupDetectionConfig(BaseModel):
    """Configuration for closeup detection."""

    extreme_closeup_threshold: float = Field(
        default=0.25,
        ge=0.15,
        le=0.5,
        description="Face area ratio threshold for extreme closeup classification",
    )
    closeup_threshold: float = Field(
        default=0.15,
        ge=0.08,
        le=0.3,
        description="Face area ratio threshold for closeup classification",
    )
    medium_closeup_threshold: float = Field(
        default=0.08,
        ge=0.04,
        le=0.15,
        description="Face area ratio threshold for medium closeup classification",
    )
    medium_shot_threshold: float = Field(
        default=0.03,
        ge=0.01,
        le=0.08,
        description="Face area ratio threshold for medium shot classification",
    )
    shoulder_width_threshold: float = Field(
        default=0.35,
        ge=0.2,
        le=0.6,
        description="Shoulder width ratio threshold for closeup detection",
    )
    enable_distance_estimation: bool = Field(
        default=True,
        description="Enable distance estimation using inter-ocular distance",
    )


class FrameSelectionConfig(BaseModel):
    """Configuration for frame selection."""

    min_quality_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for frame selection",
    )
    face_size_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for face size in selection scoring (0-1)",
    )
    quality_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for quality metrics in selection scoring (0-1)",
    )
    diversity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum diversity score to avoid selecting similar frames",
    )

    @field_validator("face_size_weight", "quality_weight")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure face_size_weight and quality_weight don't exceed 1.0 when combined."""
        if info.field_name == "quality_weight":
            # Get face_size_weight from values if it exists
            face_size_weight = info.data.get("face_size_weight", 0.3)
            if v + face_size_weight > 1.0:
                raise ValueError(
                    f"face_size_weight ({face_size_weight}) + quality_weight ({v}) must not exceed 1.0"
                )
        return v


class PngConfig(BaseModel):
    """Configuration for PNG output."""

    optimize: bool = Field(
        True, description="Enable PNG optimization for smaller file sizes."
    )


class JpegConfig(BaseModel):
    """Configuration for JPEG output."""

    quality: int = Field(
        95, ge=70, le=100, description="Quality for JPEG images (1-100)."
    )


class OutputImageConfig(BaseModel):
    """Configuration for output generation."""

    format: str = Field(
        "png", description="The output image format ('png' or 'jpeg'). 'jpg' is used as the extension for 'jpeg'."
    )
    face_crop_enabled: bool = Field(
        default=True,
        description="Enable or disable the generation of cropped face images.",
    )
    face_crop_padding: float = Field(
        default=0.2,
        description="Padding around the face bounding box for crops (as a percentage).",
    )
    enable_pose_cropping: bool = Field(
        False, description="Enable generation of cropped pose images."
    )
    pose_crop_padding: float = Field(
        0.1, ge=0.0, le=1.0, description="Padding around pose bounding box."
    )
    resize: Optional[int] = Field(
        default=None,
        ge=256,
        le=4096,
        description="Maximum dimension for proportional image resizing (None for no resizing)"
    )

    png: PngConfig = Field(default_factory=PngConfig)
    jpeg: JpegConfig = Field(default_factory=JpegConfig)


class OutputConfig(BaseModel):
    """Configuration for output generation."""

    min_frames_per_category: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum frames to output per pose/angle category",
    )

    max_frames_per_category: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum frames to output per pose/angle category",
    )

    preserve_metadata: bool = Field(
        default=True, description="Preserve metadata in output images"
    )
    # Image output configuration
    image: OutputImageConfig = Field(default_factory=OutputImageConfig)


class StorageConfig(BaseModel):
    """Configuration for storage and caching."""

    cache_directory: Path = Field(
        default_factory=lambda: Path(user_cache_dir("personfromvid", "codeprimate")),
        description="Directory for model and data caching",
    )
    temp_directory: Optional[Path] = Field(
        default=None, description="Temporary directory (None for auto-generated)"
    )
    cleanup_temp_on_success: bool = Field(
        default=True, description="Clean up temporary files on successful completion"
    )
    cleanup_temp_on_failure: bool = Field(
        default=False, description="Clean up temporary files on failure"
    )
    keep_temp: bool = Field(
        default=False,
        description="Keep temporary files after processing (overrides cleanup settings)",
    )
    force_temp_cleanup: bool = Field(
        default=False,
        description="Force cleanup of existing temp directory before processing",
    )
    max_cache_size_gb: float = Field(
        default=5.0, ge=0.5, le=50.0, description="Maximum cache size in GB"
    )

    @field_validator("cache_directory", "temp_directory", mode="before")
    @classmethod
    def convert_path(cls, v):
        """Convert string paths to Path objects."""
        if v is None:
            return v
        return Path(v) if not isinstance(v, Path) else v

    @field_serializer("cache_directory", "temp_directory")
    def serialize_path(self, value: Optional[Path]) -> Optional[str]:
        """Serialize Path objects to strings for JSON."""
        return str(value) if value is not None else None


class ProcessingConfig(BaseModel):
    """Configuration for processing behavior."""

    force_restart: bool = Field(
        default=False, 
        description="Force restart processing by deleting existing state (preserves extracted frames)"
    )
    save_intermediate_results: bool = Field(
        default=True, description="Save intermediate processing results"
    )
    max_processing_time_minutes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum processing time in minutes (None for unlimited)",
    )
    parallel_workers: int = Field(
        default=1, ge=1, le=16, description="Number of parallel workers for processing"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    enable_file_logging: bool = Field(
        default=False, description="Enable logging to file"
    )
    log_file: Optional[Path] = Field(
        default=None, description="Log file path (None for auto-generated)"
    )
    enable_rich_console: bool = Field(
        default=True, description="Enable rich console formatting"
    )
    enable_structured_output: bool = Field(
        default=True,
        description="Enable structured output format with emojis and progress bars",
    )
    verbose: bool = Field(default=False, description="Enable verbose logging")

    @field_serializer("level")
    def serialize_level(self, value) -> str:
        """Serialize LogLevel enum to string for JSON."""
        if isinstance(value, LogLevel):
            return value.value
        return str(value)

    @field_validator("log_file", mode="before")
    @classmethod
    def convert_log_path(cls, v):
        """Convert string paths to Path objects."""
        if v is None:
            return v
        return Path(v) if not isinstance(v, Path) else v

    @field_serializer("log_file")
    def serialize_log_path(self, value: Optional[Path]) -> Optional[str]:
        """Serialize Path objects to strings for JSON."""
        return str(value) if value is not None else None


class Config(BaseModel):
    """Main configuration class combining all settings."""

    # Sub-configurations
    models: ModelConfig = Field(default_factory=ModelConfig)
    frame_extraction: FrameExtractionConfig = Field(
        default_factory=FrameExtractionConfig
    )
    quality: QualityConfig = Field(default_factory=QualityConfig)
    pose_classification: PoseClassificationConfig = Field(
        default_factory=PoseClassificationConfig
    )
    head_angle: HeadAngleConfig = Field(default_factory=HeadAngleConfig)
    closeup_detection: CloseupDetectionConfig = Field(
        default_factory=CloseupDetectionConfig
    )
    frame_selection: FrameSelectionConfig = Field(default_factory=FrameSelectionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = ConfigDict(
        env_prefix="PERSONFROMVID_",
        env_nested_delimiter="__",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from YAML or JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                import json

                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration with environment variable overrides."""
        return cls()

    def to_file(self, config_path: Path) -> None:
        """Save configuration to YAML or JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.dict()

        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.safe_dump(data, f, default_flow_style=False)
            elif config_path.suffix.lower() == ".json":
                import json

                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        self.storage.cache_directory.mkdir(parents=True, exist_ok=True)

        if self.storage.temp_directory:
            self.storage.temp_directory.mkdir(parents=True, exist_ok=True)

        if self.logging.log_file:
            self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)

    def validate_system_requirements(self) -> List[str]:
        """Validate system requirements and return list of issues."""
        issues = []

        # Check available disk space
        import shutil

        try:
            cache_space = shutil.disk_usage(self.storage.cache_directory.parent)
            available_gb = cache_space.free / (1024**3)
            if available_gb < self.storage.max_cache_size_gb:
                issues.append(
                    f"Insufficient disk space. Available: {available_gb:.1f}GB, "
                    f"Required: {self.storage.max_cache_size_gb}GB"
                )
        except Exception as e:
            issues.append(f"Could not check disk space: {e}")

        # Check if required directories are writable
        try:
            self.storage.cache_directory.mkdir(parents=True, exist_ok=True)
            test_file = self.storage.cache_directory / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"Cache directory not writable: {e}")

        return issues


def get_default_config() -> Config:
    """Get default configuration with environment variable overrides."""
    return Config.from_env()


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or use defaults with env overrides."""
    if config_path and config_path.exists():
        return Config.from_file(config_path)
    return get_default_config()
