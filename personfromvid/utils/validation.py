"""Input validation utilities for Person From Vid.

This module provides validation functions for video files, paths, and system 
requirements to ensure proper input handling throughout the pipeline.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import subprocess
import shutil

from .exceptions import ValidationError, VideoFileError, DependencyError


# Supported video file extensions and MIME types
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".3gp",
    ".ogv",
}

SUPPORTED_VIDEO_MIMETYPES = {
    "video/mp4",
    "video/avi",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "video/x-flv",
    "video/3gpp",
}

# Minimum system requirements
MIN_PYTHON_VERSION = (3, 8)
MIN_DISK_SPACE_GB = 0.5

# Required external dependencies
REQUIRED_EXECUTABLES = {
    "ffmpeg": "FFmpeg is required for video processing",
    "ffprobe": "FFprobe is required for video metadata extraction",
}


def validate_video_file(video_path: Path) -> Dict[str, Any]:
    """Validate video file and return metadata.

    Args:
        video_path: Path to the video file

    Returns:
        Dict containing validation results and basic metadata

    Raises:
        ValidationError: If validation fails
        VideoFileError: If video file is invalid
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)

    validation_result = {
        "path": video_path,
        "exists": False,
        "readable": False,
        "size_bytes": 0,
        "extension_valid": False,
        "mimetype_valid": False,
        "metadata": {},
    }

    # Check if file exists
    if not video_path.exists():
        raise VideoFileError(f"Video file does not exist: {video_path}")

    validation_result["exists"] = True

    # Check if file is readable
    if not os.access(video_path, os.R_OK):
        raise VideoFileError(f"Video file is not readable: {video_path}")

    validation_result["readable"] = True

    # Check file size
    try:
        file_size = video_path.stat().st_size
        validation_result["size_bytes"] = file_size

        if file_size == 0:
            raise VideoFileError(f"Video file is empty: {video_path}")

        if file_size < 1024:  # Less than 1KB is suspicious
            raise ValidationError(
                f"Video file is suspiciously small ({file_size} bytes): {video_path}"
            )

    except OSError as e:
        raise VideoFileError(f"Could not access video file: {e}")

    # Validate file extension
    extension = video_path.suffix.lower()
    validation_result["extension_valid"] = extension in SUPPORTED_VIDEO_EXTENSIONS

    if not validation_result["extension_valid"]:
        raise ValidationError(
            f"Unsupported video file extension: {extension}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    # Validate MIME type
    mimetype, _ = mimetypes.guess_type(str(video_path))
    validation_result["mimetype_valid"] = mimetype in SUPPORTED_VIDEO_MIMETYPES

    if not validation_result["mimetype_valid"] and mimetype:
        # Not always reliable, so just warn
        import warnings

        warnings.warn(f"Unexpected MIME type for video file: {mimetype}")

    # Try to get basic video metadata using ffprobe if available
    try:
        metadata = _get_video_metadata_ffprobe(video_path)
        validation_result["metadata"] = metadata
    except Exception as e:
        # Metadata extraction failure is not fatal for validation
        validation_result["metadata"] = {"error": str(e)}

    return validation_result


def validate_output_path(output_path: Path, create_if_missing: bool = True) -> bool:
    """Validate output directory path.

    Args:
        output_path: Path to output directory
        create_if_missing: Whether to create directory if it doesn't exist

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    # Check if parent directory exists and is writable
    parent_dir = output_path.parent
    if not parent_dir.exists():
        if create_if_missing:
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValidationError(f"Cannot create output directory: {e}")
        else:
            raise ValidationError(f"Output directory does not exist: {parent_dir}")

    # Check write permissions
    if not os.access(parent_dir, os.W_OK):
        raise ValidationError(f"Output directory is not writable: {parent_dir}")

    # Check available disk space
    try:
        disk_usage = shutil.disk_usage(parent_dir)
        available_gb = disk_usage.free / (1024**3)

        if available_gb < MIN_DISK_SPACE_GB:
            raise ValidationError(
                f"Insufficient disk space in output directory. "
                f"Available: {available_gb:.1f}GB, Required: {MIN_DISK_SPACE_GB}GB"
            )
    except OSError as e:
        raise ValidationError(f"Could not check disk space: {e}")

    return True


def validate_system_requirements() -> List[str]:
    """Validate system requirements.

    Returns:
        List of validation warnings/errors (empty if all OK)
    """
    issues = []

    # Check Python version
    import sys

    python_version = sys.version_info[:2]
    if python_version < MIN_PYTHON_VERSION:
        issues.append(
            f"Python version {python_version[0]}.{python_version[1]} is not supported. "
            f"Minimum required: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
        )

    # Memory checking removed - let the OS handle memory management

    # Check required executables
    for executable, description in REQUIRED_EXECUTABLES.items():
        if not _check_executable(executable):
            issues.append(f"Missing required dependency: {executable}. {description}")

    # Check GPU availability (optional)
    gpu_info = _check_gpu_availability()
    if gpu_info["issues"]:
        issues.extend(gpu_info["issues"])

    return issues


def validate_config_values(config_dict: Dict[str, Any]) -> List[str]:
    """Validate configuration values.

    Args:
        config_dict: Configuration dictionary

    Returns:
        List of validation issues
    """
    issues = []

    # Validate batch sizes
    if "models" in config_dict and "batch_size" in config_dict["models"]:
        batch_size = config_dict["models"]["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 64:
            issues.append(
                f"Invalid batch size: {batch_size}. Must be between 1 and 64."
            )

    # Validate confidence thresholds
    if "models" in config_dict and "confidence_threshold" in config_dict["models"]:
        threshold = config_dict["models"]["confidence_threshold"]
        if (
            not isinstance(threshold, (int, float))
            or threshold < 0.0
            or threshold > 1.0
        ):
            issues.append(
                f"Invalid confidence threshold: {threshold}. Must be between 0.0 and 1.0."
            )

    # Validate quality thresholds
    if "quality" in config_dict:
        quality_config = config_dict["quality"]

        if "blur_threshold" in quality_config:
            blur_threshold = quality_config["blur_threshold"]
            if not isinstance(blur_threshold, (int, float)) or blur_threshold < 0:
                issues.append(
                    f"Invalid blur threshold: {blur_threshold}. Must be positive."
                )

        if "brightness_min" in quality_config and "brightness_max" in quality_config:
            min_brightness = quality_config["brightness_min"]
            max_brightness = quality_config["brightness_max"]

            if min_brightness >= max_brightness:
                issues.append(
                    f"Invalid brightness range: min={min_brightness}, max={max_brightness}. "
                    "Min must be less than max."
                )

    return issues


def _get_video_metadata_ffprobe(video_path: Path) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise VideoFileError(f"ffprobe failed: {result.stderr}")

        import json

        metadata = json.loads(result.stdout)

        # Extract relevant information
        format_info = metadata.get("format", {})
        video_streams = [
            s for s in metadata.get("streams", []) if s.get("codec_type") == "video"
        ]

        if not video_streams:
            raise VideoFileError("No video streams found in file")

        video_stream = video_streams[0]  # Use first video stream

        return {
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bitrate": int(format_info.get("bit_rate", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),  # Fraction to decimal
            "codec": video_stream.get("codec_name", "unknown"),
            "pixel_format": video_stream.get("pix_fmt", "unknown"),
        }

    except subprocess.TimeoutExpired:
        raise VideoFileError("Video metadata extraction timed out")
    except json.JSONDecodeError as e:
        raise VideoFileError(f"Could not parse video metadata: {e}")
    except Exception as e:
        raise VideoFileError(f"Video metadata extraction failed: {e}")


def _check_executable(executable_name: str) -> bool:
    """Check if an executable is available in PATH."""
    return shutil.which(executable_name) is not None


def _check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return status."""
    result = {"available": False, "devices": [], "issues": []}

    # Check CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            result["available"] = True
            result["devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory": torch.cuda.get_device_properties(i).total_memory
                    / (1024**3),
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            result["issues"].append("CUDA is not available (GPU acceleration disabled)")
    except ImportError:
        result["issues"].append("PyTorch not available for GPU detection")
    except Exception as e:
        result["issues"].append(f"Error checking GPU availability: {e}")

    return result


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for all filesystems
    """
    # Replace problematic characters
    replacements = {
        "<": "_",
        ">": "_",
        ":": "_",
        '"': "_",
        "|": "_",
        "?": "_",
        "*": "_",
        "/": "_",
        "\\": "_",
    }

    sanitized = filename
    for char, replacement in replacements.items():
        sanitized = sanitized.replace(char, replacement)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def validate_model_path(model_path: Path) -> bool:
    """Validate model file path.

    Args:
        model_path: Path to model file

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not model_path.exists():
        raise ValidationError(f"Model file does not exist: {model_path}")

    if not model_path.is_file():
        raise ValidationError(f"Model path is not a file: {model_path}")

    if model_path.stat().st_size == 0:
        raise ValidationError(f"Model file is empty: {model_path}")

    # Check if file is readable
    if not os.access(model_path, os.R_OK):
        raise ValidationError(f"Model file is not readable: {model_path}")

    return True
