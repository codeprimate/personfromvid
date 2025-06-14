"""Custom exception classes for Person From Vid.

This module defines the exception hierarchy for the video processing pipeline,
providing structured error handling and informative error messages.
"""

from typing import Optional, Dict, Any


class PersonFromVidError(Exception):
    """Base exception class for Person From Vid errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(PersonFromVidError):
    """Raised when there are configuration-related errors."""

    pass


class ValidationError(PersonFromVidError):
    """Raised when input validation fails."""

    pass


class VideoProcessingError(PersonFromVidError):
    """Base class for video processing errors."""

    pass


class VideoFileError(VideoProcessingError):
    """Raised when there are issues with video file access or format."""

    pass


class FrameExtractionError(VideoProcessingError):
    """Raised when frame extraction fails."""

    pass


class ModelError(PersonFromVidError):
    """Base class for AI model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found."""

    pass


class ModelDownloadError(ModelError):
    """Raised when model download fails."""

    pass


class ModelIntegrityError(ModelError):
    """Raised when model integrity verification fails."""

    pass


class ModelLoadingError(ModelError):
    """Raised when model loading fails."""

    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""

    pass


class FaceDetectionError(ModelInferenceError):
    """Raised when face detection fails."""

    pass


class PoseEstimationError(ModelInferenceError):
    """Raised when pose estimation fails."""

    pass


class HeadPoseEstimationError(ModelInferenceError):
    """Raised when head pose estimation fails."""

    pass


class AnalysisError(PersonFromVidError):
    """Base class for analysis-related errors."""

    pass


class QualityAssessmentError(AnalysisError):
    """Raised when quality assessment fails."""

    pass


class PoseClassificationError(AnalysisError):
    """Raised when pose classification fails."""

    pass


class HeadAngleClassificationError(AnalysisError):
    """Raised when head angle classification fails."""

    pass


class CloseupDetectionError(AnalysisError):
    """Raised when closeup detection fails."""

    pass


class OutputError(PersonFromVidError):
    """Base class for output generation errors."""

    pass


class ImageWriteError(OutputError):
    """Raised when image writing fails."""

    pass


class MetadataWriteError(OutputError):
    """Raised when metadata writing fails."""

    pass


class StateManagementError(PersonFromVidError):
    """Base class for state management errors."""

    pass


class StateLoadError(StateManagementError):
    """Raised when loading pipeline state fails."""

    pass


class StateSaveError(StateManagementError):
    """Raised when saving pipeline state fails."""

    pass


class TempDirectoryError(PersonFromVidError):
    """Raised when temporary directory operations fail."""

    pass


class DiskSpaceError(PersonFromVidError):
    """Raised when there is insufficient disk space."""

    pass


class InterruptionError(PersonFromVidError):
    """Raised when processing is interrupted by user."""

    pass


class PipelineInterruptedError(InterruptionError):
    """Raised when pipeline processing is interrupted."""

    pass


class TimeoutError(PersonFromVidError):
    """Raised when processing exceeds time limits."""

    pass


class DependencyError(PersonFromVidError):
    """Raised when required dependencies are missing or incompatible."""

    pass


class GPUError(PersonFromVidError):
    """Raised when GPU-related operations fail."""

    pass


# Error code mapping for structured error handling
ERROR_CODES = {
    PersonFromVidError: "PFV_000",
    ConfigurationError: "PFV_001",
    ValidationError: "PFV_002",
    VideoProcessingError: "PFV_100",
    VideoFileError: "PFV_101",
    FrameExtractionError: "PFV_102",
    ModelError: "PFV_200",
    ModelNotFoundError: "PFV_200A",
    ModelDownloadError: "PFV_201",
    ModelIntegrityError: "PFV_201A",
    ModelLoadingError: "PFV_202",
    ModelInferenceError: "PFV_203",
    FaceDetectionError: "PFV_204",
    PoseEstimationError: "PFV_205",
    HeadPoseEstimationError: "PFV_206",
    AnalysisError: "PFV_300",
    QualityAssessmentError: "PFV_301",
    PoseClassificationError: "PFV_302",
    HeadAngleClassificationError: "PFV_303",
    CloseupDetectionError: "PFV_304",
    OutputError: "PFV_400",
    ImageWriteError: "PFV_401",
    MetadataWriteError: "PFV_402",
    StateManagementError: "PFV_500",
    StateLoadError: "PFV_501",
    StateSaveError: "PFV_502",
    TempDirectoryError: "PFV_503",
    DiskSpaceError: "PFV_504",
    InterruptionError: "PFV_505",
    PipelineInterruptedError: "PFV_505A",
    TimeoutError: "PFV_506",
    DependencyError: "PFV_507",
    GPUError: "PFV_508",
}


def get_error_code(exception_class: type) -> str:
    """Get error code for exception class."""
    return ERROR_CODES.get(exception_class, "PFV_000")


def format_exception_message(
    exception: Exception, include_traceback: bool = False
) -> str:
    """Format exception message with optional traceback."""
    if isinstance(exception, PersonFromVidError):
        error_code = get_error_code(type(exception))
        message = f"[{error_code}] {exception.message}"

        if exception.details:
            details = ", ".join(f"{k}={v}" for k, v in exception.details.items())
            message += f" (Details: {details})"
    else:
        message = f"[SYSTEM] {str(exception)}"

    if include_traceback:
        import traceback

        tb = traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
        message += "\n" + "".join(tb)

    return message
