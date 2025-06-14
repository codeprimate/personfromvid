"""Processing context for unified data flow.

This module defines the ProcessingContext data class that serves as a centralized
container for all the common data needed throughout the video processing pipeline.
This reduces parameter passing and decouples components from the main orchestrator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config

# Avoid circular imports
if TYPE_CHECKING:
    from ..core.temp_manager import TempManager


@dataclass(frozen=True)
class ProcessingContext:
    """Unified processing context containing common pipeline data.

    This immutable data class consolidates the most commonly passed parameters
    throughout the video processing pipeline, including paths, configuration,
    and shared utilities.

    Attributes:
        video_path: Path to the input video file
        video_base_name: Base name of the video file (without extension)
        config: Application configuration object
        output_directory: Path to the output directory for results
        temp_manager: Temporary directory manager
    """

    video_path: Path
    video_base_name: str
    config: Config
    output_directory: Path
    temp_manager: "TempManager" = field(init=False)

    def __post_init__(self) -> None:
        """Validate the context and initialize non-constructor fields."""
        # Defer import to avoid circular dependency
        from ..core.temp_manager import TempManager

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        if not self.video_path.is_file():
            raise ValueError(f"Video path is not a file: {self.video_path}")

        # Initialize TempManager with config
        # We must cast self to object to bypass frozen=True restriction
        temp_manager = TempManager(str(self.video_path), self.config)
        temp_manager.create_temp_structure()
        object.__setattr__(self, "temp_manager", temp_manager)

        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def video_name(self) -> str:
        """Get the full name of the video file including extension."""
        return self.video_path.name

    @property
    def video_stem(self) -> str:
        """Get the video file stem (name without extension)."""
        return self.video_path.stem

    @property
    def video_suffix(self) -> str:
        """Get the video file extension."""
        return self.video_path.suffix

    @property
    def temp_directory(self) -> Path:
        """Get the temporary directory path."""
        return self.temp_manager.get_temp_path()

    @property
    def frames_directory(self) -> Path:
        """Get the frames subdirectory path."""
        return self.temp_manager.get_frames_dir()
