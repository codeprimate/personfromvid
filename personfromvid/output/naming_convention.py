"""Naming convention logic for output image files.

This module implements the NamingConvention class that generates standardized,
descriptive filenames for output images based on frame metadata.
"""

import logging
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

from ..data.frame_data import FrameData
from ..data.context import ProcessingContext
from ..utils.logging import get_logger


class NamingConvention:
    """Generates consistent output filenames based on frame metadata."""

    def __init__(self, context: ProcessingContext):
        """Initialize naming convention.

        Args:
            context: ProcessingContext with unified pipeline data
        """
        self.video_base_name = context.video_base_name
        self.output_directory = context.output_directory
        self.logger = get_logger(__name__)
        self._used_filenames: Set[str] = set()
        self._sequence_counters: Dict[str, int] = defaultdict(int)

    def get_full_frame_filename(
        self, frame: FrameData, category: str, rank: int, extension: str = "png"
    ) -> str:
        """Generate filename for full frame image.

        Args:
            frame: Frame data containing metadata
            category: Pose category (e.g., "standing", "sitting")
            rank: Rank within category (1, 2, 3)
            extension: File extension without dot

        Returns:
            Filename string
        """
        # Get head direction from frame
        head_direction = self._get_head_direction(frame)
        shot_type = self._get_shot_type(frame)

        # Build base filename: video_pose_head-direction_shot-type_rank.ext
        base_parts = [
            self.video_base_name,
            category,
            head_direction,
            shot_type,
            f"{rank:03d}",
        ]

        base_filename = "_".join(part for part in base_parts if part) + f".{extension}"

        # Handle collisions
        return self._ensure_unique_filename(base_filename)

    def get_face_crop_filename(
        self, frame: FrameData, head_angle: str, rank: int, extension: str = "png"
    ) -> str:
        """Generate filename for face crop image.

        Args:
            frame: Frame data containing metadata
            head_angle: Head angle category (e.g., "front", "profile_left")
            rank: Rank within category (1, 2, 3)
            extension: File extension without dot

        Returns:
            Filename string
        """
        # Build base filename: video_face_head-angle_rank.ext
        base_parts = [self.video_base_name, "face", head_angle, f"{rank:03d}"]

        base_filename = "_".join(base_parts) + f".{extension}"

        # Handle collisions
        return self._ensure_unique_filename(base_filename)

    def get_crop_suffixed_filename(self, base_filename: str) -> str:
        """Generate crop-suffixed filename by inserting '_crop' before extension.

        Args:
            base_filename: Base filename to add crop suffix to

        Returns:
            Filename with '_crop' suffix inserted before extension
        """
        path = Path(base_filename)
        name_without_ext = path.stem
        extension = path.suffix
        return f"{name_without_ext}_crop{extension}"

    def get_full_output_path(self, filename: str) -> Path:
        """Get full output path for a filename.

        Args:
            filename: Generated filename

        Returns:
            Full path to output file
        """
        return self.output_directory / filename

    def validate_filename(self, filename: str) -> bool:
        """Validate that filename follows expected pattern.

        Args:
            filename: Filename to validate

        Returns:
            True if filename is valid
        """
        if not filename:
            return False

        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            return False

        # Check length (most filesystems support 255 chars)
        if len(filename) > 255:
            return False

        # Check that it starts with video base name
        if not filename.startswith(self.video_base_name):
            return False

        return True

    def _get_head_direction(self, frame: FrameData) -> str:
        """Extract head direction from frame data.

        Args:
            frame: Frame data

        Returns:
            Head direction string or empty string if not available
        """
        if frame.head_poses:
            best_head_pose = frame.get_best_head_pose()
            if best_head_pose and best_head_pose.direction:
                # Convert direction to filename-safe format
                return best_head_pose.direction.replace(" ", "-").lower()

        return ""

    def _get_shot_type(self, frame: FrameData) -> str:
        """Extract shot type from frame data.

        Args:
            frame: Frame data

        Returns:
            Shot type string or empty string if not available
        """
        if frame.closeup_detections:
            # Get the best closeup detection
            for closeup in frame.closeup_detections:
                if closeup.shot_type:
                    return closeup.shot_type.replace(" ", "-").lower()

        return ""

    def _ensure_unique_filename(self, base_filename: str) -> str:
        """Ensure filename is unique by adding sequence number if needed.

        Args:
            base_filename: Base filename to make unique

        Returns:
            Unique filename
        """
        if base_filename not in self._used_filenames:
            self._used_filenames.add(base_filename)
            return base_filename

        # Extract name and extension
        path = Path(base_filename)
        name_without_ext = path.stem
        extension = path.suffix

        # Generate unique filename with sequence number
        sequence = self._sequence_counters[name_without_ext] + 1

        while True:
            new_filename = f"{name_without_ext}_{sequence:03d}{extension}"
            if new_filename not in self._used_filenames:
                self._used_filenames.add(new_filename)
                self._sequence_counters[name_without_ext] = sequence
                return new_filename
            sequence += 1

    def reset_counters(self) -> None:
        """Reset filename counters (useful for testing)."""
        self._used_filenames.clear()
        self._sequence_counters.clear()
