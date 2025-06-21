"""Naming convention logic for output image files.

This module implements the NamingConvention class that generates standardized,
descriptive filenames for output images based on frame metadata.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set

from ..data.context import ProcessingContext
from ..data.frame_data import FrameData
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
        self,
        frame: FrameData,
        category: str,
        rank: int,
        extension: str = "png",
        person_id: Optional[int] = None,
        head_direction: Optional[str] = None,
        shot_type: Optional[str] = None,
    ) -> str:
        """Generate filename for full frame image.

        Args:
            frame: Frame data containing metadata
            category: Pose category (e.g., "standing", "sitting")
            rank: Rank within category (1, 2, 3)
            extension: File extension without dot
            person_id: Optional person ID for person-based selection output
            head_direction: Optional head direction override (for person-specific data)
            shot_type: Optional shot type override (for person-specific data)

        Returns:
            Filename string
        """
        # Get head direction and shot type (use provided values or extract from frame)
        if head_direction is None:
            head_direction = self._get_head_direction(frame)
        if shot_type is None:
            shot_type = self._get_shot_type(frame)

        # Build base filename with optional person_id support
        if person_id is not None:
            # Person-based naming: video_person_{person_id}_pose_head-direction_shot-type_rank.ext
            base_parts = [
                self.video_base_name,
                f"person_{person_id}",
                category,
                head_direction,
                shot_type,
                f"{rank:03d}",
            ]
        else:
            # Traditional frame-based naming: video_pose_head-direction_shot-type_rank.ext
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
        self,
        frame: FrameData,
        head_angle: str,
        rank: int,
        extension: str = "png",
        person_id: Optional[int] = None,
        shot_type: Optional[str] = None,
    ) -> str:
        """Generate filename for face crop image.

        Args:
            frame: Frame data containing metadata
            head_angle: Head angle category (e.g., "front", "profile_left")
            rank: Rank within category (1, 2, 3)
            extension: File extension without dot
            person_id: Optional person ID for person-based selection output
            shot_type: Optional shot type override (for person-specific data)

        Returns:
            Filename string
        """
        # Get shot type (use provided value or extract from frame) - but only if explicitly requested
        if shot_type is None:
            # Don't automatically extract shot type - keep filenames simple
            shot_type = ""
        elif shot_type == "":
            # Empty string means don't include shot type
            pass
        else:
            # Use the provided shot type
            pass

        # Build base filename with optional person_id support
        if person_id is not None:
            # Person-based naming: video_person_{person_id}_face_head-angle[_shot-type]_rank.ext
            base_parts = [
                self.video_base_name,
                f"person_{person_id}",
                "face",
                head_angle,
            ]
            # Only add shot type if it's not empty
            if shot_type:
                base_parts.append(shot_type)
            base_parts.append(f"{rank:03d}")
        else:
            # Traditional frame-based naming: video_face_head-angle[_shot-type]_rank.ext
            base_parts = [
                self.video_base_name,
                "face",
                head_angle,
            ]
            # Only add shot type if it's not empty
            if shot_type:
                base_parts.append(shot_type)
            base_parts.append(f"{rank:03d}")

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

        # Additional validation for person-based filenames
        if "_person_" in filename:
            # Validate person_id pattern: video_person_{digit}_...
            parts = filename.split("_")
            if len(parts) >= 3:
                # Find person part
                person_part_idx = None
                for i, part in enumerate(parts):
                    if part == "person" and i + 1 < len(parts):
                        person_part_idx = i + 1
                        break

                if person_part_idx is not None:
                    person_id_part = parts[person_part_idx]
                    # Check if person_id is a valid integer
                    try:
                        int(person_id_part)
                    except ValueError:
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
