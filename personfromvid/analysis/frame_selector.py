"""Frame selection logic for choosing best frames based on quality and diversity.

This module implements the FrameSelector class that ranks and selects the best
frames from each pose category and head angle category based on comprehensive
quality metrics and diversity considerations.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

from ..data.frame_data import FrameData
from ..data.detection_results import QualityMetrics, FaceDetection, CloseupDetection
from ..utils.logging import get_logger


@dataclass
class SelectionCriteria:
    """Criteria for frame selection."""

    min_frames_per_category: int
    min_quality_threshold: float
    face_size_weight: float
    quality_weight: float
    diversity_threshold: float


@dataclass
class CategorySelection:
    """Selection results for a specific category."""

    category_name: str
    category_type: str  # "pose" or "head_angle"
    selected_frames: List[FrameData]
    total_candidates: int
    selection_rationale: str
    quality_range: Tuple[float, float]  # (min_quality, max_quality)
    average_quality: float


@dataclass
class SelectionSummary:
    """Complete frame selection summary."""

    total_candidates: int
    total_selected: int
    pose_selections: Dict[str, CategorySelection]
    head_angle_selections: Dict[str, CategorySelection]
    selection_criteria: SelectionCriteria
    processing_notes: List[str]


class FrameSelector:
    """Selects best frames based on quality metrics and diversity considerations.

    This class implements sophisticated frame selection logic that:
    1. Groups frames by pose category and head angle
    2. Ranks frames using quality metrics and face size
    3. Selects diverse, high-quality representatives for each category
    4. Provides detailed selection rationale and metadata
    """

    def __init__(self, criteria: SelectionCriteria):
        """Initialize frame selector with selection criteria.

        Args:
            criteria: Selection criteria (required)
        """
        self.criteria = criteria
        self.logger = get_logger(f"{__name__}.FrameSelector")

        # Categories we're interested in
        self.pose_categories = ["standing", "sitting", "squatting", "closeup"]
        self.head_angle_categories = [
            "front",
            "looking_left",
            "looking_right",
            "profile_left",
            "profile_right",
            "looking_up",
            "looking_down",
            "looking_up_left",
            "looking_up_right",
        ]

        self.logger.debug(f"FrameSelector initialized with criteria: {self.criteria}")

    def select_best_frames(
        self,
        candidate_frames: List[FrameData],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SelectionSummary:
        """Select best frames from candidates based on quality and diversity.

        Args:
            candidate_frames: List of candidate frames to select from
            progress_callback: Optional callback for progress updates

        Returns:
            SelectionSummary with complete selection results
        """
        self.logger.info(
            f"Starting frame selection from {len(candidate_frames)} candidates"
        )

        if progress_callback:
            progress_callback("Filtering candidate frames...")

        # Filter to usable frames only
        usable_frames = self._filter_usable_frames(candidate_frames)
        self.logger.info(f"Found {len(usable_frames)} usable frames after filtering")

        if not usable_frames:
            return self._create_empty_summary(len(candidate_frames))

        if progress_callback:
            progress_callback("Grouping frames by categories...")

        # Group frames by categories
        pose_groups = self.group_by_pose(usable_frames)
        head_angle_groups = self.group_by_head_angle(usable_frames)

        if progress_callback:
            progress_callback("Selecting best frames for pose categories...")

        # Select best frames for pose categories, with de-duplication
        pose_selections = {}
        claimed_frame_ids = set()

        # Iterate through pose categories in their defined order of priority
        for category in self.pose_categories:
            candidate_frames = pose_groups.get(category, [])

            # Filter out frames already claimed by a higher-priority category
            available_candidates = [
                frame
                for frame in candidate_frames
                if frame.frame_id not in claimed_frame_ids
            ]

            if available_candidates:
                selection = self._select_for_category(
                    available_candidates,
                    category,
                    "pose",
                    self._calculate_pose_frame_score,
                )

                if selection.selected_frames:
                    pose_selections[category] = selection
                    # Claim frames so they can't be used by other pose categories
                    for frame in selection.selected_frames:
                        claimed_frame_ids.add(frame.frame_id)
                    self.logger.debug(
                        f"Selected and claimed {len(selection.selected_frames)} frames for pose '{category}'"
                    )

        if progress_callback:
            progress_callback("Selecting best frames for head angle categories...")

        # Select best frames for head angle categories (no de-duplication)
        head_angle_selections = {}
        for angle, frames in head_angle_groups.items():
            if frames:
                selection = self._select_for_category(
                    frames, angle, "head_angle", self._calculate_head_angle_frame_score
                )
                head_angle_selections[angle] = selection
                self.logger.debug(
                    f"Selected {len(selection.selected_frames)} frames for head angle '{angle}'"
                )

        if progress_callback:
            progress_callback("Updating frame selection metadata...")

        # Update frame selection metadata
        self._update_frame_selection_metadata(pose_selections, head_angle_selections)

        # Create summary, counting unique selected frames
        all_selected_frame_ids = set()
        for sel in pose_selections.values():
            for frame in sel.selected_frames:
                all_selected_frame_ids.add(frame.frame_id)
        for sel in head_angle_selections.values():
            for frame in sel.selected_frames:
                all_selected_frame_ids.add(frame.frame_id)

        total_selected = len(all_selected_frame_ids)

        summary = SelectionSummary(
            total_candidates=len(candidate_frames),
            total_selected=total_selected,
            pose_selections=pose_selections,
            head_angle_selections=head_angle_selections,
            selection_criteria=self.criteria,
            processing_notes=[],
        )

        self.logger.info(
            f"Frame selection completed: {total_selected} unique frames selected"
        )
        return summary

    def group_by_pose(self, frames: List[FrameData]) -> Dict[str, List[FrameData]]:
        """Group frames by pose classification.

        Args:
            frames: List of frames to group

        Returns:
            Dictionary mapping pose categories to frame lists
        """
        pose_groups = defaultdict(list)

        for frame in frames:
            # Get pose classifications from frame
            pose_classifications = frame.get_pose_classifications()

            # Also check for closeup classification from closeup detections
            if frame.is_closeup_shot():
                pose_classifications.append("closeup")

            # Add frame to appropriate groups
            for pose in pose_classifications:
                if pose in self.pose_categories:
                    pose_groups[pose].append(frame)

        # Convert defaultdict to regular dict for cleaner output
        return dict(pose_groups)

    def group_by_head_angle(
        self, frames: List[FrameData]
    ) -> Dict[str, List[FrameData]]:
        """Group frames by head angle direction.

        Args:
            frames: List of frames to group

        Returns:
            Dictionary mapping head angle categories to frame lists
        """
        head_angle_groups = defaultdict(list)

        for frame in frames:
            # Get head directions from frame
            head_directions = frame.get_head_directions()

            # Add frame to appropriate groups
            for direction in head_directions:
                if direction in self.head_angle_categories:
                    head_angle_groups[direction].append(frame)

        # Convert defaultdict to regular dict for cleaner output
        return dict(head_angle_groups)

    def rank_by_quality(self, frames: List[FrameData]) -> List[FrameData]:
        """Rank frames by overall quality score.

        Args:
            frames: List of frames to rank

        Returns:
            List of frames sorted by quality (highest first)
        """

        def get_quality_score(frame: FrameData) -> float:
            if frame.quality_metrics is None:
                return 0.0
            return frame.quality_metrics.overall_quality

        return sorted(frames, key=get_quality_score, reverse=True)

    def _filter_usable_frames(self, frames: List[FrameData]) -> List[FrameData]:
        """Filter frames to only include usable ones.

        Args:
            frames: Input frames to filter

        Returns:
            List of frames that meet usability criteria
        """
        usable = []

        for frame in frames:
            # Must have faces
            if not frame.has_faces():
                continue

            # Must have quality metrics
            if frame.quality_metrics is None:
                continue

            # Must meet minimum quality threshold
            if (
                frame.quality_metrics.overall_quality
                < self.criteria.min_quality_threshold
            ):
                continue

            # Must be marked as usable
            if not frame.quality_metrics.usable:
                continue

            usable.append(frame)

        return usable

    def _select_for_category(
        self,
        frames: List[FrameData],
        category_name: str,
        category_type: str,
        score_function: Callable[[FrameData], float],
    ) -> CategorySelection:
        """Select best frames for a specific category.

        Args:
            frames: Candidate frames for this category
            category_name: Name of the category
            category_type: Type of category ("pose" or "head_angle")
            score_function: Function to calculate frame score

        Returns:
            CategorySelection with results
        """
        if not frames:
            return CategorySelection(
                category_name=category_name,
                category_type=category_type,
                selected_frames=[],
                total_candidates=0,
                selection_rationale="No candidate frames available",
                quality_range=(0.0, 0.0),
                average_quality=0.0,
            )

        # Calculate scores for all frames
        scored_frames = []
        for frame in frames:
            score = score_function(frame)
            scored_frames.append((frame, score))

        # Sort by score (highest first)
        scored_frames.sort(key=lambda x: x[1], reverse=True)

        # Select top frames with diversity consideration
        selected_frames = self._select_diverse_frames(
            scored_frames, self.criteria.min_frames_per_category
        )

        # Calculate statistics
        quality_scores = [
            f.quality_metrics.overall_quality for f in frames if f.quality_metrics
        ]
        quality_range = (
            (min(quality_scores), max(quality_scores)) if quality_scores else (0.0, 0.0)
        )
        average_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        # Create rationale
        rationale = self._create_selection_rationale(
            category_name,
            category_type,
            len(frames),
            len(selected_frames),
            quality_range,
            average_quality,
        )

        return CategorySelection(
            category_name=category_name,
            category_type=category_type,
            selected_frames=selected_frames,
            total_candidates=len(frames),
            selection_rationale=rationale,
            quality_range=quality_range,
            average_quality=average_quality,
        )

    def _select_diverse_frames(
        self, scored_frames: List[Tuple[FrameData, float]], max_count: int
    ) -> List[FrameData]:
        """Select diverse frames to avoid similar selections.

        Args:
            scored_frames: List of (frame, score) tuples sorted by score
            max_count: Maximum number of frames to select

        Returns:
            List of selected frames
        """
        if not scored_frames:
            return []

        selected = []

        for frame, score in scored_frames:
            if len(selected) >= max_count:
                break

            # Check diversity against already selected frames
            if self._is_diverse_enough(frame, selected):
                selected.append(frame)

        return selected

    def _is_diverse_enough(
        self, candidate: FrameData, selected: List[FrameData]
    ) -> bool:
        """Check if candidate frame is diverse enough from selected frames.

        Args:
            candidate: Frame to check
            selected: Already selected frames

        Returns:
            True if candidate is diverse enough
        """
        if not selected:
            return True

        # Check temporal diversity (avoid frames too close in time)
        candidate_timestamp = candidate.source_info.video_timestamp

        for selected_frame in selected:
            selected_timestamp = selected_frame.source_info.video_timestamp
            time_diff = abs(candidate_timestamp - selected_timestamp)

            # If frames are within 2 seconds of each other, consider them too similar
            if time_diff < 2.0:
                return False

        return True

    def _calculate_pose_frame_score(self, frame: FrameData) -> float:
        """Calculate selection score for pose category frames (full frames).

        Args:
            frame: Frame to score

        Returns:
            Combined score (0.0 - 1.0)
        """
        if frame.quality_metrics is None:
            return 0.0

        quality_score = frame.quality_metrics.overall_quality

        # For pose frames, we care about overall frame quality
        # and pose detection confidence
        pose_score = 0.0
        if frame.pose_detections:
            best_pose = frame.get_best_pose()
            if best_pose:
                pose_score = best_pose.confidence

        # Combine scores
        final_score = (
            self.criteria.quality_weight * quality_score
            + (1.0 - self.criteria.quality_weight) * pose_score
        )

        return min(1.0, max(0.0, final_score))

    def _calculate_head_angle_frame_score(self, frame: FrameData) -> float:
        """Calculate selection score for head angle category frames (face crops).

        Args:
            frame: Frame to score

        Returns:
            Combined score (0.0 - 1.0)
        """
        if frame.quality_metrics is None:
            return 0.0

        quality_score = frame.quality_metrics.overall_quality

        # For head angle frames, prioritize face size and quality
        face_size_score = 0.0
        if frame.face_detections:
            best_face = frame.get_best_face()
            if best_face and frame.image_properties:
                # Calculate face area ratio
                face_area = best_face.area
                frame_area = frame.image_properties.total_pixels
                face_ratio = face_area / frame_area if frame_area > 0 else 0.0
                # Normalize face ratio to 0-1 score (faces should be 5-50% of frame)
                face_size_score = min(1.0, max(0.0, (face_ratio - 0.05) / 0.45))

        # Head pose confidence
        head_pose_score = 0.0
        if frame.head_poses:
            best_head_pose = frame.get_best_head_pose()
            if best_head_pose:
                head_pose_score = best_head_pose.confidence

        # Combine scores with emphasis on face size for head angle categories
        final_score = (
            self.criteria.quality_weight * quality_score
            + self.criteria.face_size_weight * face_size_score
            + (1.0 - self.criteria.quality_weight - self.criteria.face_size_weight)
            * head_pose_score
        )

        return min(1.0, max(0.0, final_score))

    def _create_selection_rationale(
        self,
        category_name: str,
        category_type: str,
        total_candidates: int,
        selected_count: int,
        quality_range: Tuple[float, float],
        average_quality: float,
    ) -> str:
        """Create human-readable rationale for selection.

        Args:
            category_name: Name of the category
            category_type: Type of category
            total_candidates: Total number of candidate frames
            selected_count: Number of frames selected
            quality_range: Range of quality scores
            average_quality: Average quality score

        Returns:
            Selection rationale string
        """
        rationale_parts = [
            f"Selected {selected_count} of {total_candidates} candidate frames",
            f"for {category_type} category '{category_name}'.",
        ]

        if selected_count > 0:
            rationale_parts.extend(
                [
                    f"Quality range: {quality_range[0]:.2f} - {quality_range[1]:.2f}",
                    f"(average: {average_quality:.2f}).",
                ]
            )

            if category_type == "head_angle":
                rationale_parts.append(
                    "Selection prioritized face size and image quality."
                )
            else:
                rationale_parts.append(
                    "Selection prioritized overall frame and pose quality."
                )

            if selected_count < total_candidates:
                rationale_parts.append(
                    f"Diversity filtering applied to avoid similar frames."
                )

        return " ".join(rationale_parts)

    def _update_frame_selection_metadata(
        self,
        pose_selections: Dict[str, CategorySelection],
        head_angle_selections: Dict[str, CategorySelection],
    ) -> None:
        """Update frame metadata with selection information.

        Args:
            pose_selections: Pose category selections
            head_angle_selections: Head angle category selections
        """
        # Use a set to track which frames have been updated to prevent duplicate ranks
        updated_frames = set()

        # Update pose selection metadata
        for category_name, selection in pose_selections.items():
            for rank, frame in enumerate(selection.selected_frames, 1):
                frame.selections.selected_for_poses.append(category_name)
                frame.selections.final_output = True

                # Only set rank if it hasn't been set by a higher-priority category
                if frame.frame_id not in updated_frames:
                    frame.selections.selection_rank = rank
                    updated_frames.add(frame.frame_id)

        # Update head angle selection metadata
        # These are for crops, so they don't conflict with pose ranks
        for category_name, selection in head_angle_selections.items():
            for rank, frame in enumerate(selection.selected_frames, 1):
                # Add to head angle selections, but don't overwrite primary pose rank
                if category_name not in frame.selections.selected_for_head_angles:
                    frame.selections.selected_for_head_angles.append(category_name)
                frame.selections.final_output = True

    def _create_empty_summary(self, total_candidates: int) -> SelectionSummary:
        """Create empty selection summary when no frames are usable.

        Args:
            total_candidates: Total number of candidate frames

        Returns:
            Empty SelectionSummary
        """
        return SelectionSummary(
            total_candidates=total_candidates,
            total_selected=0,
            pose_selections={},
            head_angle_selections={},
            selection_criteria=self.criteria,
            processing_notes=["No usable frames found for selection"],
        )


def create_frame_selector(criteria: SelectionCriteria) -> FrameSelector:
    """Factory function to create FrameSelector instance.

    Args:
        criteria: Selection criteria

    Returns:
        Configured FrameSelector instance
    """
    return FrameSelector(criteria)
