"""Frame selection logic for choosing best frames based on quality and diversity.

This module implements the FrameSelector class that ranks and selects the best
frames from each pose category and head angle category based on comprehensive
quality metrics and diversity considerations.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

from ..data.frame_data import FrameData
from ..data.detection_results import QualityMetrics, FaceDetection, CloseupDetection
from ..utils.logging import get_logger


@dataclass
class SelectionCriteria:
    """Criteria for frame selection."""

    min_frames_per_category: int
    max_frames_per_category: int
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


@dataclass
class SelectionContext:
    """State container for the frame selection process.
    
    This class encapsulates all the state and intermediate results during
    the frame selection process, making the main orchestration method
    cleaner and more testable.
    """
    
    # Input data
    candidate_frames: List[FrameData]
    progress_callback: Optional[Callable[[str], None]]
    interruption_check: Optional[Callable[[], None]]
    
    # Intermediate processing results
    usable_frames: List[FrameData] = field(default_factory=list)
    pose_groups: Dict[str, List[FrameData]] = field(default_factory=dict)
    head_angle_groups: Dict[str, List[FrameData]] = field(default_factory=dict)
    
    # Final selection results
    pose_selections: Dict[str, CategorySelection] = field(default_factory=dict)
    head_angle_selections: Dict[str, CategorySelection] = field(default_factory=dict)


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
        interruption_check: Optional[Callable[[], None]] = None,
    ) -> SelectionSummary:
        """Select best frames from candidates based on quality and diversity.

        This is the main orchestration method that coordinates the entire frame
        selection process using a clean, phase-based approach.

        Args:
            candidate_frames: List of candidate frames to select from
            progress_callback: Optional callback for progress updates
            interruption_check: Optional callback to check for interruption

        Returns:
            SelectionSummary with complete selection results
        """
        # Phase 1: Preparation - Filter and group frames
        context = self._prepare_selection_context(
            candidate_frames, progress_callback, interruption_check
        )
        
        # Early exit if no usable frames
        if not context.usable_frames:
            return self._create_empty_summary(len(candidate_frames))

        # Phase 2: Process pose categories (with priority claiming)
        if context.progress_callback:
            context.progress_callback("Selecting best frames for pose categories...")
            
        context.pose_selections = self._process_categories(
            context,
            context.pose_groups,
            "pose",
            self._calculate_pose_frame_score,
            use_priority_claiming=True,
        )

        # Phase 3: Process head angle categories (without priority claiming)
        if context.progress_callback:
            context.progress_callback("Selecting best frames for head angle categories...")
            
        context.head_angle_selections = self._process_categories(
            context,
            context.head_angle_groups,
            "head_angle",
            self._calculate_head_angle_frame_score,
            use_priority_claiming=False,
        )

        # Phase 4: Finalization
        self._finalize_selection(context)
        
        # Phase 5: Update metadata and create summary
        if context.progress_callback:
            context.progress_callback("Updating frame selection metadata...")
            
        self._update_frame_selection_metadata(
            context.pose_selections, context.head_angle_selections
        )
        
        summary = self._create_selection_summary(context)
        
        self.logger.info(
            f"Frame selection completed: {summary.total_selected} unique frames selected"
        )
        return summary

    def group_by_pose(self, frames: List[FrameData], interruption_check: Optional[Callable[[], None]] = None) -> Dict[str, List[FrameData]]:
        """Group frames by pose classification.

        Args:
            frames: List of frames to group
            interruption_check: Optional callback to check for interruption

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

            # Check for interruption during pose processing
            if interruption_check and frames.index(frame) % 3 == 0:
                interruption_check()

        # Convert defaultdict to regular dict for cleaner output
        return dict(pose_groups)

    def group_by_head_angle(
        self, frames: List[FrameData], interruption_check: Optional[Callable[[], None]] = None
    ) -> Dict[str, List[FrameData]]:
        """Group frames by head angle direction.

        Args:
            frames: List of frames to group
            interruption_check: Optional callback to check for interruption

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

            # Check for interruption during head angle processing
            if interruption_check and frames.index(frame) % 3 == 0:
                interruption_check()

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

    def _filter_usable_frames(self, frames: List[FrameData], interruption_check: Optional[Callable[[], None]] = None) -> List[FrameData]:
        """Filter frames to only include usable ones.

        Args:
            frames: Input frames to filter
            interruption_check: Optional callback to check for interruption

        Returns:
            List of frames that meet usability criteria
        """
        usable = []

        for frame in frames:
            # Must have faces
            if not frame.has_faces():
                frame.selections.rejection_reason = "no_faces_detected"
                continue

            # Must have quality metrics
            if frame.quality_metrics is None:
                frame.selections.rejection_reason = "no_quality_assessment"
                continue

            # Must meet minimum quality threshold
            if (
                frame.quality_metrics.overall_quality
                < self.criteria.min_quality_threshold
            ):
                frame.selections.rejection_reason = "below_min_quality_threshold"
                continue

            # Must be marked as usable
            if not frame.quality_metrics.usable:
                frame.selections.rejection_reason = "marked_unusable_by_quality_assessment"
                continue

            usable.append(frame)

            # Check for interruption during frame processing
            if interruption_check and frames.index(frame) % 3 == 0:
                interruption_check()

        return usable



    def _select_diverse_frames(
        self, scored_frames: List[Tuple[FrameData, float]], max_count: int, interruption_check: Optional[Callable[[], None]] = None
    ) -> Tuple[List[FrameData], List[FrameData]]:
        """Select diverse frames to avoid similar selections.

        Args:
            scored_frames: List of (frame, score) tuples sorted by score
            max_count: Maximum number of frames to select
            interruption_check: Optional callback to check for interruption

        Returns:
            Tuple of (selected_frames, rejected_diversity_frames)
        """
        if not scored_frames:
            return [], []

        selected = []
        rejected_diversity = []

        for i, (frame, score) in enumerate(scored_frames):
            if len(selected) >= max_count:
                # Frames that exceed max count are not processed further
                break

            # Check for interruption during selection
            if interruption_check and i % 5 == 0:
                interruption_check()
                
            # Check diversity against already selected frames
            if self._is_diverse_enough(frame, selected):
                selected.append(frame)
            else:
                # Frame rejected due to insufficient diversity
                rejected_diversity.append(frame)

        return selected, rejected_diversity

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

    def _calculate_pose_frame_score(self, frame: FrameData, return_breakdown: bool = False) -> float | Tuple[float, Dict[str, float]]:
        """Calculate selection score for pose category frames (full frames).

        Args:
            frame: Frame to score
            return_breakdown: If True, return (score, breakdown) tuple

        Returns:
            Combined score (0.0 - 1.0) or (score, breakdown) if return_breakdown=True
        """
        if frame.quality_metrics is None:
            if return_breakdown:
                return 0.0, {"quality": 0.0, "pose_confidence": 0.0}
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

        final_score = min(1.0, max(0.0, final_score))
        
        if return_breakdown:
            breakdown = {
                "quality": quality_score,
                "pose_confidence": pose_score
            }
            return final_score, breakdown
        
        return final_score

    def _calculate_head_angle_frame_score(self, frame: FrameData, return_breakdown: bool = False) -> float | Tuple[float, Dict[str, float]]:
        """Calculate selection score for head angle category frames (face crops).

        Args:
            frame: Frame to score
            return_breakdown: If True, return (score, breakdown) tuple

        Returns:
            Combined score (0.0 - 1.0) or (score, breakdown) if return_breakdown=True
        """
        if frame.quality_metrics is None:
            if return_breakdown:
                return 0.0, {"quality": 0.0, "face_size": 0.0, "head_pose_confidence": 0.0}
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

        final_score = min(1.0, max(0.0, final_score))
        
        if return_breakdown:
            breakdown = {
                "quality": quality_score,
                "face_size": face_size_score,
                "head_pose_confidence": head_pose_score
            }
            return final_score, breakdown
        
        return final_score

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
            category_key = f"pose_{category_name}"
            for rank, frame in enumerate(selection.selected_frames, 1):
                frame.selections.selected_for_poses.append(category_name)
                frame.selections.final_output = True

                # Only set rank if it hasn't been set by a higher-priority category
                if frame.frame_id not in updated_frames:
                    frame.selections.selection_rank = rank
                    updated_frames.add(frame.frame_id)
                    
                # Set primary selection category and final score (NEW)
                if frame.selections.primary_selection_category is None:
                    frame.selections.primary_selection_category = category_key
                    frame.selections.final_selection_score = frame.selections.category_scores.get(category_key, 0.0)

        # Update head angle selection metadata
        # These are for crops, so they don't conflict with pose ranks
        for category_name, selection in head_angle_selections.items():
            category_key = f"head_angle_{category_name}"
            for rank, frame in enumerate(selection.selected_frames, 1):
                # Add to head angle selections, but don't overwrite primary pose rank
                if category_name not in frame.selections.selected_for_head_angles:
                    frame.selections.selected_for_head_angles.append(category_name)
                frame.selections.final_output = True
                
                # Set primary selection category and final score only if not already set by pose selection (NEW)
                if frame.selections.primary_selection_category is None:
                    frame.selections.primary_selection_category = category_key
                    frame.selections.final_selection_score = frame.selections.category_scores.get(category_key, 0.0)

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

    def _score_and_rank_candidates_for_category(
        self,
        frames: List[FrameData],
        category_name: str,
        category_type: str,
        score_function: Callable[[FrameData], float],
        interruption_check: Optional[Callable[[], None]] = None,
    ) -> List[Tuple[FrameData, float]]:
        """Score and rank candidates for a specific category with full transparency.

        This method separates scoring/ranking from selection, populating the new
        transparency fields on each frame for complete auditability.

        Args:
            frames: Candidate frames for this category
            category_name: Name of the category (e.g., "standing", "front")
            category_type: Type of category ("pose" or "head_angle")
            score_function: Function to calculate frame score with breakdown
            interruption_check: Optional callback to check for interruption

        Returns:
            List of (frame, score) tuples sorted by score (descending)
        """
        if not frames:
            return []

        # Generate unique category key
        category_key = f"{category_type}_{category_name}"
        
        # Score all frames and populate transparency fields
        scored_frames = []
        for i, frame in enumerate(frames):
            # Check for interruption during scoring
            if interruption_check and i % 5 == 0:
                interruption_check()
                
            # Get score with breakdown for transparency
            score_result = score_function(frame, return_breakdown=True)
            if isinstance(score_result, tuple):
                score, breakdown = score_result
            else:
                # Fallback for functions that don't support breakdown yet
                score = score_result
                breakdown = {}
            
            # Populate transparency fields (NEW)
            frame.selections.category_scores[category_key] = score
            frame.selections.category_score_breakdowns[category_key] = breakdown
            
            scored_frames.append((frame, score))

        # Sort by score (highest first) 
        scored_frames.sort(key=lambda x: x[1], reverse=True)

        # Populate ranking fields (NEW)
        for rank, (frame, score) in enumerate(scored_frames, 1):
            frame.selections.category_ranks[category_key] = rank

        return scored_frames

    def _prepare_selection_context(
        self,
        candidate_frames: List[FrameData],
        progress_callback: Optional[Callable[[str], None]] = None,
        interruption_check: Optional[Callable[[], None]] = None,
    ) -> SelectionContext:
        """Prepare the selection context with filtered and grouped frames.
        
        Args:
            candidate_frames: Input frames to process
            progress_callback: Optional callback for progress updates
            interruption_check: Optional callback to check for interruption
            
        Returns:
            SelectionContext with usable frames and category groups prepared
        """
        context = SelectionContext(
            candidate_frames=candidate_frames,
            progress_callback=progress_callback,
            interruption_check=interruption_check
        )
        
        self.logger.info(f"Starting frame selection from {len(candidate_frames)} candidates")

        # Check for interruption at the start
        if interruption_check:
            interruption_check()

        if progress_callback:
            progress_callback("Filtering candidate frames...")

        # Filter to usable frames only
        context.usable_frames = self._filter_usable_frames(candidate_frames, interruption_check)
        self.logger.info(f"Found {len(context.usable_frames)} usable frames after filtering")

        if not context.usable_frames:
            return context

        # Check for interruption after filtering
        if interruption_check:
            interruption_check()

        if progress_callback:
            progress_callback("Grouping frames by categories...")

        # Group frames by categories
        context.pose_groups = self.group_by_pose(context.usable_frames, interruption_check)
        context.head_angle_groups = self.group_by_head_angle(context.usable_frames, interruption_check)

        # Check for interruption after grouping
        if interruption_check:
            interruption_check()

        return context

    def _calculate_selection_statistics(
        self, 
        candidate_frames: List[FrameData]
    ) -> Tuple[Tuple[float, float], float]:
        """Calculate quality statistics for a set of candidate frames.
        
        Args:
            candidate_frames: Frames to calculate statistics for
            
        Returns:
            Tuple of (quality_range, average_quality)
        """
        quality_scores = [
            f.quality_metrics.overall_quality 
            for f in candidate_frames 
            if f.quality_metrics
        ]
        quality_range = (
            (min(quality_scores), max(quality_scores)) 
            if quality_scores else (0.0, 0.0)
        )
        average_quality = (
            sum(quality_scores) / len(quality_scores) 
            if quality_scores else 0.0
        )
        return quality_range, average_quality

    def _create_category_selection_object(
        self,
        category_name: str,
        category_type: str,
        selected_frames: List[FrameData],
        candidate_frames: List[FrameData],
    ) -> CategorySelection:
        """Create a CategorySelection object with all required metadata.
        
        Args:
            category_name: Name of the category
            category_type: Type of category ("pose" or "head_angle")
            selected_frames: Frames that were selected
            candidate_frames: All candidate frames for this category
            
        Returns:
            CategorySelection object with complete metadata
        """
        quality_range, average_quality = self._calculate_selection_statistics(candidate_frames)
        
        rationale = self._create_selection_rationale(
            category_name,
            category_type,
            len(candidate_frames),
            len(selected_frames),
            quality_range,
            average_quality,
        )

        return CategorySelection(
            category_name=category_name,
            category_type=category_type,
            selected_frames=selected_frames,
            total_candidates=len(candidate_frames),
            selection_rationale=rationale,
            quality_range=quality_range,
            average_quality=average_quality,
        )

    def _process_categories(
        self,
        context: SelectionContext,
        category_groups: Dict[str, List[FrameData]],
        category_type: str,
        score_function: Callable,
        use_priority_claiming: bool = False,
    ) -> Dict[str, CategorySelection]:
        """Process categories generically, handling both poses and head angles.
        
        Args:
            context: Selection context with all state
            category_groups: Dictionary of category_name -> frames
            category_type: "pose" or "head_angle"
            score_function: Function to score frames for this category type
            use_priority_claiming: Whether to check for already-claimed frames
            
        Returns:
            Dictionary of category_name -> CategorySelection
        """
        selections = {}
        
        for category_name, candidate_frames in category_groups.items():
            if context.interruption_check:
                context.interruption_check()
                
            if context.progress_callback:
                context.progress_callback(f"Processing {category_type} category: {category_name}")

            self.logger.info(f"Processing {category_type} '{category_name}' with {len(candidate_frames)} candidates")

            # Step 1: Score and rank all candidates for this category
            scored_frames = self._score_and_rank_candidates_for_category(
                candidate_frames, category_name, category_type, score_function
            )

            # Step 2: Filter available candidates (priority claiming for poses only)
            if use_priority_claiming:
                available_candidates = [
                    (frame, score) for frame, score in scored_frames
                    if frame.selections.primary_selection_category is None
                ]
                # Log competition for frames lost to higher priority
                for frame, _ in scored_frames:
                    if frame.selections.primary_selection_category is not None:
                        category_key = f"{category_type}_{category_name}"
                        frame.selections.selection_competition[category_key] = "lost_to_higher_priority"
            else:
                available_candidates = scored_frames

            # Step 3: Select diverse frames
            if available_candidates:
                selected_frames, rejected_diversity_frames = self._select_diverse_frames(
                    available_candidates, self.criteria.max_frames_per_category
                )
            else:
                selected_frames, rejected_diversity_frames = [], []

            # Step 4: Update competition tracking
            category_key = f"{category_type}_{category_name}"
            self._update_competition_tracking_for_category(
                available_candidates, selected_frames, rejected_diversity_frames, category_key
            )

            # Step 5: Create the CategorySelection object
            selections[category_name] = self._create_category_selection_object(
                category_name, category_type, selected_frames, candidate_frames
            )

            self.logger.info(f"Selected {len(selected_frames)} frames for {category_type} '{category_name}'")

        return selections

    def _update_competition_tracking_for_category(
        self,
        scored_frames: List[Tuple[FrameData, float]],
        selected_frames: List[FrameData],
        rejected_diversity_frames: List[FrameData],
        category_key: str,
    ) -> None:
        """Update competition tracking and metadata for all frames in a category.

        Args:
            scored_frames: All frames that were scored for this category
            selected_frames: Frames that were ultimately selected
            rejected_diversity_frames: Frames rejected due to insufficient diversity
            category_key: The category key (e.g., "pose_standing", "head_angle_front")
        """
        # Create sets for quick lookup
        selected_frame_ids = {frame.frame_id for frame in selected_frames}
        rejected_diversity_frame_ids = {frame.frame_id for frame in rejected_diversity_frames}

        # Process each scored frame
        for frame, score in scored_frames:
            if frame.frame_id in selected_frame_ids:
                # Subtask 3.1: Process newly selected frames
                frame.selections.selection_competition[category_key] = "selected"
                frame.selections.primary_selection_category = category_key
                frame.selections.final_selection_score = score
                
                # Set selection rank (1-based) within the selected group for this category
                selection_rank = selected_frames.index(frame) + 1
                frame.selections.selection_rank = selection_rank
                
            elif frame.frame_id in rejected_diversity_frame_ids:
                # Subtask 3.2: Process frames rejected for diversity
                frame.selections.selection_competition[category_key] = "rejected_insufficient_diversity"
                
                # Set rejection reason only if not already set with a more specific reason
                if frame.selections.rejection_reason is None:
                    frame.selections.rejection_reason = "insufficient_diversity"
                    
            else:
                # Subtask 3.3: Process frames that were not top-ranked
                frame.selections.selection_competition[category_key] = "not_top_ranked"
                
                # Set rejection reason only if not already set with a more specific reason
                if frame.selections.rejection_reason is None:
                    frame.selections.rejection_reason = "not_top_ranked"

    def _finalize_selection(self, context: SelectionContext) -> None:
        """Apply final rejection reasons to unselected frames.
        
        Args:
            context: Selection context with all state and results
        """
        if context.progress_callback:
            context.progress_callback("Finalizing selection results...")
            
        # Apply final rejection reasons for frames that were never selected
        for frame in context.usable_frames:
            if (frame.selections.primary_selection_category is None 
                and frame.selections.rejection_reason is None):
                frame.selections.rejection_reason = "not_selected"

    def _count_unique_selected_frames(self, context: SelectionContext) -> int:
        """Count total unique frames that were selected across all categories.
        
        Args:
            context: Selection context with selection results
            
        Returns:
            Number of unique selected frames
        """
        selected_frame_ids = set()
        
        # Count from pose selections
        for selection in context.pose_selections.values():
            for frame in selection.selected_frames:
                selected_frame_ids.add(frame.frame_id)
                
        # Count from head angle selections
        for selection in context.head_angle_selections.values():
            for frame in selection.selected_frames:
                selected_frame_ids.add(frame.frame_id)
                
        return len(selected_frame_ids)

    def _create_selection_summary(self, context: SelectionContext) -> SelectionSummary:
        """Create the final selection summary.
        
        Args:
            context: Selection context with all results
            
        Returns:
            Complete SelectionSummary object
        """
        total_selected = self._count_unique_selected_frames(context)
        
        processing_notes = []
        if not context.usable_frames:
            processing_notes.append("No usable frames found for selection")
        elif total_selected == 0:
            processing_notes.append("No frames met selection criteria")
        else:
            processing_notes.append(f"Successfully selected {total_selected} frames from {len(context.usable_frames)} candidates")

        return SelectionSummary(
            total_candidates=len(context.usable_frames),
            total_selected=total_selected,
            pose_selections=context.pose_selections,
            head_angle_selections=context.head_angle_selections,
            selection_criteria=self.criteria,
            processing_notes=processing_notes,
        )


def create_frame_selector(criteria: SelectionCriteria) -> FrameSelector:
    """Factory function to create FrameSelector instance.

    Args:
        criteria: Selection criteria

    Returns:
        Configured FrameSelector instance
    """
    return FrameSelector(criteria)
