"""Person-based selection using positional identity strategy.

This module implements the PersonSelector class which groups persons by
person_id across frames and applies quality-first selection with temporal
diversity constraints following the specification Section 5.2-5.3.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..data.config import PersonSelectionCriteria
from ..data.person import Person, BodyUnknown
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..data.frame_data import FrameData

logger = get_logger("person_selector")


@dataclass
class PersonCandidate:
    """A person detection candidate for selection."""

    frame: "FrameData"
    person: Person

    @property
    def person_id(self) -> int:
        """Get person_id for grouping."""
        return self.person.person_id

    @property
    def quality_score(self) -> float:
        """Get quality score for ranking."""
        return self.person.quality.overall_quality

    @property
    def timestamp(self) -> float:
        """Get frame timestamp for temporal diversity."""
        return self.frame.timestamp

    def get_pose_classifications(self) -> List[str]:
        """Get pose classifications for this person."""
        if isinstance(self.person.body, BodyUnknown):
            return []
        
        classifications = []
        for classification, _ in self.person.body.pose_classifications:
            classifications.append(classification)
        return classifications

    def get_primary_pose_classification(self) -> Optional[str]:
        """Get the highest confidence pose classification."""
        if isinstance(self.person.body, BodyUnknown):
            return None
            
        if not self.person.body.pose_classifications:
            return None
            
        # Find the pose classification with the highest confidence
        best_pose = max(self.person.body.pose_classifications, key=lambda x: x[1])
        return best_pose[0]

    def get_head_direction(self) -> Optional[str]:
        """Get head direction for this person."""
        if self.person.head_pose and self.person.head_pose.direction:
            return self.person.head_pose.direction
        return None


@dataclass
class PersonSelection:
    """A selected person instance for output generation."""

    frame_data: "FrameData"
    person_id: int
    person: Person
    selection_score: float
    category: str  # Category name (pose name, head direction, or "minimum"/"additional")

    @property
    def timestamp(self) -> float:
        """Get frame timestamp."""
        return self.frame_data.timestamp


class PersonSelector:
    """Selects diverse person instances using quality metrics and category diversity."""

    def __init__(self, criteria: Optional[PersonSelectionCriteria] = None):
        """Initialize with selection criteria.

        Args:
            criteria: Selection criteria (uses default if None)
        """
        if criteria is None:
            from ..data.config import PersonSelectionCriteria

            self.criteria = PersonSelectionCriteria()
        else:
            self.criteria = criteria

        self.logger = logger.getChild(f"{self.__class__.__name__}")

        # Log configuration
        self.logger.info(
            f"PersonSelector initialized: min={self.criteria.min_instances_per_person}, "
            f"max={self.criteria.max_instances_per_person}, "
            f"poses_enabled={self.criteria.enable_pose_categories}, "
            f"head_angles_enabled={self.criteria.enable_head_angle_categories}, "
            f"temporal_diversity={self.criteria.temporal_diversity_threshold}s"
        )

    def select_persons(self, frames: List["FrameData"]) -> List[PersonSelection]:
        """Select diverse person instances from frames.

        Uses category-based diversity selection when enabled, falling back to
        quality-first selection with temporal diversity.

        Args:
            frames: List of FrameData objects containing person detections

        Returns:
            List of PersonSelection objects representing selected instances
        """
        if not frames:
            self.logger.warning("❌ No frames provided for person selection")
            return []

        start_time = time.time()
        self.logger.info(f"🔧 Starting person selection from {len(frames)} frames")

        try:
            # Step 1: Extract and group persons by person_id
            person_groups = self.extract_and_group_persons(frames)

            if not person_groups:
                self.logger.warning("❌ No valid persons found in frames")
                return []

            self.logger.info(
                f"📍 Found {len(person_groups)} unique person IDs: "
                f"{sorted(person_groups.keys())}"
            )

            # Step 2: Select best instances for each person
            all_selections = []
            for person_id, candidates in person_groups.items():
                person_selections = self.select_best_instances_for_person(
                    person_id, candidates
                )
                all_selections.extend(person_selections)

                self.logger.debug(
                    f"✅ Person {person_id}: selected {len(person_selections)} "
                    f"instances from {len(candidates)} candidates"
                )

            # Step 3: Apply global max_total_selections limit
            if len(all_selections) > self.criteria.max_total_selections:
                self.logger.info(
                    f"🔢 Applying global limit: {len(all_selections)} → "
                    f"{self.criteria.max_total_selections} selections"
                )

                # Sort by quality score (descending) and keep top N
                all_selections.sort(key=lambda s: s.selection_score, reverse=True)
                all_selections = all_selections[: self.criteria.max_total_selections]

            # Log final statistics
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"✅ Person selection completed: {len(all_selections)} "
                f"selections in {processing_time:.1f}ms"
            )

            # Log selection breakdown by category
            category_counts = defaultdict(int)
            for selection in all_selections:
                category_counts[selection.category] += 1

            category_summary = ", ".join(
                [f"{cat}: {count}" for cat, count in category_counts.items()]
            )
            self.logger.debug(f"📊 Selection breakdown: {category_summary}")

            return all_selections

        except Exception as e:
            self.logger.error(f"❌ Person selection failed: {e}")
            return []

    def extract_and_group_persons(
        self, frames: List["FrameData"]
    ) -> Dict[int, List[PersonCandidate]]:
        """Extract persons from frames and group by person_id.

        Args:
            frames: List of FrameData objects

        Returns:
            Dictionary mapping person_id to list of PersonCandidate objects
        """
        person_groups = defaultdict(list)
        total_persons = 0

        for frame in frames:
            if not hasattr(frame, "persons") or not frame.persons:
                continue

            for person in frame.persons:
                # Apply quality threshold filter
                if person.quality.overall_quality < self.criteria.min_quality_threshold:
                    self.logger.debug(
                        f"✗ Person {person.person_id} in frame {frame.frame_id} "
                        f"below quality threshold: {person.quality.overall_quality:.3f} "
                        f"< {self.criteria.min_quality_threshold}"
                    )
                    continue

                candidate = PersonCandidate(frame=frame, person=person)
                person_groups[person.person_id].append(candidate)
                total_persons += 1

        self.logger.info(
            f"📍 Extracted {total_persons} valid persons across "
            f"{len(person_groups)} person IDs"
        )

        # Log person distribution
        for person_id, candidates in person_groups.items():
            self.logger.debug(f"📊 Person {person_id}: {len(candidates)} candidates")

        return person_groups

    def select_best_instances_for_person(
        self, person_id: int, candidates: List[PersonCandidate]
    ) -> List[PersonSelection]:
        """Select best instances for a single person using category-based diversity.

        This method implements category-based selection when enabled, otherwise
        falls back to quality-first selection with temporal diversity.

        Args:
            person_id: The person ID
            candidates: List of PersonCandidate objects for this person

        Returns:
            List of PersonSelection objects for this person
        """
        if not candidates:
            return []

        self.logger.debug(
            f"🔧 Selecting instances for person {person_id} "
            f"from {len(candidates)} candidates"
        )

        # Check if category-based selection is enabled
        if self.criteria.enable_pose_categories or self.criteria.enable_head_angle_categories:
            return self._select_with_category_diversity(person_id, candidates)
        else:
            return self._select_with_quality_priority(person_id, candidates)

    def _select_with_category_diversity(
        self, person_id: int, candidates: List[PersonCandidate]
    ) -> List[PersonSelection]:
        """Select instances using category-based diversity with independent pose and head angle selection."""
        self.logger.debug(f"🎯 Using category-based selection for person {person_id}")
        
        selected = []
        
        # Step 1: Group candidates by categories
        pose_groups = defaultdict(list)
        head_angle_groups = defaultdict(list)
        
        for candidate in candidates:
            # Group by pose categories if enabled
            if self.criteria.enable_pose_categories:
                primary_pose = candidate.get_primary_pose_classification()
                if primary_pose:
                    pose_groups[primary_pose].append(candidate)
            
            # Group by head angle categories if enabled
            if self.criteria.enable_head_angle_categories:
                head_direction = candidate.get_head_direction()
                if head_direction:
                    head_angle_groups[head_direction].append(candidate)
        
        # Step 2: Independent pose category selection
        pose_selections = []
        if self.criteria.enable_pose_categories and pose_groups:
            pose_selections = self._select_pose_diversity(
                pose_groups, person_id, self.criteria.min_poses_per_person
            )
            selected.extend(pose_selections)
            self.logger.debug(
                f"✓ Person {person_id}: selected {len(pose_selections)} pose instances "
                f"across {len(set(s.category.replace('pose_', '') for s in pose_selections))} poses"
            )
        
        # Step 3: Independent head angle category selection
        head_angle_selections = []
        if self.criteria.enable_head_angle_categories and head_angle_groups:
            head_angle_selections = self._select_head_angle_diversity(
                head_angle_groups, person_id, self.criteria.min_head_angles_per_person
            )
            selected.extend(head_angle_selections)
            self.logger.debug(
                f"✓ Person {person_id}: selected {len(head_angle_selections)} head angle instances "
                f"across {len(set(s.category.replace('head_angle_', '') for s in head_angle_selections))} angles"
            )
        
        # Step 4: Remove duplicates (same frame selected for both pose and head angle)
        unique_selections = self._deduplicate_selections(selected)
        if len(unique_selections) < len(selected):
            self.logger.debug(
                f"✓ Person {person_id}: removed {len(selected) - len(unique_selections)} "
                f"duplicate selections (same frame selected for multiple categories)"
            )
        
        # Step 5: Apply overall limits
        final_selections = self._apply_person_limits(unique_selections, person_id)
        
        # Step 6: Fill with quality-first if under minimum
        if len(final_selections) < self.criteria.min_instances_per_person:
            remaining_needed = self.criteria.min_instances_per_person - len(final_selections)
            quality_selections = self._select_remaining_by_quality(
                candidates, final_selections, [], remaining_needed, person_id
            )
            final_selections.extend(quality_selections)
        
        self.logger.info(
            f"✅ Person {person_id}: selected {len(final_selections)} instances "
            f"using category-based diversity ({len(pose_selections)} poses, "
            f"{len(head_angle_selections)} head angles, {len(final_selections) - len(pose_selections) - len(head_angle_selections)} quality-based)"
        )
        
        return final_selections

    def _select_pose_diversity(
        self, pose_groups: Dict[str, List[PersonCandidate]], 
        person_id: int, min_poses: int
    ) -> List[PersonSelection]:
        """Select diverse pose instances independently."""
        selections = []
        
        # Sort poses by availability (most candidates first) for better selection
        sorted_poses = sorted(
            pose_groups.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Try to get at least min_poses different poses, but aim for maximum diversity
        poses_to_select = min(len(sorted_poses), max(min_poses, len(sorted_poses)))
        
        for pose_name, pose_candidates in sorted_poses[:poses_to_select]:
            # Sort candidates by quality (descending)
            pose_candidates.sort(key=lambda c: c.quality_score, reverse=True)
            
            # Select best candidate from this pose
            best_candidate = pose_candidates[0]
            selection = PersonSelection(
                frame_data=best_candidate.frame,
                person_id=person_id,
                person=best_candidate.person,
                selection_score=best_candidate.quality_score,
                category=f"pose_{pose_name}"
            )
            selections.append(selection)
            
            self.logger.debug(
                f"✓ Person {person_id} pose_{pose_name}: "
                f"frame {best_candidate.frame.frame_id}, quality {best_candidate.quality_score:.3f}"
            )
        
        return selections

    def _select_head_angle_diversity(
        self, head_angle_groups: Dict[str, List[PersonCandidate]], 
        person_id: int, min_head_angles: int
    ) -> List[PersonSelection]:
        """Select diverse head angle instances independently."""
        selections = []
        
        # Sort head angles by availability (most candidates first)
        sorted_head_angles = sorted(
            head_angle_groups.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Try to get at least min_head_angles different angles, but aim for maximum diversity
        angles_to_select = min(len(sorted_head_angles), max(min_head_angles, len(sorted_head_angles)))
        
        for head_angle, angle_candidates in sorted_head_angles[:angles_to_select]:
            # Sort candidates by quality (descending)
            angle_candidates.sort(key=lambda c: c.quality_score, reverse=True)
            
            # Select best candidate from this head angle
            best_candidate = angle_candidates[0]
            selection = PersonSelection(
                frame_data=best_candidate.frame,
                person_id=person_id,
                person=best_candidate.person,
                selection_score=best_candidate.quality_score,
                category=f"head_angle_{head_angle}"
            )
            selections.append(selection)
            
            self.logger.debug(
                f"✓ Person {person_id} head_angle_{head_angle}: "
                f"frame {best_candidate.frame.frame_id}, quality {best_candidate.quality_score:.3f}"
            )
        
        return selections

    def _deduplicate_selections(self, selections: List[PersonSelection]) -> List[PersonSelection]:
        """Remove duplicate selections (same frame selected for multiple categories)."""
        # Group by frame_id and keep the highest scoring selection per frame
        frame_groups = defaultdict(list)
        for selection in selections:
            frame_groups[selection.frame_data.frame_id].append(selection)
        
        deduplicated = []
        for frame_id, frame_selections in frame_groups.items():
            if len(frame_selections) == 1:
                # No duplicates for this frame
                deduplicated.append(frame_selections[0])
            else:
                # Multiple categories selected the same frame - combine categories
                best_selection = max(frame_selections, key=lambda s: s.selection_score)
                # Combine category names
                categories = [s.category for s in frame_selections]
                combined_category = " + ".join(sorted(categories))
                best_selection.category = combined_category
                deduplicated.append(best_selection)
        
        return deduplicated

    def _apply_person_limits(
        self, selections: List[PersonSelection], person_id: int
    ) -> List[PersonSelection]:
        """Apply per-person selection limits."""
        if len(selections) <= self.criteria.max_instances_per_person:
            return selections
        
        # Too many selections - prioritize by quality
        selections.sort(key=lambda s: s.selection_score, reverse=True)
        limited_selections = selections[:self.criteria.max_instances_per_person]
        
        self.logger.debug(
            f"✓ Person {person_id}: applied max limit, reduced from "
            f"{len(selections)} to {len(limited_selections)} selections"
        )
        
        return limited_selections

    def _select_with_quality_priority(
        self, person_id: int, candidates: List[PersonCandidate]
    ) -> List[PersonSelection]:
        """Select instances using quality-first approach with temporal diversity."""
        self.logger.debug(f"📊 Using quality-first selection for person {person_id}")
        
        # Sort candidates by quality score (descending)
        candidates.sort(key=lambda c: c.quality_score, reverse=True)

        selected = []
        selected_timestamps = []

        # Apply temporal diversity filtering to ALL selections (including minimum)
        for candidate in candidates:
            # Check if we've reached max instances limit
            if len(selected) >= self.criteria.max_instances_per_person:
                break

            # Check temporal diversity
            if self._is_temporally_diverse(candidate.timestamp, selected_timestamps):
                # Determine category based on whether we've met minimum requirements
                category = "minimum" if len(selected) < self.criteria.min_instances_per_person else "additional"
                
                selection = PersonSelection(
                    frame_data=candidate.frame,
                    person_id=person_id,
                    person=candidate.person,
                    selection_score=candidate.quality_score,
                    category=category,
                )
                selected.append(selection)
                selected_timestamps.append(candidate.timestamp)

                self.logger.debug(
                    f"✓ Person {person_id} {category}: "
                    f"frame {candidate.frame.frame_id}, "
                    f"quality {candidate.quality_score:.3f}, "
                    f"timestamp {candidate.timestamp:.1f}s"
                )

        # Check if we met minimum requirements after temporal filtering
        if len(selected) < self.criteria.min_instances_per_person:
            self.logger.warning(
                f"⚠️  Person {person_id}: only {len(selected)} instances selected "
                f"(below minimum {self.criteria.min_instances_per_person}) due to temporal diversity filtering. "
                f"Consider reducing temporal_diversity_threshold ({self.criteria.temporal_diversity_threshold}s) "
                f"or increasing min_instances_per_person."
            )

        self.logger.info(
            f"✅ Person {person_id}: selected {len(selected)} instances "
            f"with quality-first selection"
        )

        return selected

    def _is_temporally_diverse(self, candidate_timestamp: float, used_timestamps: List[float]) -> bool:
        """Check if candidate is temporally diverse from already selected instances."""
        if self.criteria.temporal_diversity_threshold <= 0:
            return True
            
        if not used_timestamps:
            return True
            
        for existing_timestamp in used_timestamps:
            time_diff = abs(candidate_timestamp - existing_timestamp)
            if time_diff < self.criteria.temporal_diversity_threshold:
                return False
                
        return True

    def _select_remaining_by_quality(
        self, all_candidates: List[PersonCandidate], 
        already_selected: List[PersonSelection],
        used_timestamps: List[float], remaining_needed: int, person_id: int
    ) -> List[PersonSelection]:
        """Select remaining instances by quality to meet minimum requirements."""
        # Get frame IDs that are already selected
        selected_frame_ids = {s.frame_data.frame_id for s in already_selected}
        
        # Filter out already selected candidates
        available_candidates = [
            c for c in all_candidates 
            if c.frame.frame_id not in selected_frame_ids
        ]
        
        # Sort by quality (descending)
        available_candidates.sort(key=lambda c: c.quality_score, reverse=True)
        
        selections = []
        for candidate in available_candidates:
            if len(selections) >= remaining_needed:
                break
                
            # Check temporal diversity
            if self._is_temporally_diverse(candidate.timestamp, used_timestamps):
                selection = PersonSelection(
                    frame_data=candidate.frame,
                    person_id=person_id,
                    person=candidate.person,
                    selection_score=candidate.quality_score,
                    category="minimum"
                )
                selections.append(selection)
                used_timestamps.append(candidate.timestamp)
                
                self.logger.debug(
                    f"✓ Person {person_id} minimum: "
                    f"frame {candidate.frame.frame_id}, quality {candidate.quality_score:.3f}"
                )
        
        return selections
