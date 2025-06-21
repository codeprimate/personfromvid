"""Person-based selection using positional identity strategy.

This module implements the PersonSelector class which groups persons by
person_id across frames and applies quality-first selection with temporal
diversity constraints following the specification Section 5.2-5.3.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from ..data.config import PersonSelectionCriteria
from ..data.person import Person
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


@dataclass
class PersonSelection:
    """A selected person instance for output generation."""

    frame_data: "FrameData"
    person_id: int
    person: Person
    selection_score: float
    category: str  # "minimum", "additional", "quality_ranked"

    @property
    def timestamp(self) -> float:
        """Get frame timestamp."""
        return self.frame_data.timestamp


class PersonSelector:
    """Selects best person instances using positional identity strategy."""

    def __init__(self, criteria: Optional[PersonSelectionCriteria] = None):
        """Initialize PersonSelector with selection criteria.

        Args:
            criteria: PersonSelectionCriteria configuration. If None, uses default.
        """
        self.logger = get_logger("person_selector")

        # Use provided criteria or get default from config
        if criteria is None:
            from ..data.config import get_default_config

            config = get_default_config()
            self.criteria = config.person_selection
        else:
            self.criteria = criteria

        self.logger.debug(
            f"🔧 PersonSelector initialized with criteria: "
            f"min_instances={self.criteria.min_instances_per_person}, "
            f"max_instances={self.criteria.max_instances_per_person}, "
            f"quality_threshold={self.criteria.min_quality_threshold}"
        )

    def select_persons(self, frames: List["FrameData"]) -> List[PersonSelection]:
        """Select best person instances using positional identity strategy.

        Args:
            frames: List of FrameData objects with populated persons

        Returns:
            List of PersonSelection objects for output generation

        Raises:
            ValueError: If frames list is empty or invalid
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
        """Select best instances for a single person using temporal diversity filtering.

        This method now applies temporal diversity filtering to ALL selections,
        including the minimum instances, to avoid outputs with similar timestamps.

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

        # Sort candidates by quality score (descending)
        candidates.sort(key=lambda c: c.quality_score, reverse=True)

        selected = []
        selected_timestamps = []

        # Apply temporal diversity filtering to ALL selections (including minimum)
        for candidate in candidates:
            # Check if we've reached max instances limit
            if len(selected) >= self.criteria.max_instances_per_person:
                break

            # Check temporal diversity against all previously selected instances
            candidate_timestamp = candidate.timestamp
            too_close = False

            # Only apply temporal diversity if threshold > 0 and we have existing selections
            if self.criteria.temporal_diversity_threshold > 0 and selected_timestamps:
                for existing_timestamp in selected_timestamps:
                    time_diff = abs(candidate_timestamp - existing_timestamp)
                    if time_diff < self.criteria.temporal_diversity_threshold:
                        too_close = True
                        self.logger.debug(
                            f"✗ Person {person_id} candidate frame {candidate.frame.frame_id} "
                            f"too close temporally: {time_diff:.1f}s < "
                            f"{self.criteria.temporal_diversity_threshold}s"
                        )
                        break

            if not too_close:
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
                selected_timestamps.append(candidate_timestamp)

                self.logger.debug(
                    f"✓ Person {person_id} {category}: "
                    f"frame {candidate.frame.frame_id}, "
                    f"quality {candidate.quality_score:.3f}, "
                    f"timestamp {candidate_timestamp:.1f}s"
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
            f"with temporal diversity filtering"
        )

        return selected
