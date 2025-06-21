"""Person-based selection pipeline step.

This module implements PersonSelectionStep which uses PersonSelector to select
best person instances using positional identity strategy as an alternative to
frame-based selection, providing backwards compatibility with existing pipeline.
"""

from typing import List

from ...analysis.person_selector import PersonSelection, PersonSelector
from ...data.constants import ALL_SELECTED_PERSONS_KEY
from .base import PipelineStep


class PersonSelectionStep(PipelineStep):
    """Pipeline step for selecting the best person instances using positional identity."""

    @property
    def step_name(self) -> str:
        return "person_selection"

    def execute(self) -> None:
        """Select best person instances based on positional identity and quality."""
        self.state.start_step(self.step_name)

        try:
            # Filter candidate frames: must have persons AND quality metrics (following FrameSelectionStep pattern)
            candidate_frames = [
                f
                for f in self.state.frames
                if hasattr(f, "persons") and f.persons and f.quality_metrics is not None
            ]

            if not candidate_frames:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames with persons and quality assessments for selection"
                    )
                else:
                    self.logger.warning(
                        "⚠️ No frames with persons and quality assessments for selection"
                    )
                self.state.get_step_progress(self.step_name).start(0)
                return

            # Count total person candidates across all frames
            total_person_candidates = sum(len(f.persons) for f in candidate_frames)

            if self.formatter:
                self.formatter.print_info(
                    "🎯 Optimizing person selection...", "targeting"
                )
            else:
                self.logger.info(
                    f"🎯 Selecting from {total_person_candidates} person candidates across {len(candidate_frames)} frames..."
                )

            # Create PersonSelector with configuration from pipeline config
            person_selector = PersonSelector(self.config.person_selection)
            self.state.get_step_progress(self.step_name).start(total_person_candidates)

            current_progress = 0

            def progress_callback(message: str):
                nonlocal current_progress
                self._check_interrupted()
                # Update progress based on person candidates processed
                current_progress = min(current_progress + 1, total_person_candidates)
                self.state.update_step_progress(self.step_name, current_progress)
                if self.formatter:
                    self.formatter.update_progress(1)

            # Execute person selection with progress tracking
            if self.formatter:
                with self.formatter.create_progress_bar(
                    "Selecting persons", total_person_candidates
                ):
                    selected_persons = person_selector.select_persons(candidate_frames)
                    # Mark progress as complete
                    self.state.update_step_progress(
                        self.step_name, total_person_candidates
                    )
            else:
                selected_persons = person_selector.select_persons(candidate_frames)
                self.state.update_step_progress(self.step_name, total_person_candidates)

            # Store detailed results and summary statistics
            self._store_selection_results(selected_persons, total_person_candidates)
            self._format_and_log_results(selected_persons, total_person_candidates)

        except Exception as e:
            self.logger.error(f"❌ Person selection failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise

    def _store_selection_results(
        self, selected_persons: List[PersonSelection], total_candidates: int
    ):
        """Store detailed person selection results in pipeline state."""

        # Group selections by person_id for summary statistics
        person_groups = {}
        selection_categories = {}

        for selection in selected_persons:
            person_id = selection.person_id
            category = selection.category

            if person_id not in person_groups:
                person_groups[person_id] = []
            person_groups[person_id].append(selection)

            if category not in selection_categories:
                selection_categories[category] = 0
            selection_categories[category] += 1

        # Create selection summary following FrameSelectionStep pattern
        person_selections = {
            "summary": {
                "total_candidates": total_candidates,
                "total_selected": len(selected_persons),
                "unique_persons": len(person_groups),
                "selection_criteria": {
                    "min_instances_per_person": self.config.person_selection.min_instances_per_person,
                    "max_instances_per_person": self.config.person_selection.max_instances_per_person,
                    "min_quality_threshold": self.config.person_selection.min_quality_threshold,
                    "temporal_diversity_threshold": self.config.person_selection.temporal_diversity_threshold,
                    "max_total_selections": self.config.person_selection.max_total_selections,
                },
            },
            "person_groups": {},
            "selection_categories": selection_categories,
        }

        # Process and store person group details
        for person_id, selections in person_groups.items():
            if selections:
                quality_scores = [s.selection_score for s in selections]
                person_selections["person_groups"][f"person_{person_id}"] = {
                    "selected_count": len(selections),
                    "person_id": person_id,
                    "quality_stats": {
                        "range": [min(quality_scores), max(quality_scores)],
                        "average": sum(quality_scores) / len(quality_scores),
                    },
                    "categories": list({s.category for s in selections}),
                }

        # Store detailed selection data in pipeline state
        self.state.get_step_progress(self.step_name).set_data(
            "person_selections", person_selections
        )

        # Store serializable PersonSelection data for OutputGenerationStep access
        serializable_persons = [
            {
                "frame_id": ps.frame_data.frame_id,
                "person_id": ps.person_id,
                "selection_score": ps.selection_score,
                "category": ps.category,
                "timestamp": ps.timestamp,
            }
            for ps in selected_persons
        ]
        self.state.get_step_progress(self.step_name).set_data(
            ALL_SELECTED_PERSONS_KEY, serializable_persons
        )

    def _format_and_log_results(
        self, selected_persons: List[PersonSelection], total_candidates: int
    ):
        """Format person selection results for display and logging."""

        if self.formatter:
            # Group selections by person_id for display
            person_groups = {}
            for selection in selected_persons:
                person_id = selection.person_id
                if person_id not in person_groups:
                    person_groups[person_id] = []
                person_groups[person_id].append(selection)

            # Create person breakdown display
            person_breakdown = [
                f"person_{person_id} ({len(selections)})"
                for person_id, selections in sorted(person_groups.items())
            ]

            # Create category breakdown display
            category_counts = {}
            for selection in selected_persons:
                category = selection.category
                category_counts[category] = category_counts.get(category, 0) + 1

            category_breakdown = [
                f"{category} ({count})" for category, count in category_counts.items()
            ]

            results = {
                "candidates_summary": f"📊 Candidates: {total_candidates} person instances",
                "selected_summary": f"✅ Selected {len(selected_persons)} person instances",
                "person_breakdown": (
                    f"👥 Persons: {', '.join(person_breakdown)}"
                    if person_breakdown
                    else "👥 No persons selected"
                ),
                "category_breakdown": (
                    f"📂 Categories: {', '.join(category_breakdown)}"
                    if category_breakdown
                    else "📂 No categories"
                ),
            }

            self.state.get_step_progress(self.step_name).set_data(
                "step_results", results
            )
        else:
            self.logger.info(
                f"✅ Person selection completed: {len(selected_persons)} person instances from {total_candidates} candidates"
            )
            if len(selected_persons) == 0:
                self.logger.warning(
                    "⚠️  No person instances were selected - check quality thresholds!"
                )
