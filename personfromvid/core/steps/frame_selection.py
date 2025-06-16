from .base import PipelineStep
from ...analysis.frame_selector import create_frame_selector, SelectionCriteria
from ...data.constants import ALL_SELECTED_FRAMES_KEY


class FrameSelectionStep(PipelineStep):
    """Pipeline step for selecting the best frames."""

    @property
    def step_name(self) -> str:
        return "frame_selection"

    def _calculate_dynamic_max_frames(
        self, baseline_frames: int, video_duration: float
    ) -> int:
        """Calculate dynamic max frames per category based on video duration.

        Uses baseline minimum frames for 30s video, increments by 1 for every 10 additional seconds after 30s.
        Result is capped by max_frames_per_category configuration.

        Args:
            baseline_frames: Base minimum number of frames for 30s video
            video_duration: Video duration in seconds

        Returns:
            Dynamic max frames per category (baseline + duration scaling, capped by config)
        """
        additional_frames = max(0, int((video_duration - 30.0) // 10))
        calculated_frames = baseline_frames + additional_frames
        
        # Respect the max_frames_per_category configuration limit
        max_allowed = self.config.output.max_frames_per_category
        return min(calculated_frames, max_allowed)

    def execute(self) -> None:
        """Select best frames based on quality and diversity."""
        self.state.start_step(self.step_name)

        try:
            candidate_frames = [
                f
                for f in self.state.frames
                if f.has_faces() and f.has_poses() and f.quality_metrics is not None
            ]

            if not candidate_frames:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames with quality assessments for selection"
                    )
                else:
                    self.logger.warning(
                        "⚠️ No frames with quality assessments for selection"
                    )
                self.state.get_step_progress(self.step_name).start(0)
                return

            total_candidates = len(candidate_frames)
            if self.formatter:
                self.formatter.print_info(
                    "🎯 Optimizing frame selection...", "targeting"
                )
            else:
                self.logger.info(f"🎯 Selecting from {total_candidates} candidates...")

            # Calculate dynamic frames per category based on video duration
            baseline_frames = self.config.output.min_frames_per_category
            video_duration = self.state.video_metadata.duration
            dynamic_max_frames = self._calculate_dynamic_max_frames(
                baseline_frames, video_duration
            )

            criteria = SelectionCriteria(
                min_frames_per_category=self.config.output.min_frames_per_category,
                max_frames_per_category=dynamic_max_frames,
                min_quality_threshold=self.config.frame_selection.min_quality_threshold,
                face_size_weight=self.config.frame_selection.face_size_weight,
                quality_weight=self.config.frame_selection.quality_weight,
                diversity_threshold=self.config.frame_selection.diversity_threshold,
            )
            frame_selector = create_frame_selector(criteria)
            self.state.get_step_progress(self.step_name).start(total_candidates)

            current_progress = 0

            def progress_callback(message: str):
                nonlocal current_progress
                self._check_interrupted()
                # Simplified progress update
                current_progress += 1
                self.state.update_step_progress(self.step_name, current_progress)
                if self.formatter:
                    self.formatter.update_progress(1)

            if self.formatter:
                with self.formatter.create_progress_bar(
                    "Selecting frames", total_candidates
                ):
                    selection_summary = frame_selector.select_best_frames(
                        candidate_frames, progress_callback,
                        interruption_check=self._check_interrupted
                    )
            else:
                selection_summary = frame_selector.select_best_frames(
                    candidate_frames, lambda m: self.logger.debug(f"   {m}"),
                    interruption_check=self._check_interrupted
                )

            self.state.update_step_progress(
                self.step_name, total_candidates
            )  # Mark as complete

            # Store detailed summary and high-level selections
            self._store_selection_results(selection_summary, criteria)
            self._format_and_log_results(selection_summary)

        except Exception as e:
            self.logger.error(f"❌ Frame selection failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise

    def _store_selection_results(self, selection_summary, criteria):
        """Stores detailed and high-level selection results in the pipeline state."""
        frame_selections = {
            "summary": {
                "total_candidates": selection_summary.total_candidates,
                "total_selected": selection_summary.total_selected,
                "selection_criteria": criteria.__dict__,
            },
            "pose_categories": {},
            "head_angle_categories": {},
        }

        # Process and store pose selections
        for category, selection in selection_summary.pose_selections.items():
            if selection.selected_frames:
                frame_selections["pose_categories"][category] = {
                    "selected_count": len(selection.selected_frames),
                    "total_candidates": selection.total_candidates,
                    "rationale": selection.selection_rationale,
                    "quality_stats": {
                        "range": selection.quality_range,
                        "average": selection.average_quality,
                    },
                }

        # Process and store head angle selections
        for category, selection in selection_summary.head_angle_selections.items():
            if selection.selected_frames:
                frame_selections["head_angle_categories"][category] = {
                    "selected_count": len(selection.selected_frames),
                    "total_candidates": selection.total_candidates,
                    "rationale": selection.selection_rationale,
                    "quality_stats": {
                        "range": selection.quality_range,
                        "average": selection.average_quality,
                    },
                }

        self.state.get_step_progress(self.step_name).set_data(
            "frame_selections", frame_selections
        )

        # Store a flat list of all selected frames for the next step
        all_selected_frames = []
        for _, selection in selection_summary.pose_selections.items():
            all_selected_frames.extend(selection.selected_frames)
        for _, selection in selection_summary.head_angle_selections.items():
            all_selected_frames.extend(selection.selected_frames)

        # Remove duplicates
        unique_frames_map = {frame.frame_id: frame for frame in all_selected_frames}
        unique_frame_ids = list(unique_frames_map.keys())
        self.state.get_step_progress(self.step_name).set_data(
            ALL_SELECTED_FRAMES_KEY, unique_frame_ids
        )

    def _format_and_log_results(self, selection_summary):
        """Formats the results for display and logs them."""
        if self.formatter:
            body_poses = [
                f"{cat} ({len(sel.selected_frames)})"
                for cat, sel in selection_summary.pose_selections.items()
                if sel.selected_frames
            ]
            head_angles = [
                f"{cat} ({len(sel.selected_frames)})"
                for cat, sel in selection_summary.head_angle_selections.items()
                if sel.selected_frames
            ]

            results = {
                "candidates_summary": f"📊 Candidates: {selection_summary.total_candidates} frames",
                "selected_summary": f"✅ Selected {selection_summary.total_selected} frames",
                "body_poses_breakdown": (
                    f"🏃 Body poses: {', '.join(body_poses)}"
                    if body_poses
                    else "🏃 No body poses selected"
                ),
                "head_angles_breakdown": (
                    f"👤 Head angles: {', '.join(head_angles)}"
                    if head_angles
                    else "👤 No head angles selected"
                ),
            }
            self.state.get_step_progress(self.step_name).set_data(
                "step_results", results
            )
        else:
            self.logger.info(
                f"✅ Frame selection completed: {selection_summary.total_selected} frames from {selection_summary.total_candidates} candidates"
            )
            if selection_summary.total_selected == 0:
                self.logger.warning(
                    "⚠️  No frames were selected - check quality thresholds!"
                )
