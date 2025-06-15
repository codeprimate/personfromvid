from .base import PipelineStep
from ...analysis.closeup_detector import CloseupDetector


class CloseupDetectionStep(PipelineStep):
    """Pipeline step for detecting closeups and shot types."""

    @property
    def step_name(self) -> str:
        return "closeup_detection"

    def execute(self) -> None:
        """Perform closeup detection and shot type classification."""
        self.state.start_step(self.step_name)

        try:
            frames_with_faces = self.state.get_frames_with_faces()
            if not frames_with_faces:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames with faces found for closeup detection"
                    )
                else:
                    self.logger.warning(
                        "⚠️  No frames with faces found for closeup detection"
                    )
                self.state.get_step_progress(self.step_name).start(0)
                self.state.update_step_progress(self.step_name, 0)
                return

            total_frames = len(frames_with_faces)
            if self.formatter:
                self.formatter.print_info(
                    "🎯 Analyzing shot types...", "targeting"
                )
            else:
                self.logger.info(
                    f"🎯 Starting closeup detection on {total_frames} frames..."
                )

            closeup_detector = CloseupDetector()
            self.state.get_step_progress(self.step_name).start(total_frames)

            last_processed_count = 0

            def progress_callback(processed_count: int, rate: float = None):
                nonlocal last_processed_count
                self._check_interrupted()
                advance = processed_count - last_processed_count
                self.state.update_step_progress(
                    self.step_name,
                    self.state.get_step_progress(self.step_name).processed_count
                    + advance,
                )
                last_processed_count = processed_count
                if self.formatter:
                    # Pass rate information to formatter if available
                    if rate is not None:
                        self.formatter.update_progress(advance, rate=rate)
                    else:
                        self.formatter.update_progress(advance)

            # Process closeups
            if self.formatter:
                with self.formatter.create_progress_bar(
                    "Detecting closeups", total_frames
                ):
                    closeup_detector.process_frame_batch(
                        frames_with_faces, progress_callback,
                        interruption_check=self._check_interrupted
                    )
            else:
                closeup_detector.process_frame_batch(
                    frames_with_faces, progress_callback,
                    interruption_check=self._check_interrupted
                )

            # Collect and store stats
            closeup_counts = {}
            for frame in frames_with_faces:
                for detection in frame.closeup_detections:
                    closeup_counts[detection.shot_type] = (
                        closeup_counts.get(detection.shot_type, 0) + 1
                    )

            total_closeups = sum(closeup_counts.values())
            self.state.get_step_progress(self.step_name).set_data(
                "shot_types_found", closeup_counts
            )
            self.state.get_step_progress(self.step_name).set_data(
                "total_closeups", total_closeups
            )

            if self.formatter:
                sorted_shots = sorted(
                    closeup_counts.items(), key=lambda x: x[1], reverse=True
                )[:4]
                shot_types_str = ", ".join([f"{st} ({c})" for st, c in sorted_shots])
                results = {
                    "total_closeups": total_closeups,
                    "shot_analysis_summary": f"✅ Shot analysis: {total_closeups} classifications",
                    "shot_types_breakdown": f"📊 Types: {shot_types_str}",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Closeup detection completed: {total_closeups} detections"
                )
                if closeup_counts:
                    for shot_type, count in sorted(
                        closeup_counts.items(), key=lambda x: x[1], reverse=True
                    )[:3]:
                        self.logger.info(f"      • {shot_type}: {count} instances")

        except Exception as e:
            self.logger.error(f"❌ Closeup detection failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise
