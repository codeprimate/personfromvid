from .base import PipelineStep
from ...analysis.quality_assessor import create_quality_assessor
from ...utils.logging import get_logger
from collections import defaultdict
from typing import List


class QualityAssessmentStep(PipelineStep):
    """Pipeline step for assessing frame quality."""

    @property
    def step_name(self) -> str:
        return "quality_assessment"

    def execute(self) -> None:
        """Assess quality of frames with faces and poses."""
        self.state.start_step(self.step_name)

        try:
            frames_for_quality = [
                frame
                for frame in self.state.frames
                if frame.has_faces() and frame.has_poses()
            ]

            if not frames_for_quality:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames with faces and poses for quality assessment"
                    )
                else:
                    self.logger.warning(
                        "⚠️  No frames with faces/poses for quality assessment"
                    )
                self.state.get_step_progress(self.step_name).start(0)
                self.state.update_step_progress(self.step_name, 0)
                return

            total_frames = len(frames_for_quality)
            if self.formatter:
                self.formatter.print_info("🔍 Evaluating frame quality...", "analysis")
            else:
                self.logger.info(f"🔍 Assessing quality for {total_frames} frames...")

            quality_assessor = create_quality_assessor()
            self.state.get_step_progress(self.step_name).start(total_frames)

            issue_counts = defaultdict(int)
            high_quality_count = 0

            def progress_callback(processed_count: int):
                self._check_interrupted()
                self.state.update_step_progress(self.step_name, processed_count)
                if self.formatter:
                    self.formatter.update_progress(1)

            if self.formatter and hasattr(self.formatter, "step_progress_context"):
                with self.formatter.step_progress_context(
                    "Evaluating quality", total_frames
                ) as progress_updater:
                    for i, frame in enumerate(frames_for_quality):
                        # Check for interruption at regular intervals
                        if i % 10 == 0:
                            self._check_interrupted()
                            
                        try:
                            quality_assessor.assess_quality_in_frame(frame)

                            # Update stats
                            if frame.quality_metrics:
                                if frame.quality_metrics.is_high_quality:
                                    high_quality_count += 1
                                for issue in frame.quality_metrics.quality_issues:
                                    issue_counts[issue] += 1
                        finally:
                            # Unload image from memory to conserve resources
                            frame.unload_image()
                            if callable(progress_updater):
                                progress_updater(i + 1)
            else:
                for i, frame in enumerate(frames_for_quality):
                    # Check for interruption at regular intervals
                    if i % 10 == 0:
                        self._check_interrupted()
                        
                    try:
                        quality_assessor.assess_quality_in_frame(frame)

                        # Update stats
                        if frame.quality_metrics:
                            if frame.quality_metrics.is_high_quality:
                                high_quality_count += 1
                            for issue in frame.quality_metrics.quality_issues:
                                issue_counts[issue] += 1
                    finally:
                        # Unload image from memory to conserve resources
                        frame.unload_image()
                        progress_callback(i + 1)

            # PHASE 2: Rank all assessed frames by quality
            self._rank_frames_by_quality(self.state.frames)

            total_assessed = len(frames_for_quality)
            quality_stats = {
                "high_quality": high_quality_count,
                "usable": total_assessed - len(issue_counts),
                "issues": dict(issue_counts),
            }

            self.state.get_step_progress(self.step_name).set_data(
                "total_assessed", total_assessed
            )
            self.state.get_step_progress(self.step_name).set_data(
                "quality_stats", quality_stats
            )

            high = quality_stats.get("high_quality", 0)
            usable = quality_stats.get("usable", 0)
            poor = quality_stats.get("poor", 0)

            if self.formatter:
                results = {
                    "quality_assessment_summary": "✅ Quality assessment complete",
                    "high_quality_count": f"📊 High quality: {high} frames",
                    "usable_quality_count": f"📊 Usable quality: {usable} frames",
                    "poor_quality_count": f"📊 Poor quality: {poor} frames (excluded)",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Quality assessment completed: {total_assessed}/{total_frames} frames"
                )
                self.logger.info(f"   ✨ Usable quality: {usable} frames")
                self.logger.info(f"   🏆 High quality: {high} frames")
                if usable == 0:
                    self.logger.warning("⚠️  No frames meet minimum quality standards!")

        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise

    def _rank_frames_by_quality(self, frames: List["FrameData"]) -> None:
        """Rank frames globally by their overall quality score.
        
        This method assigns a global quality_rank to frames based on their
        quality_metrics.overall_quality score. Only frames that have been
        quality assessed (have quality_metrics populated) receive a rank.
        
        Args:
            frames: List of all frames in the pipeline state
        """
        # Filter to only frames that have been quality assessed
        assessed_frames = [
            frame for frame in frames 
            if frame.quality_metrics is not None
        ]
        
        if not assessed_frames:
            self.logger.debug("No frames were quality assessed - no ranking performed")
            return
        
        # Sort frames by overall quality in descending order (higher score = better)
        # Using stable sort to handle ties gracefully
        assessed_frames.sort(
            key=lambda frame: frame.quality_metrics.overall_quality,
            reverse=True
        )
        
        # Assign ranks (1 = highest quality)
        for rank, frame in enumerate(assessed_frames, start=1):
            frame.selections.quality_rank = rank
        
        self.logger.debug(
            f"Assigned quality ranks to {len(assessed_frames)} frames "
            f"(range: 1-{len(assessed_frames)})"
        )
