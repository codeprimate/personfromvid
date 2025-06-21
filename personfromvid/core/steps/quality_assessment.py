import time
from collections import defaultdict
from typing import TYPE_CHECKING, List

from ...analysis.quality_assessor import create_quality_assessor
from ...data.constants import QualityMethod
from ...data.detection_results import QualityMetrics
from .base import PipelineStep

if TYPE_CHECKING:
    from ...data.frame_data import FrameData


class QualityAssessmentStep(PipelineStep):
    """Pipeline step for assessing frame quality."""

    @property
    def step_name(self) -> str:
        return "quality_assessment"

    def execute(self) -> None:
        """Assess quality of frames with faces and poses using inferred quality approach."""
        self.state.start_step(self.step_name)

        try:
            frames_for_quality = [
                frame
                for frame in self.state.frames
                if frame.has_faces()
                or frame.has_poses()  # either faces or poses are required for quality assessment
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
                self.formatter.print_info(
                    "🔍 Evaluating frame and person quality...", "analysis"
                )
            else:
                self.logger.info(f"🔍 Assessing quality for {total_frames} frames...")

            quality_assessor = create_quality_assessor()
            self.state.get_step_progress(self.step_name).start(total_frames)

            step_start_time = time.time()
            issue_counts = defaultdict(int)
            high_quality_count = 0
            inferred_count = 0
            direct_count = 0

            def progress_callback(processed_count: int):
                self._check_interrupted()
                self.state.update_step_progress(self.step_name, processed_count)
                if self.formatter:
                    # Calculate rate
                    elapsed = time.time() - step_start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    self.formatter.update_progress(1, rate=rate)

            # PHASE 1: Assess person-level quality for frames with persons
            self._assess_person_quality(frames_for_quality, quality_assessor)

            # PHASE 2: Infer frame quality from persons or use direct analysis
            if self.formatter and hasattr(self.formatter, "step_progress_context"):
                with self.formatter.step_progress_context(
                    "Inferring quality", total_frames
                ) as progress_updater:
                    for i, frame in enumerate(frames_for_quality):
                        # Check for interruption at regular intervals
                        if i % 10 == 0:
                            self._check_interrupted()

                        try:
                            # Determine quality assessment method
                            persons = frame.get_persons()

                            if persons:
                                # INFERRED: Use person quality averaging
                                frame.quality_metrics = (
                                    self._infer_frame_quality_from_persons(frame)
                                )
                                inferred_count += 1
                            else:
                                # DIRECT: Fall back to full frame analysis
                                quality_assessor.assess_quality_in_frame(frame)
                                if frame.quality_metrics:
                                    frame.quality_metrics.method = QualityMethod.DIRECT
                                direct_count += 1

                            # Update stats
                            if frame.quality_metrics:
                                if frame.quality_metrics.is_high_quality:
                                    high_quality_count += 1
                                for issue in frame.quality_metrics.quality_issues:
                                    issue_counts[issue] += 1
                        finally:
                            # Unload image from memory to conserve resources
                            frame.unload_image()
                            # Update both formatter context and pipeline state progress
                            if callable(progress_updater):
                                progress_updater(i + 1)
                            progress_callback(i + 1)
            else:
                for i, frame in enumerate(frames_for_quality):
                    # Check for interruption at regular intervals
                    if i % 10 == 0:
                        self._check_interrupted()

                    try:
                        # Determine quality assessment method
                        persons = frame.get_persons()

                        if persons:
                            # INFERRED: Use person quality averaging
                            frame.quality_metrics = (
                                self._infer_frame_quality_from_persons(frame)
                            )
                            inferred_count += 1
                        else:
                            # DIRECT: Fall back to full frame analysis
                            quality_assessor.assess_quality_in_frame(frame)
                            if frame.quality_metrics:
                                frame.quality_metrics.method = QualityMethod.DIRECT
                            direct_count += 1

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

            # PHASE 3: Rank all assessed frames by quality
            self._rank_frames_by_quality(self.state.frames)

            total_assessed = len(frames_for_quality)
            total_persons = sum(
                len(frame.get_persons()) for frame in frames_for_quality
            )
            high_quality_persons = sum(
                1
                for frame in frames_for_quality
                for person in frame.get_persons()
                if person.quality.overall_quality >= 0.7
            )

            quality_stats = {
                "high_quality": high_quality_count,
                "usable": total_assessed - len(issue_counts),
                "issues": dict(issue_counts),
                "total_persons": total_persons,
                "high_quality_persons": high_quality_persons,
                "inferred_frames": inferred_count,
                "direct_frames": direct_count,
                "performance_improvement": f"{(inferred_count / total_assessed * 100):.1f}%"
                if total_assessed > 0
                else "0%",
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
            total_persons = quality_stats.get("total_persons", 0)
            high_quality_persons = quality_stats.get("high_quality_persons", 0)
            inferred = quality_stats.get("inferred_frames", 0)
            direct = quality_stats.get("direct_frames", 0)
            performance_improvement = quality_stats.get("performance_improvement", "0%")

            if self.formatter:
                results = {
                    "quality_assessment_summary": "✅ Quality assessment complete",
                    "high_quality_count": f"📊 High quality: {high} frames",
                    "usable_quality_count": f"📊 Usable quality: {usable} frames",
                    "poor_quality_count": f"📊 Poor quality: {poor} frames (excluded)",
                    "person_quality_summary": f"🧑 Persons assessed: {total_persons}",
                    "high_quality_persons": f"🏆 High quality persons: {high_quality_persons}",
                    "method_breakdown": f"🔍 Quality methods: {inferred} inferred, {direct} direct",
                    "performance_optimization": f"⚡ Performance improvement: {performance_improvement} frames optimized",
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
                self.logger.info(f"   🧑 Total persons: {total_persons}")
                self.logger.info(f"   🏆 High quality persons: {high_quality_persons}")
                self.logger.info(
                    f"   🔍 Quality methods: {inferred} inferred, {direct} direct"
                )
                self.logger.info(
                    f"   ⚡ Performance improvement: {performance_improvement} frames optimized"
                )
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
            frame for frame in frames if frame.quality_metrics is not None
        ]

        if not assessed_frames:
            self.logger.debug("No frames were quality assessed - no ranking performed")
            return

        # Sort frames by overall quality in descending order (higher score = better)
        # Using stable sort to handle ties gracefully
        assessed_frames.sort(
            key=lambda frame: frame.quality_metrics.overall_quality, reverse=True
        )

        # Assign ranks (1 = highest quality)
        for rank, frame in enumerate(assessed_frames, start=1):
            frame.selections.quality_rank = rank

        self.logger.debug(
            f"Assigned quality ranks to {len(assessed_frames)} frames "
            f"(range: 1-{len(assessed_frames)})"
        )

    def _assess_person_quality(
        self, frames: List["FrameData"], quality_assessor
    ) -> None:
        """Assess quality for Person objects within frames.

        Updates Person.quality in-place using sophisticated image quality metrics
        instead of simple confidence scores from PersonBuilder.

        Args:
            frames: List of frames that have been quality assessed
            quality_assessor: QualityAssessor instance for bbox assessment
        """
        total_persons = sum(len(frame.get_persons()) for frame in frames)

        if total_persons == 0:
            self.logger.debug("No persons found for person-level quality assessment")
            return

        self.logger.info(
            f"🧑 Assessing quality for {total_persons} persons across {len(frames)} frames..."
        )

        persons_processed = 0
        high_quality_persons = 0

        for frame in frames:
            # Image is automatically loaded by FrameData.image property
            if frame.image is None:
                self.logger.warning(
                    f"⚠️  Skipping frame {frame.frame_number}: unable to load image"
                )
                continue

            persons = frame.get_persons()
            if not persons:
                continue

            for person in persons:
                try:
                    # Assess face quality if face is present
                    face_quality = 0.0
                    if not self._is_sentinel_face(person.face):
                        face_bbox = person.face.bbox
                        face_quality, _ = quality_assessor.assess_quality_of_bbox(
                            frame.image, face_bbox
                        )

                    # Assess body quality if body is present
                    body_quality = 0.0
                    if not self._is_sentinel_body(person.body):
                        body_bbox = person.body.bbox
                        body_quality, _ = quality_assessor.assess_quality_of_bbox(
                            frame.image, body_bbox
                        )

                    # Update Person quality with assessed values
                    from ...data.person import PersonQuality

                    person.quality = PersonQuality(
                        face_quality=face_quality, body_quality=body_quality
                    )

                    if person.quality.overall_quality >= 0.7:  # High quality threshold
                        high_quality_persons += 1

                    persons_processed += 1

                    self.logger.debug(
                        f"Person {person.person_id} quality updated: "
                        f"face={face_quality:.3f}, body={body_quality:.3f}, "
                        f"combined={person.quality.overall_quality:.3f}"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Person quality assessment failed for person {person.person_id}: {e}"
                    )
                    # Keep existing quality as fallback
                    continue

            # Note: Frame images will be unloaded in the main processing loop for memory conservation

        self.logger.info(
            f"✅ Person quality assessment completed: {persons_processed}/{total_persons} persons processed"
        )
        self.logger.info(f"   🏆 High quality persons: {high_quality_persons}")

    def _is_sentinel_face(self, face) -> bool:
        """Check if face is a FaceUnknown sentinel object."""
        from ...data.person import FaceUnknown

        return isinstance(face, FaceUnknown)

    def _is_sentinel_body(self, body) -> bool:
        """Check if body is a BodyUnknown sentinel object."""
        from ...data.person import BodyUnknown

        return isinstance(body, BodyUnknown)

    def _infer_frame_quality_from_persons(self, frame: "FrameData") -> QualityMetrics:
        """Infer frame quality from person quality assessments.

        Creates QualityMetrics with method=INFERRED by averaging person quality scores
        and generating synthetic component scores based on the overall quality.

        Args:
            frame: FrameData with assessed persons

        Returns:
            QualityMetrics with inferred quality scores
        """
        persons = frame.get_persons()
        if not persons:
            raise ValueError("Cannot infer quality from frame without persons")

        # Average all person quality scores with equal weighting
        person_qualities = [person.quality.overall_quality for person in persons]
        overall_quality = sum(person_qualities) / len(person_qualities)

        # Generate synthetic component scores based on overall quality
        # Use reasonable mappings from overall quality to component scores
        laplacian_variance = overall_quality * 1000.0  # Scale to typical range
        sobel_variance = overall_quality * 800.0
        brightness_score = 50.0 + (overall_quality * 50.0)  # 50-100 range
        contrast_score = overall_quality * 0.8  # Slightly lower than overall

        # Determine quality issues based on thresholds
        quality_issues = []
        if overall_quality < 0.3:
            quality_issues.append("low_person_quality")
        if overall_quality < 0.5:
            quality_issues.append("below_average_quality")

        # Frame is usable if any person meets minimum standards
        usable = overall_quality >= 0.3

        self.logger.debug(
            f"🔍 Inferred frame quality from {len(persons)} person(s): "
            f"overall={overall_quality:.3f}, persons=[{', '.join(f'{q:.3f}' for q in person_qualities)}]"
        )

        return QualityMetrics(
            laplacian_variance=laplacian_variance,
            sobel_variance=sobel_variance,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            overall_quality=overall_quality,
            method=QualityMethod.INFERRED,
            quality_issues=quality_issues,
            usable=usable,
        )
