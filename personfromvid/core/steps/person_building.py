"""Person building pipeline step for associating face and body detections.

This module provides the PersonBuildingStep class that integrates the PersonBuilder
into the pipeline to create Person objects from face and pose detections within each frame.
"""

import time
from typing import TYPE_CHECKING, List

from ...analysis.person_builder import PersonBuilder
from .base import PipelineStep

if TYPE_CHECKING:
    from ...data.frame_data import FrameData


class PersonBuildingStep(PipelineStep):
    """Pipeline step for building Person objects from detections."""

    @property
    def step_name(self) -> str:
        return "person_building"

    def execute(self) -> None:
        """Build Person objects from face and pose detections in each frame."""
        self.state.start_step(self.step_name)

        try:
            # Get frames that have face and/or pose detections
            frames_with_detections = self._get_frames_with_detections()

            if not frames_with_detections:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames with face or pose detections for person building"
                    )
                else:
                    self.logger.warning(
                        "⚠️  No frames with detections for person building"
                    )
                self.state.get_step_progress(self.step_name).start(0)
                self.state.update_step_progress(self.step_name, 0)
                return

            total_frames = len(frames_with_detections)
            if self.formatter:
                self.formatter.print_info("🔧 Building person objects...", "analysis")
            else:
                self.logger.info(f"🔧 Building persons for {total_frames} frames...")

            # Initialize PersonBuilder
            person_builder = PersonBuilder()
            self.state.get_step_progress(self.step_name).start(total_frames)

            # Statistics tracking
            total_persons_found = 0
            single_person_frames = 0
            multi_person_frames = 0
            step_start_time = time.time()

            def progress_callback(processed_count: int, rate: float = None):
                self._check_interrupted()
                self.state.update_step_progress(self.step_name, processed_count)
                if self.formatter:
                    if rate is not None:
                        self.formatter.update_progress(1, rate=rate)
                    else:
                        self.formatter.update_progress(1)

            # Process frames with progress tracking
            if self.formatter and hasattr(self.formatter, "step_progress_context"):
                with self.formatter.step_progress_context(
                    "Building persons", total_frames
                ) as progress_updater:
                    for i, frame in enumerate(frames_with_detections):
                        # Check for interruption at regular intervals
                        if i % 10 == 0:
                            self._check_interrupted()

                        try:
                            # Build Person objects for this frame
                            persons = person_builder.build_persons(
                                frame.face_detections,
                                frame.pose_detections,
                                frame.head_poses,
                            )

                            # Update frame with Person objects
                            frame.persons = persons

                            # Update statistics
                            person_count = len(persons)
                            total_persons_found += person_count

                            if person_count == 1:
                                single_person_frames += 1
                            elif person_count > 1:
                                multi_person_frames += 1

                            # TODO: Track spatial vs fallback associations in future enhancement

                        except Exception as e:
                            self.logger.error(
                                f"Failed to build persons for frame {frame.frame_number}: {e}"
                            )
                            # Set empty persons list on failure
                            frame.persons = []

                        finally:
                            # Update both formatter context and pipeline state progress
                            if callable(progress_updater):
                                progress_updater(i + 1)

                            # Calculate processing rate
                            elapsed = time.time() - step_start_time
                            rate = (i + 1) / elapsed if elapsed > 0 else 0
                            progress_callback(i + 1, rate)
            else:
                # Basic progress tracking without rich formatter
                for i, frame in enumerate(frames_with_detections):
                    # Check for interruption at regular intervals
                    if i % 10 == 0:
                        self._check_interrupted()

                    try:
                        # Build Person objects for this frame
                        persons = person_builder.build_persons(
                            frame.face_detections,
                            frame.pose_detections,
                            frame.head_poses,
                        )

                        # Update frame with Person objects
                        frame.persons = persons

                        # Update statistics
                        person_count = len(persons)
                        total_persons_found += person_count

                        if person_count == 1:
                            single_person_frames += 1
                        elif person_count > 1:
                            multi_person_frames += 1

                    except Exception as e:
                        self.logger.error(
                            f"Failed to build persons for frame {frame.frame_number}: {e}"
                        )
                        # Set empty persons list on failure
                        frame.persons = []

                    finally:
                        # Calculate processing rate
                        elapsed = time.time() - step_start_time
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        progress_callback(i + 1, rate)

            # Calculate statistics
            average_persons_per_frame = (
                total_persons_found / total_frames if total_frames > 0 else 0
            )

            # Store step data for display
            step_data = {
                "total_frames_processed": total_frames,
                "total_persons_found": total_persons_found,
                "single_person_frames": single_person_frames,
                "multi_person_frames": multi_person_frames,
                "average_persons_per_frame": average_persons_per_frame,
            }

            for key, value in step_data.items():
                self.state.get_step_progress(self.step_name).set_data(key, value)

            # Display results
            if self.formatter:
                results = {
                    "person_building_summary": "✅ Person building complete",
                    "total_persons": f"👥 Total persons: {total_persons_found} across {total_frames} frames",
                    "single_person_frames": f"👤 Single-person frames: {single_person_frames}",
                    "multi_person_frames": f"👥 Multi-person frames: {multi_person_frames}",
                    "average_persons": f"📊 Average persons per frame: {average_persons_per_frame:.1f}",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Person building completed: {total_persons_found} persons found"
                )
                self.logger.info(f"   👤 Single-person frames: {single_person_frames}")
                self.logger.info(f"   👥 Multi-person frames: {multi_person_frames}")
                self.logger.info(
                    f"   📊 Average: {average_persons_per_frame:.1f} persons per frame"
                )

        except Exception as e:
            self.logger.error(f"❌ Person building failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise

    def _get_frames_with_detections(self) -> List["FrameData"]:
        """Get frames that have face and/or pose detections for person building.

        Returns:
            List of frames with at least one face or pose detection
        """
        frames_with_detections = []

        for frame in self.state.frames:
            has_faces = frame.has_faces()
            has_poses = frame.has_poses()

            # Include frame if it has at least one type of detection
            if has_faces or has_poses:
                frames_with_detections.append(frame)

        return frames_with_detections
