from .base import PipelineStep
from ...models.face_detector import create_face_detector


class FaceDetectionStep(PipelineStep):
    """Pipeline step for detecting faces in frames."""

    @property
    def step_name(self) -> str:
        return "face_detection"

    def execute(self) -> None:
        """Detect faces in all extracted frames."""
        self.state.start_step(self.step_name)

        try:
            if self.formatter:
                self.formatter.print_info("👤 Running YOLOv8 face detection...", "face")
            else:
                self.logger.info("👤 Starting face detection...")

            if not self.state.frames:
                self.logger.warning("⚠️  No frames found from extraction step")
                self.state.get_step_progress(self.step_name).start(0)
                self.state.update_step_progress(self.step_name, 0)
                return

            face_detector = create_face_detector(
                model_name=self.config.models.face_detection_model,
                device=(
                    self.config.models.device.value
                    if hasattr(self.config.models.device, "value")
                    else str(self.config.models.device)
                ),
                confidence_threshold=self.config.models.confidence_threshold,
            )

            total_frames = len(self.state.frames)
            self.state.get_step_progress(self.step_name).start(total_frames)

            last_processed_count = 0

            def progress_callback(processed_count: int, rate: float = None):
                nonlocal last_processed_count
                self._check_interrupted()
                self.state.update_step_progress(self.step_name, processed_count)
                advance_amount = processed_count - last_processed_count
                last_processed_count = processed_count
                if self.formatter:
                    # Pass rate information to formatter if available
                    if rate is not None:
                        self.formatter.update_progress(advance_amount, rate=rate)
                    else:
                        self.formatter.update_progress(advance_amount)

            if self.formatter:
                progress_bar = self.formatter.create_progress_bar(
                    "Processing frames", total_frames
                )
                with progress_bar:
                    face_detector.process_frame_batch(
                        self.state.frames, self.state.video_metadata, progress_callback,
                        interruption_check=self._check_interrupted
                    )
            else:
                face_detector.process_frame_batch(
                    self.state.frames, self.state.video_metadata, progress_callback,
                    interruption_check=self._check_interrupted
                )

            total_faces_found = sum(len(f.face_detections) for f in self.state.frames)
            frames_with_faces = len([f for f in self.state.frames if f.has_faces()])
            coverage = (
                (frames_with_faces / total_frames * 100) if total_frames > 0 else 0
            )

            self.state.get_step_progress(self.step_name).set_data(
                "faces_found", total_faces_found
            )

            if self.formatter:
                results = {
                    "faces_found": total_faces_found,
                    "frames_with_faces": frames_with_faces,
                    "detection_summary": f"Found {total_faces_found} faces across {frames_with_faces} frames ({coverage:.1f}% coverage)",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Face detection completed: {total_faces_found} faces found"
                )
                self.logger.info(
                    f"   📊 Frames with faces: {frames_with_faces}/{total_frames}"
                )

        except Exception as e:
            self.logger.error(f"❌ Face detection failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise
