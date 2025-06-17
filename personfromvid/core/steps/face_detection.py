from .base import PipelineStep
from ...models.face_detector import create_face_detector
from ...models.head_pose_estimator import create_head_pose_estimator


class FaceDetectionStep(PipelineStep):
    """Pipeline step for detecting faces in frames."""

    @property
    def step_name(self) -> str:
        return "face_detection"

    def execute(self) -> None:
        """Detect faces in all extracted frames and filter for forward-facing only."""
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

            # Filter out non-forward-facing detections
            self._filter_forward_facing_detections()

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
                    "detection_summary": f"Found {total_faces_found} forward-facing faces across {frames_with_faces} frames ({coverage:.1f}% coverage)",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Face detection completed: {total_faces_found} forward-facing faces found"
                )
                self.logger.info(
                    f"   📊 Frames with faces: {frames_with_faces}/{total_frames}"
                )

        except Exception as e:
            self.logger.error(f"❌ Face detection failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise

    def _filter_forward_facing_detections(self) -> None:
        """Filter face detections to keep only forward-facing faces."""
        import cv2
        import time
        
        if self.formatter:
            self.formatter.print_info("🎯 Filtering for forward-facing faces...", "face")
        else:
            self.logger.info("🎯 Filtering face detections for forward-facing poses...")

        # Create head pose estimator for filtering
        head_pose_estimator = create_head_pose_estimator(
            model_name=self.config.models.head_pose_model,
            device=(
                self.config.models.device.value
                if hasattr(self.config.models.device, "value")
                else str(self.config.models.device)
            ),
            confidence_threshold=self.config.models.confidence_threshold,
        )

        frames_with_faces = [f for f in self.state.frames if f.has_faces()]
        total_original_faces = sum(len(f.face_detections) for f in frames_with_faces)
        total_filtered_faces = 0
        processed_frames = 0
        start_time = time.time()

        def filtering_progress_callback():
            nonlocal processed_frames
            processed_frames += 1
            if self.formatter:
                # Calculate current processing rate
                elapsed = time.time() - start_time
                current_rate = processed_frames / elapsed if elapsed > 0 else 0
                # Pass rate information to formatter
                self.formatter.update_progress(1, rate=current_rate)

        if self.formatter:
            progress_bar = self.formatter.create_progress_bar(
                "Filtering faces", len(frames_with_faces)
            )
            with progress_bar:
                for frame in frames_with_faces:
                    self._check_interrupted()
                    
                    if not frame.face_detections:
                        filtering_progress_callback()
                        continue

                    # Load frame image
                    image = frame.image
                    if image is None:
                        self.logger.warning(f"Could not load image for frame {frame.frame_id}, keeping all detections")
                        filtering_progress_callback()
                        continue

                    filtered_detections = []
                    
                    for face_detection in frame.face_detections:
                        try:
                            # Crop face from image
                            x1, y1, x2, y2 = face_detection.bbox
                            # Add small padding to ensure we get the full face
                            padding = 10
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(image.shape[1], x2 + padding)
                            y2 = min(image.shape[0], y2 + padding)
                            
                            face_crop = image[y1:y2, x1:x2]
                            
                            # Skip if crop is too small
                            if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                                continue
                            
                            # Estimate head pose
                            head_pose_result = head_pose_estimator.estimate_head_pose(face_crop)
                            
                            # Check if face is forward-facing
                            if head_pose_estimator.is_facing_forward(
                                head_pose_result.yaw, 
                                head_pose_result.pitch, 
                                head_pose_result.roll
                            ):
                                filtered_detections.append(face_detection)
                                
                        except Exception as e:
                            self.logger.debug(f"Error filtering face in frame {frame.frame_id}: {e}")
                            # In case of error, keep the detection to be safe
                            filtered_detections.append(face_detection)

                    # Replace face detections with filtered ones
                    frame.face_detections = filtered_detections
                    total_filtered_faces += len(filtered_detections)
                    
                    filtering_progress_callback()
        else:
            # Non-formatted version without progress bar
            for frame in frames_with_faces:
                self._check_interrupted()
                
                if not frame.face_detections:
                    continue

                # Load frame image
                image = frame.image
                if image is None:
                    self.logger.warning(f"Could not load image for frame {frame.frame_id}, keeping all detections")
                    continue

                filtered_detections = []
                
                for face_detection in frame.face_detections:
                    try:
                        # Crop face from image
                        x1, y1, x2, y2 = face_detection.bbox
                        # Add small padding to ensure we get the full face
                        padding = 10
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(image.shape[1], x2 + padding)
                        y2 = min(image.shape[0], y2 + padding)
                        
                        face_crop = image[y1:y2, x1:x2]
                        
                        # Skip if crop is too small
                        if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                            continue
                        
                        # Estimate head pose
                        head_pose_result = head_pose_estimator.estimate_head_pose(face_crop)
                        
                        # Check if face is forward-facing
                        if head_pose_estimator.is_facing_forward(
                            head_pose_result.yaw, 
                            head_pose_result.pitch, 
                            head_pose_result.roll
                        ):
                            filtered_detections.append(face_detection)
                            
                    except Exception as e:
                        self.logger.debug(f"Error filtering face in frame {frame.frame_id}: {e}")
                        # In case of error, keep the detection to be safe
                        filtered_detections.append(face_detection)

                # Replace face detections with filtered ones
                frame.face_detections = filtered_detections
                total_filtered_faces += len(filtered_detections)

        faces_removed = total_original_faces - total_filtered_faces
        
        if self.formatter:
            self.logger.info(f"   🎯 Filtering complete: kept {total_filtered_faces}/{total_original_faces} faces (removed {faces_removed} non-forward-facing)")
        else:
            self.logger.info(f"🎯 Forward-facing filter: kept {total_filtered_faces}/{total_original_faces} faces (removed {faces_removed})")
