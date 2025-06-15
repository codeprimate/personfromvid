from .base import PipelineStep
from ...models.pose_estimator import PoseEstimator
from ...models.head_pose_estimator import HeadPoseEstimator


class PoseAnalysisStep(PipelineStep):
    """Pipeline step for analyzing body and head poses."""

    @property
    def step_name(self) -> str:
        return "pose_analysis"

    def execute(self) -> None:
        """Analyze poses and head angles for faces."""
        self.state.start_step(self.step_name)

        try:
            frames_with_faces = self.state.get_frames_with_faces()
            if not frames_with_faces:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames with faces found, skipping pose analysis"
                    )
                else:
                    self.logger.info(
                        "⚠️  No frames with faces found, skipping pose analysis"
                    )
                self.state.get_step_progress(self.step_name).start(0)
                self.state.update_step_progress(self.step_name, 0)
                return

            total_frames = len(frames_with_faces)

            pose_estimator = PoseEstimator(
                model_name=self.config.models.pose_estimation_model,
                device=self.config.models.device,
                confidence_threshold=self.config.models.confidence_threshold,
            )
            head_pose_estimator = HeadPoseEstimator(
                model_name=self.config.models.head_pose_model,
                device=self.config.models.device,
                confidence_threshold=self.config.models.confidence_threshold,
            )

            self.state.get_step_progress(self.step_name).total_items = total_frames * 2

            body_last_processed, head_last_processed = 0, 0

            def body_progress_callback(processed_count: int, rate: float = None):
                nonlocal body_last_processed
                self._check_interrupted()
                advance = processed_count - body_last_processed
                self.state.update_step_progress(
                    self.step_name,
                    self.state.get_step_progress(self.step_name).processed_count
                    + advance,
                )
                body_last_processed = processed_count
                if self.formatter:
                    # Pass rate information to formatter if available
                    if rate is not None:
                        self.formatter.update_progress(advance, rate=rate)
                    else:
                        self.formatter.update_progress(advance)

            def head_progress_callback(processed_count: int, rate: float = None):
                nonlocal head_last_processed
                self._check_interrupted()
                advance = processed_count - head_last_processed
                self.state.update_step_progress(
                    self.step_name,
                    self.state.get_step_progress(self.step_name).processed_count
                    + advance,
                )
                head_last_processed = processed_count
                if self.formatter:
                    # Pass rate information to formatter if available
                    if rate is not None:
                        self.formatter.update_progress(advance, rate=rate)
                    else:
                        self.formatter.update_progress(advance)

            # Body and Head pose estimation
            if self.formatter:
                with self.formatter.create_progress_bar(
                    "Analyzing body poses", total_frames
                ):
                    poses_by_category, _ = pose_estimator.process_frame_batch(
                        frames_with_faces, body_progress_callback,
                        interruption_check=self._check_interrupted
                    )
                with self.formatter.create_progress_bar(
                    "Analyzing head angles", total_frames
                ):
                    head_angles_by_category, _ = (
                        head_pose_estimator.process_frame_batch(
                            frames_with_faces, head_progress_callback,
                            interruption_check=self._check_interrupted
                        )
                    )
            else:
                poses_by_category, _ = pose_estimator.process_frame_batch(
                    frames_with_faces, body_progress_callback,
                    interruption_check=self._check_interrupted
                )
                head_angles_by_category, _ = head_pose_estimator.process_frame_batch(
                    frames_with_faces, head_progress_callback,
                    interruption_check=self._check_interrupted
                )

            # Store results
            total_poses_found = sum(
                len(frame.pose_detections) for frame in frames_with_faces
            )
            total_head_poses_found = sum(
                len(frame.head_poses) for frame in frames_with_faces
            )

            self.state.get_step_progress(self.step_name).set_data(
                "poses_found", poses_by_category
            )
            self.state.get_step_progress(self.step_name).set_data(
                "head_angles_found", head_angles_by_category
            )
            self.state.get_step_progress(self.step_name).set_data(
                "total_poses_found", total_poses_found
            )
            self.state.get_step_progress(self.step_name).set_data(
                "total_head_poses_found", total_head_poses_found
            )
            self.state.get_step_progress(self.step_name).set_data(
                "total_analyzed", len(frames_with_faces)
            )

            top_poses = sorted(
                poses_by_category.items(), key=lambda x: x[1], reverse=True
            )[:3]
            top_head_poses = sorted(
                head_angles_by_category.items(), key=lambda x: x[1], reverse=True
            )[:3]

            if self.formatter:
                results = {
                    "total_analyzed": len(frames_with_faces),
                    "poses_found": poses_by_category,
                    "head_angles_found": head_angles_by_category,
                    "body_poses_summary": f"✅ Body poses: {total_poses_found} across {len(poses_by_category)} categories",
                    "head_poses_summary": f"✅ Head poses: {total_head_poses_found} across {len(head_angles_by_category)} categories",
                    "top_poses_display": f"🏆 Top poses: {', '.join([f'{name} ({count})' for name, count in top_poses])}",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(f"✅ Pose analysis completed")
                self.logger.info(
                    f"   📊 Body poses: {total_poses_found} across {len(poses_by_category)} categories"
                )
                self.logger.info(
                    f"   📊 Head poses: {total_head_poses_found} across {len(head_angles_by_category)} categories"
                )
                if top_poses:
                    self.logger.info(
                        f"   🏆 Top body poses: {', '.join([f'{name} ({count})' for name, count in top_poses])}"
                    )
                if top_head_poses:
                    self.logger.info(
                        f"   🏆 Top head poses: {', '.join([f'{name} ({count})' for name, count in top_head_poses])}"
                    )

        except Exception as e:
            self.logger.error(f"❌ Pose analysis failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise
