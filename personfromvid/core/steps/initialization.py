from .base import PipelineStep
from ...utils.exceptions import VideoProcessingError


class InitializationStep(PipelineStep):
    """Pipeline step for initializing the processing environment."""

    @property
    def step_name(self) -> str:
        return "initialization"

    def execute(self) -> None:
        """Validate video and set up the processing environment."""
        self.state.start_step(self.step_name)

        try:
            if self.formatter:
                self.formatter.print_info(
                    "🔧 Setting up processing environment...", "setup"
                )
            else:
                self.logger.info("🔧 Initializing processing environment...")

            # Video processor is already initialized in the pipeline
            # Validate video format and extract basic metadata
            if self.formatter:
                progress_bar = self.formatter.create_progress_bar(
                    "Setting up environment"
                )
                with progress_bar:
                    self.pipeline.video_processor.validate_format()
                    video_info = self.pipeline.video_processor.get_video_info_summary()
                    self.state.get_step_progress(self.step_name).set_data(
                        "video_info", video_info
                    )
            else:
                self.logger.info("🔍 Validating video format...")
                self.pipeline.video_processor.validate_format()
                video_info = self.pipeline.video_processor.get_video_info_summary()
                self.state.get_step_progress(self.step_name).set_data(
                    "video_info", video_info
                )

            # Store results for formatted output
            if self.formatter:
                results = {
                    "validation_success": "✅ Video format validated",
                    "workspace_success": "✅ Temporary workspace created",
                    "pipeline_success": "✅ Processing pipeline ready",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Video validated: {video_info['resolution']}, "
                    f"{video_info['duration_seconds']:.1f}s, "
                    f"{video_info['fps']:.1f}fps"
                )
                self.logger.info("✅ Processing environment initialized")

        except Exception as e:
            error_msg = f"Video validation failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            self.state.fail_step(self.step_name, error_msg)
            raise VideoProcessingError(error_msg)
