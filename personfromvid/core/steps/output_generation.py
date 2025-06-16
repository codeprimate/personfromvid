from .base import PipelineStep
from ...output.image_writer import ImageWriter
from ...data.constants import ALL_SELECTED_FRAMES_KEY


class OutputGenerationStep(PipelineStep):
    """Pipeline step for generating output files."""

    @property
    def step_name(self) -> str:
        return "output_generation"

    def execute(self) -> None:
        """Generate output files for selected frames."""
        self.state.start_step(self.step_name)

        try:
            # Get selected frame IDs from the frame selection step
            frame_selection_progress = self.state.get_step_progress("frame_selection")
            selected_frame_ids = frame_selection_progress.get_data(
                ALL_SELECTED_FRAMES_KEY, []
            )

            # Map IDs to the actual FrameData objects
            all_frames_map = {frame.frame_id: frame for frame in self.state.frames}
            selected_frames = [
                all_frames_map[fid]
                for fid in selected_frame_ids
                if fid in all_frames_map
            ]

            if not selected_frames:
                if self.formatter:
                    self.formatter.print_warning(
                        "No frames selected for output generation"
                    )
                else:
                    self.logger.warning("⚠️ No frames selected for output generation")
                self.state.get_step_progress(self.step_name).start(0)
                return

            output_dir = self.pipeline.context.output_directory
            image_writer = ImageWriter(context=self.pipeline.context)

            if self.formatter:
                self.formatter.print_info("📁 Creating output files...", "files")
            else:
                self.logger.info(f"📁 Generating output in {output_dir}...")

            total_frames = len(selected_frames)
            self.state.get_step_progress(self.step_name).start(total_frames)

            all_output_files = []

            def progress_callback(processed_count):
                self._check_interrupted()
                self.state.update_step_progress(self.step_name, processed_count)
                if self.formatter:
                    self.formatter.update_progress(1)

            if self.formatter:
                with self.formatter.create_progress_bar(
                    "Generating files", total_frames
                ):
                    for i, frame in enumerate(selected_frames):
                        # Check for interruption at regular intervals
                        if i % 5 == 0:
                            self._check_interrupted()
                            
                        output_files = image_writer.save_frame_outputs(frame)
                        all_output_files.extend(output_files)
                        progress_callback(i + 1)
            else:
                for i, frame in enumerate(selected_frames):
                    # Check for interruption at regular intervals
                    if i % 5 == 0:
                        self._check_interrupted()
                        
                    output_files = image_writer.save_frame_outputs(frame)
                    all_output_files.extend(output_files)
                    progress_callback(i + 1)

            self.state.processing_stats["output_files"] = all_output_files
            self.state.processing_stats["total_output_files"] = len(all_output_files)

            if self.formatter:
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results",
                    {
                        "files_generated": f"✅ Generated {len(all_output_files)} files",
                        "location_info": f"📂 Location: {output_dir}",
                    },
                )
            else:
                self.logger.info(
                    f"✅ Output generation completed: {len(all_output_files)} files in {output_dir}"
                )

        except Exception as e:
            self.logger.error(f"❌ Output generation failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise
