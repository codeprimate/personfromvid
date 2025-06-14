import time
from .base import PipelineStep
from ..frame_extractor import FrameExtractor


class FrameExtractionStep(PipelineStep):
    """Pipeline step for extracting frames from the video."""

    @property
    def step_name(self) -> str:
        return "frame_extraction"

    def execute(self) -> None:
        """Extract frames using a hybrid approach."""
        self.state.start_step(self.step_name)

        try:
            # Get video metadata (already extracted in initialization)
            video_metadata = self.pipeline.video_processor.extract_metadata()

            # Initialize frame extractor
            frame_extractor = FrameExtractor(
                str(self.pipeline.video_path), video_metadata
            )

            # Get frames output directory
            frames_dir = self.pipeline.temp_manager.get_frames_dir()

            # Calculate realistic estimate for progress tracking
            temporal_samples = int(video_metadata.duration / 0.25)
            estimated_i_frames = max(1, int(video_metadata.duration / 1.5))
            estimated_before_dedup = temporal_samples + estimated_i_frames
            max_allowed = int(video_metadata.duration * 8)
            estimated_total_frames = min(estimated_before_dedup, max_allowed)

            self.state.get_step_progress(self.step_name).start(estimated_total_frames)

            last_processed_count = 0
            step_start_time = self._get_step_start_time()

            def progress_callback(current: int, total: int):
                nonlocal last_processed_count
                self.state.update_step_progress(self.step_name, current)
                advance_amount = current - last_processed_count
                last_processed_count = current

                if self.formatter and hasattr(self.formatter, "update_step_progress"):
                    rate = (
                        current / (time.time() - step_start_time)
                        if step_start_time and time.time() > step_start_time
                        else 0
                    )
                    self.formatter.update_step_progress(advance_amount, rate=rate)

            # Extract frames
            if self.formatter and hasattr(self.formatter, "step_progress_context"):
                with self.formatter.step_progress_context(
                    "Extracting frames", estimated_total_frames
                ):
                    extracted_frames = frame_extractor.extract_frames(
                        frames_dir, progress_callback
                    )
            else:
                self.logger.info("🎬 Starting frame extraction...")
                extracted_frames = frame_extractor.extract_frames(
                    frames_dir, progress_callback
                )

            # Update state with final results
            self.state.get_step_progress(self.step_name).total_items = len(
                extracted_frames
            )
            self.state.update_step_progress(self.step_name, len(extracted_frames))

            extraction_stats = frame_extractor.get_extraction_statistics()
            self.state.get_step_progress(self.step_name).set_data(
                "extraction_stats", extraction_stats
            )
            self.state.frames.extend(extracted_frames)

            # Store results for formatter
            if self.formatter:
                results = {
                    "i_frames_info": f"📊 I-frames found: {extraction_stats['i_frames_found']}",
                    "temporal_info": f"📊 Temporal samples: {extraction_stats['temporal_samples_generated']}",
                    "extraction_summary": f"Extracted {len(extracted_frames)} unique frames ({extraction_stats['duplicates_removed']} duplicates removed)",
                }
                self.state.get_step_progress(self.step_name).set_data(
                    "step_results", results
                )
            else:
                self.logger.info(
                    f"✅ Frame extraction completed: {len(extracted_frames)} frames"
                )
                self.logger.info(
                    f"   📊 I-frames: {extraction_stats['i_frames_found']}, "
                    f"Temporal: {extraction_stats['temporal_samples_generated']}, "
                    f"Duplicates removed: {extraction_stats['duplicates_removed']}"
                )

        except Exception as e:
            self.logger.error(f"❌ Frame extraction failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise
