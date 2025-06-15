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
            # Check for interruption before starting
            self._check_interrupted()
            
            # Get video metadata (already extracted in initialization)
            video_metadata = self.pipeline.video_processor.extract_metadata()

            # Initialize frame extractor
            frame_extractor = FrameExtractor(
                str(self.pipeline.video_path), video_metadata
            )

            # Get frames output directory
            frames_dir = self.pipeline.temp_manager.get_frames_dir()

            # Check for interruption before analysis
            self._check_interrupted()
            
            # Pre-calculate actual candidates to get accurate progress total
            # This is better than using estimates since it accounts for existing frames
            if self.formatter and hasattr(self.formatter, "create_progress_bar"):
                with self.formatter.create_progress_bar("Analyzing video structure (may take time for large files)...") as spinner:
                    # Step 1: Get I-frame and temporal candidates (FFprobe can be slow)
                    self._check_interrupted()
                    i_frame_candidates = frame_extractor._extract_i_frames()
                    self.logger.info(f"Found {len(i_frame_candidates)} I-frame candidates")
                    
                    self._check_interrupted()
                    temporal_candidates = frame_extractor._generate_temporal_samples()
                    self.logger.info(f"Generated {len(temporal_candidates)} temporal sampling candidates")
                    
                    # Step 2: Combine and deduplicate candidates
                    self._check_interrupted()
                    all_candidates = frame_extractor._combine_and_deduplicate_candidates(
                        i_frame_candidates, temporal_candidates
                    )
                    self.logger.info(f"Combined to {len(all_candidates)} unique frame candidates")
                    
                    # Note: We don't filter out existing frames anymore since _extract_frame_images handles them
                    actual_total_frames = len(all_candidates)
                    self.logger.info(f"Processing {actual_total_frames} frame candidates (including existing frames)")
            else:
                # Fallback for non-formatter mode
                self.logger.info("🎬 Analyzing frame candidates...")
                
                # Step 1: Get I-frame and temporal candidates (fast operations)
                self._check_interrupted()
                i_frame_candidates = frame_extractor._extract_i_frames()
                self.logger.info(f"Found {len(i_frame_candidates)} I-frame candidates")
                
                self._check_interrupted()
                temporal_candidates = frame_extractor._generate_temporal_samples()
                self.logger.info(f"Generated {len(temporal_candidates)} temporal sampling candidates")
                
                # Step 2: Combine and deduplicate candidates
                self._check_interrupted()
                all_candidates = frame_extractor._combine_and_deduplicate_candidates(
                    i_frame_candidates, temporal_candidates
                )
                self.logger.info(f"Combined to {len(all_candidates)} unique frame candidates")
                
                # Note: We don't filter out existing frames anymore since _extract_frame_images handles them
                actual_total_frames = len(all_candidates)
                self.logger.info(f"Processing {actual_total_frames} frame candidates (including existing frames)")

            self.state.get_step_progress(self.step_name).start(actual_total_frames)

            last_processed_count = 0
            step_start_time = self._get_step_start_time()

            def progress_callback(current: int, total: int):
                nonlocal last_processed_count
                # Check for interruption on each progress update
                self._check_interrupted()
                
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

            # Check for interruption before frame extraction
            self._check_interrupted()
            
            # Extract frames using all candidates (existing frames will be loaded, new ones extracted)
            if self.formatter and hasattr(self.formatter, "step_progress_context"):
                with self.formatter.step_progress_context(
                    "Processing frames", actual_total_frames
                ):
                    extracted_frames = frame_extractor.extract_frames(
                        frames_dir, progress_callback, pre_calculated_candidates=all_candidates,
                        interruption_check=self._check_interrupted
                    )
            else:
                self.logger.info("🎬 Starting frame processing...")
                extracted_frames = frame_extractor.extract_frames(
                    frames_dir, progress_callback, pre_calculated_candidates=all_candidates,
                    interruption_check=self._check_interrupted
                )

            # Final interruption check
            self._check_interrupted()
            
            # Update state with final results
            self.state.get_step_progress(self.step_name).total_items = len(
                extracted_frames
            )
            self.state.update_step_progress(self.step_name, len(extracted_frames))

            # Manually update statistics since we bypassed extract_frames()
            frame_extractor.stats["i_frames_found"] = len(i_frame_candidates)
            frame_extractor.stats["temporal_samples"] = len(temporal_candidates)
            frame_extractor.stats["total_extracted"] = len(extracted_frames)

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
