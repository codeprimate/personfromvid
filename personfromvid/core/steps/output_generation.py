import time
from typing import List, Union

from ...analysis.person_selector import PersonSelection
from ...data.constants import ALL_SELECTED_FRAMES_KEY, ALL_SELECTED_PERSONS_KEY
from ...output.image_writer import ImageWriter
from .base import PipelineStep


class OutputGenerationStep(PipelineStep):
    """Pipeline step for generating output files with dual input support."""

    @property
    def step_name(self) -> str:
        return "output_generation"

    def execute(self) -> None:
        """Generate output files for selected frames or persons."""
        self.state.start_step(self.step_name)

        try:
            # Step 1: Input Type Detection
            input_data, input_type = self._detect_input_type()

            if not input_data:
                self._handle_no_input_data()
                return

            # Step 2: Process based on input type
            if input_type == "person_selection":
                self._process_person_selections(input_data)
            else:  # input_type == "frame_selection"
                self._process_frame_selections(input_data)

        except Exception as e:
            self.logger.error(f"❌ Output generation failed: {e}")
            self.state.fail_step(self.step_name, str(e))
            raise

    def _detect_input_type(self) -> tuple[Union[List[PersonSelection], List[str]], str]:
        """Detect input type and return (data, type).

        Returns:
            Tuple of (input_data, input_type) where input_type is
            "person_selection" or "frame_selection"
        """
        # Priority 1: Check for PersonSelection data (person-based pipeline)
        person_selection_progress = self.state.get_step_progress("person_selection")
        if person_selection_progress:
            serializable_persons = person_selection_progress.get_data(
                ALL_SELECTED_PERSONS_KEY, []
            )
            if serializable_persons:
                # Reconstruct PersonSelection objects from serializable data
                person_selections = self._reconstruct_person_selections(serializable_persons)
                return person_selections, "person_selection"

        # Priority 2: Check for frame selection data (backwards compatibility)
        frame_selection_progress = self.state.get_step_progress("frame_selection")
        if frame_selection_progress:
            selected_frame_ids = frame_selection_progress.get_data(
                ALL_SELECTED_FRAMES_KEY, []
            )
            if selected_frame_ids:
                return selected_frame_ids, "frame_selection"

        # No input data found
        return [], "none"

    def _reconstruct_person_selections(self, serializable_persons: List[dict]) -> List[PersonSelection]:
        """Reconstruct PersonSelection objects from serializable data.
        
        Args:
            serializable_persons: List of dictionaries containing person selection data
                                 or actual PersonSelection objects (for testing)
            
        Returns:
            List of PersonSelection objects
        """
        from ...analysis.person_selector import PersonSelection
        
        # Handle case where PersonSelection objects are passed directly (for tests)
        if serializable_persons and isinstance(serializable_persons[0], PersonSelection):
            return serializable_persons
        
        # Create lookup map for frames and persons
        frames_map = {frame.frame_id: frame for frame in self.state.frames}
        
        person_selections = []
        for data in serializable_persons:
            frame_id = data["frame_id"]
            person_id = data["person_id"]
            
            if frame_id not in frames_map:
                self.logger.warning(f"Frame {frame_id} not found, skipping person selection")
                continue
                
            frame_data = frames_map[frame_id]
            
            # Find the person in the frame
            person = None
            if hasattr(frame_data, 'persons') and frame_data.persons:
                for p in frame_data.persons:
                    if p.person_id == person_id:
                        person = p
                        break
            
            if person is None:
                self.logger.warning(f"Person {person_id} not found in frame {frame_id}, skipping")
                continue
                
            # Reconstruct PersonSelection object
            selection = PersonSelection(
                frame_data=frame_data,
                person_id=person_id,
                person=person,
                selection_score=data["selection_score"],
                category=data["category"]
            )
            person_selections.append(selection)
            
        return person_selections

    def _handle_no_input_data(self) -> None:
        """Handle case where no input data is available."""
        if self.formatter:
            self.formatter.print_warning(
                "No frames or persons selected for output generation"
            )
        else:
            self.logger.warning(
                "⚠️ No frames or persons selected for output generation"
            )
        self.state.get_step_progress(self.step_name).start(0)

    def _process_person_selections(
        self, person_selections: List[PersonSelection]
    ) -> None:
        """Process PersonSelection objects for output generation."""
        if self.formatter:
            self.formatter.print_info("👥 Creating person-based outputs...", "persons")
        else:
            self.logger.info(
                f"👥 Generating person-based output for {len(person_selections)} persons..."
            )

        total_persons = len(person_selections)
        self.state.get_step_progress(self.step_name).start(total_persons)

        output_dir = self.pipeline.context.output_directory
        image_writer = ImageWriter(context=self.pipeline.context)
        all_output_files = []
        step_start_time = time.time()

        def progress_callback(processed_count):
            self._check_interrupted()
            self.state.update_step_progress(self.step_name, processed_count)
            if self.formatter:
                # Calculate rate
                elapsed = time.time() - step_start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                self.formatter.update_progress(1, rate=rate)

        if self.formatter:
            with self.formatter.create_progress_bar(
                "Generating person files", total_persons
            ):
                for i, person_selection in enumerate(person_selections):
                    # Check for interruption at regular intervals
                    if i % 5 == 0:
                        self._check_interrupted()

                    try:
                        output_files = self._generate_output_for_person_selection(
                            person_selection, image_writer
                        )
                        all_output_files.extend(output_files)
                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Failed to generate output for person {person_selection.person_id}: {e}"
                        )
                        continue

                    progress_callback(i + 1)
        else:
            for i, person_selection in enumerate(person_selections):
                # Check for interruption at regular intervals
                if i % 5 == 0:
                    self._check_interrupted()

                try:
                    output_files = self._generate_output_for_person_selection(
                        person_selection, image_writer
                    )
                    all_output_files.extend(output_files)
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to generate output for person {person_selection.person_id}: {e}"
                    )
                    continue

                progress_callback(i + 1)

        self._finalize_output_generation(all_output_files, output_dir, "person-based")

    def _process_frame_selections(self, selected_frame_ids: List[str]) -> None:
        """Process frame IDs for output generation (backwards compatibility)."""
        # Map IDs to the actual FrameData objects
        all_frames_map = {frame.frame_id: frame for frame in self.state.frames}
        selected_frames = [
            all_frames_map[fid] for fid in selected_frame_ids if fid in all_frames_map
        ]

        if not selected_frames:
            self._handle_no_input_data()
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
        step_start_time = time.time()

        def progress_callback(processed_count):
            self._check_interrupted()
            self.state.update_step_progress(self.step_name, processed_count)
            if self.formatter:
                # Calculate rate
                elapsed = time.time() - step_start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                self.formatter.update_progress(1, rate=rate)

        if self.formatter:
            with self.formatter.create_progress_bar("Generating files", total_frames):
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

        self._finalize_output_generation(all_output_files, output_dir, "frame-based")

    def _generate_output_for_person_selection(
        self, person_selection: PersonSelection, image_writer: ImageWriter
    ) -> List[str]:
        """Generate output files for a single PersonSelection.

        Args:
            person_selection: PersonSelection object containing frame and person data
            image_writer: ImageWriter instance for file generation

        Returns:
            List of generated output file paths
        """
        # Use person-specific output generation
        return image_writer.save_person_outputs(person_selection)

    def _finalize_output_generation(
        self, all_output_files: List[str], output_dir, generation_type: str
    ) -> None:
        """Finalize output generation with statistics and logging.

        Args:
            all_output_files: List of generated output file paths
            output_dir: Output directory path
            generation_type: Type of generation ("person-based" or "frame-based")
        """
        self.state.processing_stats["output_files"] = all_output_files
        self.state.processing_stats["total_output_files"] = len(all_output_files)

        if self.formatter:
            self.state.get_step_progress(self.step_name).set_data(
                "step_results",
                {
                    "files_generated": f"✅ Generated {len(all_output_files)} files ({generation_type})",
                    "location_info": f"📂 Location: {output_dir}",
                },
            )
        else:
            self.logger.info(
                f"✅ Output generation completed: {len(all_output_files)} files ({generation_type}) in {output_dir}"
            )
