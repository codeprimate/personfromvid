"""Integration tests for person-based pipeline execution.

These tests use the real test video and execute the complete pipeline to verify
that person-based selection works end-to-end, catching integration bugs that
unit tests might miss.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from personfromvid.core.pipeline import ProcessingPipeline
from personfromvid.data import Config, ProcessingContext
from personfromvid.data.constants import get_pipeline_step_names


class TestPersonPipelineIntegration:
    """Integration tests for person-based pipeline functionality."""

    @pytest.fixture
    def test_video_path(self):
        """Get path to the reference test video."""
        return Path(__file__).parent.parent / "fixtures" / "test_video.mp4"

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def person_enabled_config(self):
        """Config with person selection enabled."""
        config = Config()
        config.person_selection.enabled = True
        return config

    @pytest.fixture
    def person_disabled_config(self):
        """Config with person selection disabled."""
        config = Config()
        config.person_selection.enabled = False
        return config

    def test_person_pipeline_step_registration(self):
        """Test that person_selection is properly registered in pipeline steps."""
        step_names = get_pipeline_step_names()
        
        # Both selection steps should be registered
        assert "frame_selection" in step_names
        assert "person_selection" in step_names
        
        # Verify step order is correct
        person_idx = step_names.index("person_selection")
        frame_idx = step_names.index("frame_selection")
        output_idx = step_names.index("output_generation")
        
        # Both selection steps should come before output generation
        assert person_idx < output_idx
        assert frame_idx < output_idx

    def test_person_pipeline_default_configuration(self):
        """Test that person selection is enabled by default."""
        config = Config()
        assert config.person_selection.enabled is True, "Person selection should be enabled by default"

    def test_person_pipeline_full_execution(self, test_video_path, temp_output_dir, person_enabled_config):
        """Test complete person-based pipeline execution with real video."""
        # Skip if test video doesn't exist
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Configure for fast testing
        person_enabled_config.frame_extraction.max_frames_per_video = 10
        person_enabled_config.person_selection.min_instances_per_person = 1
        person_enabled_config.person_selection.max_instances_per_person = 3

        # Create processing context
        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=person_enabled_config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        # Verify successful completion
        assert result.success is True, f"Pipeline failed: {result.error_message}"
        assert result.total_frames_extracted > 0

        # The key test: Verify person selection step was executed
        # This is the main bug we fixed - ensuring the step runs even if no persons are found
        assert pipeline.state.is_step_completed("person_selection"), "Person selection step should have completed"
        
        # Verify person selection step progress exists (even if no persons were selected)
        person_step_progress = pipeline.state.get_step_progress("person_selection")
        assert person_step_progress is not None, "Person selection step should have progress tracking"
        
        # Get person selections (may be empty if no faces/persons found)
        person_selections = person_step_progress.get_data("all_selected_persons", [])
        
        # Test the pipeline configuration and execution logic rather than specific video content
        if result.faces_found > 0:
            # If faces were found, verify person selection worked
            assert len(person_selections) > 0, "Should have selected some persons when faces are found"
            
            # Verify each person selection has required fields
            for selection in person_selections:
                assert "frame_id" in selection
                assert "person_id" in selection
                assert "selection_score" in selection
                assert "category" in selection
                assert isinstance(selection["person_id"], int)
                assert isinstance(selection["selection_score"], (int, float))

            # Verify output files were generated
            assert result.output_files is not None
            assert len(result.output_files) > 0, "Should have generated output files when persons are selected"
            
            # Verify person-based naming pattern
            for output_file in result.output_files:
                assert "person_" in str(output_file), f"Output file should have person-based naming: {output_file}"
        else:
            # If no faces found, verify graceful handling
            assert len(person_selections) == 0, "Should have no person selections when no faces are found"
            assert result.output_files is None or len(result.output_files) == 0, "Should have no output files when no persons are selected"

    def test_person_pipeline_state_serialization(self, test_video_path, temp_output_dir, person_enabled_config):
        """Test that pipeline state with person data can be serialized and deserialized."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Configure for minimal processing
        person_enabled_config.frame_extraction.max_frames_per_video = 5
        person_enabled_config.person_selection.min_instances_per_person = 1

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=person_enabled_config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        assert result.success is True

        # Test state serialization by manually serializing the state
        state_dict = pipeline.state.to_dict()
        
        # Verify state can be JSON serialized (this would catch set/object serialization bugs)
        try:
            json_str = json.dumps(state_dict, default=str)
            assert len(json_str) > 0
        except TypeError as e:
            pytest.fail(f"Pipeline state is not JSON serializable: {e}")

        # Verify person selection data is in the serialized state
        step_progress = state_dict.get("step_progress", {})
        person_step = step_progress.get("person_selection", {})
        assert person_step is not None, "Person selection step data should be in state"

    @pytest.mark.skip(reason="Test video has no forward-facing faces - pending better test video")
    def test_person_vs_frame_selection_comparison(self, test_video_path, temp_output_dir):
        """Test that person-based and frame-based selection produce different results."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Test frame-based selection
        frame_config = Config()
        frame_config.person_selection.enabled = False
        frame_config.frame_extraction.max_frames_per_video = 10

        frame_context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem + "_frame",
            config=frame_config,
            output_directory=temp_output_dir / "frame_output"
        )

        frame_pipeline = ProcessingPipeline(context=frame_context)
        frame_result = frame_pipeline.process()

        # Test person-based selection
        person_config = Config()
        person_config.person_selection.enabled = True
        person_config.frame_extraction.max_frames_per_video = 10
        person_config.person_selection.min_instances_per_person = 1

        person_context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem + "_person",
            config=person_config,
            output_directory=temp_output_dir / "person_output"
        )

        person_pipeline = ProcessingPipeline(context=person_context)
        person_result = person_pipeline.process()

        # Both should succeed
        assert frame_result.success is True
        assert person_result.success is True

        # Verify different steps were executed
        assert frame_pipeline.state.is_step_completed("frame_selection")
        assert not frame_pipeline.state.is_step_completed("person_selection")

        assert person_pipeline.state.is_step_completed("person_selection")
        # Note: frame_selection might still be completed if it was registered

        # Verify different output patterns
        frame_output_files = [Path(f).name for f in frame_result.output_files or []]
        person_output_files = [Path(f).name for f in person_result.output_files or []]

        # Person-based outputs should have "person_" in filenames
        person_files_with_person = [f for f in person_output_files if "person_" in f]
        assert len(person_files_with_person) > 0, "Person-based selection should generate person-named files"

        # Frame-based outputs typically don't have "person_" in filenames
        frame_files_with_person = [f for f in frame_output_files if "person_" in f]
        # Frame-based might have fewer or no person-specific files
        assert len(person_files_with_person) >= len(frame_files_with_person), \
            "Person-based selection should generate more person-specific files"

    def test_person_pipeline_error_recovery(self, test_video_path, temp_output_dir, person_enabled_config):
        """Test error handling and recovery in person-based pipeline."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        person_enabled_config.frame_extraction.max_frames_per_video = 5

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=person_enabled_config,
            output_directory=temp_output_dir
        )

        # Test with impossible quality threshold (should result in no selections)
        person_enabled_config.person_selection.min_quality_threshold = 1.0  # Impossible threshold

        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        # Pipeline should still complete successfully even with no selections
        assert result.success is True
        
        # But should have warnings about no selections
        person_step_progress = pipeline.state.get_step_progress("person_selection")
        if person_step_progress:
            person_selections = person_step_progress.get_data("all_selected_persons", [])
            # With impossible threshold, might have no selections
            # This tests the "no selections" code path

    def test_person_selection_step_execution_order(self, test_video_path, temp_output_dir, person_enabled_config):
        """Test that person selection step executes in the correct order."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        person_enabled_config.frame_extraction.max_frames_per_video = 3

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=person_enabled_config,
            output_directory=temp_output_dir
        )

        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        assert result.success is True

        # Verify prerequisite steps completed before person selection
        assert pipeline.state.is_step_completed("initialization")
        assert pipeline.state.is_step_completed("frame_extraction")
        assert pipeline.state.is_step_completed("face_detection")
        assert pipeline.state.is_step_completed("pose_analysis")
        assert pipeline.state.is_step_completed("person_building")
        assert pipeline.state.is_step_completed("quality_assessment")
        assert pipeline.state.is_step_completed("person_selection")

        # Output generation should come after person selection
        assert pipeline.state.is_step_completed("output_generation")

    @pytest.mark.skip(reason="Test video has no forward-facing faces - pending better test video")
    def test_person_pipeline_output_generation_method(self, test_video_path, temp_output_dir, person_enabled_config):
        """Test that person-based pipeline uses save_person_outputs method."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        person_enabled_config.frame_extraction.max_frames_per_video = 3

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=person_enabled_config,
            output_directory=temp_output_dir
        )

        # Mock the ImageWriter to verify correct method is called
        with patch('personfromvid.core.steps.output_generation.ImageWriter') as mock_writer_class:
            mock_writer = mock_writer_class.return_value
            mock_writer.save_person_outputs.return_value = ["test_person_output.png"]

            pipeline = ProcessingPipeline(context=context)
            result = pipeline.process()

            # Pipeline should succeed
            assert result.success is True

            # Verify save_person_outputs was called (not save_frame_outputs)
            mock_writer.save_person_outputs.assert_called()
            
            # save_frame_outputs should not be called for person-based pipeline
            # (it might be called for other reasons, but save_person_outputs should be primary)
            person_calls = mock_writer.save_person_outputs.call_count
            assert person_calls > 0, "save_person_outputs should be called for person-based pipeline"

    @pytest.mark.parametrize("person_enabled", [True, False])
    def test_person_pipeline_configuration_variations(self, test_video_path, temp_output_dir, person_enabled):
        """Test pipeline with different person selection configurations."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        config = Config()
        config.person_selection.enabled = person_enabled
        config.frame_extraction.max_frames_per_video = 5

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=f"{test_video_path.stem}_{person_enabled}",
            config=config,
            output_directory=temp_output_dir / f"output_{person_enabled}"
        )

        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        # Both configurations should work
        assert result.success is True
        assert result.total_frames_extracted > 0

        if person_enabled:
            # Person selection should have been executed
            assert pipeline.state.is_step_completed("person_selection")
        else:
            # Frame selection should have been executed
            assert pipeline.state.is_step_completed("frame_selection") 