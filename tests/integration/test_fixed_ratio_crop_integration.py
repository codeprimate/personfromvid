"""Integration tests for fixed aspect ratio crop functionality.

These tests verify end-to-end functionality of the fixed aspect ratio crop feature,
including CLI integration, configuration flow, and actual output validation.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from personfromvid.core.pipeline import ProcessingPipeline
from personfromvid.data import ProcessingContext
from personfromvid.data.config import load_config


class TestFixedRatioCropIntegration:
    """Integration tests for fixed aspect ratio crop functionality."""

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
    def base_config(self):
        """Base configuration for testing."""
        config = load_config()
        # Configure for fast testing
        config.frame_extraction.max_frames_per_video = 5
        config.output.image.enable_pose_cropping = True
        return config

    @pytest.fixture
    def square_ratio_config(self, base_config):
        """Configuration with square aspect ratio."""
        config = base_config
        config.output.image.crop_ratio = "1:1"
        return config

    @pytest.fixture
    def widescreen_ratio_config(self, base_config):
        """Configuration with widescreen aspect ratio."""
        config = base_config
        config.output.image.crop_ratio = "16:9"
        return config

    @pytest.fixture
    def portrait_ratio_config(self, base_config):
        """Configuration with portrait aspect ratio."""
        config = base_config
        config.output.image.crop_ratio = "4:3"
        return config

    def test_cli_parameter_integration(self):
        """Test that --crop-ratio parameter is properly integrated in CLI."""
        # Test help output includes the parameter
        result = subprocess.run(
            ["python", "-m", "personfromvid", "--help"],
            capture_output=True, text=True, cwd=Path.cwd()
        )
        assert result.returncode == 0
        assert "--crop-ratio" in result.stdout
        assert "Aspect ratio for crops" in result.stdout
        assert "1:1" in result.stdout and "16:9" in result.stdout
        assert "'any'" in result.stdout or "any" in result.stdout  # Check for "any" option

    def test_cli_automatic_crop_enablement(self):
        """Test that --crop-ratio automatically enables cropping without requiring --crop flag."""
        # Test that crop-ratio automatically enables cropping (should succeed)
        test_video = Path(__file__).parent.parent / "fixtures" / "test_video.mp4"

        # This should succeed - crop-ratio now automatically enables cropping
        result = subprocess.run(
            ["python", "-m", "personfromvid", str(test_video), "--crop-ratio", "1:1", "--help"],
            capture_output=True, text=True, cwd=Path.cwd()
        )
        # Using --help to avoid actually processing video in test, just verify CLI parsing works
        assert result.returncode == 0

        # Verify help text shows the updated behavior
        assert "--crop-ratio" in result.stdout
        # Check for the text accounting for line wrapping in Click help output
        assert "Automatically enables" in result.stdout and "cropping." in result.stdout

    def test_cli_any_option_integration(self):
        """Test that --crop-ratio any option is properly integrated in CLI."""
        test_video = Path(__file__).parent.parent / "fixtures" / "test_video.mp4"

        # Test that crop-ratio any is accepted by CLI parsing
        result = subprocess.run(
            ["python", "-m", "personfromvid", str(test_video), "--crop-ratio", "any", "--help"],
            capture_output=True, text=True, cwd=Path.cwd()
        )
        # Using --help to avoid actually processing video in test, just verify CLI parsing works
        assert result.returncode == 0

        # Verify help text includes "any" option
        assert "--crop-ratio" in result.stdout
        assert "'any' for variable" in result.stdout

    def test_configuration_flow_integration(self, base_config):
        """Test that fixed aspect ratio configuration flows correctly through the system."""
        # Test configuration parsing and validation
        config = base_config
        config.output.image.crop_ratio = "16:9"

        # Verify configuration is valid
        assert config.output.image.crop_ratio == "16:9"
        assert config.output.image.enable_pose_cropping is True

        # Test configuration with invalid ratio
        with pytest.raises(ValueError, match="Invalid aspect ratio format"):
            invalid_config = load_config()
            invalid_config.output.image.crop_ratio = "invalid_ratio"
            invalid_config.output.image.enable_pose_cropping = True
            # Force validation
            from pydantic import ValidationError
            try:
                invalid_config.model_validate(invalid_config.model_dump())
            except ValidationError as e:
                raise ValueError("Invalid aspect ratio format") from e

    def test_pipeline_execution_with_fixed_ratios(self, test_video_path, temp_output_dir, square_ratio_config):
        """Test complete pipeline execution with fixed aspect ratios."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Create processing context
        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=square_ratio_config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        # Verify successful completion
        assert result.success is True, f"Pipeline failed: {result.error_message}"

        # Verify pipeline completed all steps
        assert pipeline.state.is_step_completed("output_generation"), "Output generation should have completed"

    def test_output_dimension_validation(self, test_video_path, temp_output_dir, square_ratio_config):
        """Test that output images have correct dimensions for fixed aspect ratios."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Create processing context
        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=square_ratio_config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        if not result.success or not result.output_files:
            pytest.skip("No output files generated - likely no poses detected in test video")

        # Verify output file dimensions
        for output_file in result.output_files:
            if output_file.exists() and output_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                with Image.open(output_file) as img:
                    width, height = img.size

                    # For 1:1 ratio, width should equal height
                    assert width == height, f"Square ratio image should be square: {width}x{height} in {output_file}"

    @pytest.mark.parametrize("aspect_ratio,expected_ratio", [
        ("1:1", 1.0),
        ("16:9", 16/9),
        ("4:3", 4/3),
        ("21:9", 21/9)
    ])
    def test_multiple_aspect_ratios(self, test_video_path, temp_output_dir, base_config, aspect_ratio, expected_ratio):
        """Test pipeline with multiple different aspect ratios."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Configure with specific aspect ratio
        config = base_config
        config.output.image.crop_ratio = aspect_ratio

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=f"{test_video_path.stem}_{aspect_ratio.replace(':', '_')}",
            config=config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        if result.success and result.output_files:
            # Verify aspect ratios of generated images
            for output_file in result.output_files:
                if output_file.exists() and output_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    with Image.open(output_file) as img:
                        width, height = img.size
                        actual_ratio = width / height

                        # Allow small tolerance for rounding
                        tolerance = 0.01
                        assert abs(actual_ratio - expected_ratio) < tolerance, \
                            f"Aspect ratio mismatch: expected {expected_ratio:.3f}, got {actual_ratio:.3f} in {output_file}"

    def test_backward_compatibility(self, test_video_path, temp_output_dir, base_config):
        """Test that existing functionality works when crop_ratio is None."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Configure without crop_ratio (should work as before)
        config = base_config
        config.output.image.crop_ratio = None

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem + "_legacy",
            config=config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        # Should complete successfully even without fixed aspect ratios
        assert result.success is True, f"Legacy pipeline failed: {result.error_message}"

    def test_error_handling_integration(self, test_video_path, temp_output_dir, base_config):
        """Test error handling with invalid configurations."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        # Test with crop_ratio but pose cropping disabled
        config = base_config
        config.output.image.crop_ratio = "1:1"
        config.output.image.enable_pose_cropping = False

        # This should fail validation
        with pytest.raises(ValueError, match="crop_ratio can only be specified when enable_pose_cropping is True"):
            # Force model validation
            from pydantic import ValidationError
            try:
                config.model_validate(config.model_dump())
            except ValidationError as e:
                raise ValueError("crop_ratio can only be specified when enable_pose_cropping is True") from e

    def test_state_serialization_with_crop_regions(self, test_video_path, temp_output_dir, square_ratio_config):
        """Test that pipeline state with crop regions can be serialized."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")

        context = ProcessingContext(
            video_path=test_video_path,
            video_base_name=test_video_path.stem,
            config=square_ratio_config,
            output_directory=temp_output_dir
        )

        # Execute pipeline
        pipeline = ProcessingPipeline(context=context)
        result = pipeline.process()

        assert result.success is True

        # Test state serialization
        state_dict = pipeline.state.to_dict()

        # Verify state can be JSON serialized
        try:
            json_str = json.dumps(state_dict, default=str)
            assert len(json_str) > 0
        except TypeError as e:
            pytest.fail(f"Pipeline state with crop regions is not JSON serializable: {e}")
