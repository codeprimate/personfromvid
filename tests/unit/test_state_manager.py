"""Unit tests for StateManager."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from personfromvid.core import StateManager
from personfromvid.core.temp_manager import TempManager
from personfromvid.data import Config, PipelineState, ProcessingContext, VideoMetadata
from personfromvid.utils.exceptions import StateLoadError, StateSaveError

# Test constants
TEST_FACE_MODEL = "scrfd_10g"


class TestStateManager:
    """Test StateManager functionality."""

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Write some dummy content
            f.write(b"dummy video content for testing")
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

        # Also cleanup potential state file
        state_file = temp_path.parent / f"{temp_path.stem}_info.json"
        if state_file.exists():
            state_file.unlink()

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir

        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def processing_context(self, temp_video_file, temp_cache_dir):
        """Create ProcessingContext for testing."""
        from personfromvid.data.config import StorageConfig

        # Create config with temporary cache directory
        storage_config = StorageConfig(cache_directory=temp_cache_dir)
        config = Config(storage=storage_config)
        output_dir = temp_video_file.parent / 'test_output'

        # We need a real TempManager for StateManager tests as it deals with file paths
        with patch('personfromvid.core.temp_manager.TempManager') as MockTempManager:
            # Create a real temp manager instance to be returned by the mock
            real_temp_manager = TempManager(str(temp_video_file), config)
            real_temp_manager.create_temp_structure()
            MockTempManager.return_value = real_temp_manager

            context = ProcessingContext(
                video_path=temp_video_file,
                video_base_name=temp_video_file.stem,
                config=config,
                output_directory=output_dir
            )

            yield context

            # Cleanup
            real_temp_manager.cleanup_temp_files()

    @pytest.fixture
    def sample_pipeline_state(self):
        """Create a sample pipeline state for testing."""
        video_metadata = VideoMetadata(
            duration=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            total_frames=3615,
            file_size_bytes=50000000,
            format="mp4"
        )

        now = datetime.now()
        return PipelineState(
            video_file="/path/to/video.mp4",
            video_hash="abc123def456",
            video_metadata=video_metadata,
            model_versions={
                "face_detection": "TEST_FACE_MODEL.onnx",
                "pose_estimation": "yolov8n-pose.pt"
            },
            created_at=now,
            last_updated=now
        )

    def test_state_manager_initialization(self, processing_context):
        """Test StateManager initialization."""
        manager = StateManager(context=processing_context)

        assert manager.video_path == processing_context.video_path
        assert manager.state_file_path.name == f"{processing_context.video_base_name}_info.json"
        assert manager.state_file_path.parent == processing_context.temp_manager.get_temp_path()

    def test_load_state_no_file(self, processing_context):
        """Test loading state when no state file exists."""
        manager = StateManager(context=processing_context)
        state = manager.load_state()

        assert state is None

    def test_save_and_load_state(self, processing_context, sample_pipeline_state):
        """Test saving and loading pipeline state."""
        manager = StateManager(context=processing_context)

        # Mock the video hash calculation to match our test state
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            # Update state to match our video file
            sample_pipeline_state.video_file = str(processing_context.video_path)

            # Save state
            manager.save_state(sample_pipeline_state)

            # Verify state file was created
            assert manager.state_file_path.exists()

            # Load state
            loaded_state = manager.load_state()

            assert loaded_state is not None
            assert loaded_state.video_file == str(processing_context.video_path)
            assert loaded_state.video_hash == "abc123def456"
            assert loaded_state.video_metadata.duration == 120.5

    def test_save_state_creates_backup(self, processing_context, sample_pipeline_state):
        """Test that saving state creates backup of existing file."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            # Update state to match our video file
            sample_pipeline_state.video_file = str(processing_context.video_path)

            # Save state first time
            manager.save_state(sample_pipeline_state)

            # Modify state and save again
            sample_pipeline_state.current_step = "face_detection"
            manager.save_state(sample_pipeline_state)

            # Load state and verify it has the updated step
            loaded_state = manager.load_state()
            assert loaded_state.current_step == "face_detection"

    def test_validate_state_video_hash_mismatch(self, processing_context, sample_pipeline_state):
        """Test state validation when video hash doesn't match."""
        manager = StateManager(context=processing_context)

        # Save state with one hash
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)
            manager.save_state(sample_pipeline_state)

        # Try to load with different hash
        with patch.object(manager, '_calculate_video_hash', return_value="different_hash"):
            with pytest.raises(StateLoadError, match="Video file has been modified"):
                manager.load_state()

    def test_validate_state_video_file_missing(self, processing_context, sample_pipeline_state):
        """Test state validation when video file is missing."""
        manager = StateManager(context=processing_context)

        # Save state
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)
            manager.save_state(sample_pipeline_state)

        # Remove video file
        processing_context.video_path.unlink()

        # Try to load state
        with pytest.raises(StateLoadError, match="Video file no longer exists"):
            manager.load_state()

    def test_update_step_progress(self, processing_context, sample_pipeline_state):
        """Test updating step progress."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)
            sample_pipeline_state.start_step("frame_extraction", 1000)
            manager.save_state(sample_pipeline_state)

            # Update progress
            manager.update_step_progress("frame_extraction", {"processed_count": 500})

            # Load and verify
            updated_state = manager.load_state()
            progress = updated_state.get_step_progress("frame_extraction")
            assert progress.processed_count == 500

    def test_mark_step_complete(self, processing_context, sample_pipeline_state):
        """Test marking step as complete."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)
            sample_pipeline_state.start_step("frame_extraction", 1000)
            manager.save_state(sample_pipeline_state)

            # Mark complete
            manager.mark_step_complete("frame_extraction")

            # Load and verify
            updated_state = manager.load_state()
            assert updated_state.is_step_completed("frame_extraction") is True

    def test_can_resume(self, processing_context, sample_pipeline_state):
        """Test can_resume functionality."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)

            # No state saved yet
            assert manager.can_resume() is False

            # Save incomplete state - complete one step but not all
            sample_pipeline_state.start_step("initialization", 1)
            sample_pipeline_state.complete_step("initialization")
            sample_pipeline_state.start_step("frame_extraction", 1000)
            manager.save_state(sample_pipeline_state)

            # Should be able to resume
            assert manager.can_resume() is True

            # Complete all steps to make pipeline fully completed
            all_steps = ["initialization", "frame_extraction", "face_detection", "pose_analysis",
                        "closeup_detection", "quality_assessment", "frame_selection", "output_generation"]
            for step in all_steps:
                sample_pipeline_state.complete_step(step)
            manager.save_state(sample_pipeline_state)

            # Should not be able to resume completed pipeline
            assert manager.can_resume() is False

    def test_get_resume_point(self, processing_context, sample_pipeline_state):
        """Test getting resume point."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)

            # No state saved yet
            assert manager.get_resume_point() is None

            # Save state with current step
            sample_pipeline_state.current_step = "face_detection"
            manager.save_state(sample_pipeline_state)

            # Should return current step
            assert manager.get_resume_point() == "face_detection"

    def test_delete_state(self, processing_context, sample_pipeline_state):
        """Test deleting state file."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)
            manager.save_state(sample_pipeline_state)

            # Verify state file exists
            assert manager.state_file_path.exists()

            # Delete state
            manager.delete_state()

            # Verify state file is gone
            assert not manager.state_file_path.exists()

            # Should be safe to call again
            manager.delete_state()

    def test_get_state_info(self, processing_context, sample_pipeline_state):
        """Test getting state information."""
        manager = StateManager(context=processing_context)

        # No state file
        info = manager.get_state_info()
        assert info is None

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)
            sample_pipeline_state.current_step = "pose_analysis"
            manager.save_state(sample_pipeline_state)

            # With state file
            info = manager.get_state_info()
            assert info is not None
            assert info["current_step"] == "pose_analysis"
            assert info["can_resume"] is False  # No completed steps yet
            assert "created_at" in info
            assert "last_updated" in info

    def test_context_manager(self, processing_context, sample_pipeline_state):
        """Test StateManager as context manager."""
        with patch.object(StateManager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)

            with StateManager(context=processing_context) as manager:
                manager.save_state(sample_pipeline_state)
                assert manager.state_file_path.exists()

            # Context manager should not delete state file
            assert manager.state_file_path.exists()

    def test_save_state_error_handling(self, processing_context, sample_pipeline_state):
        """Test error handling during state save."""
        manager = StateManager(context=processing_context)

        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(StateSaveError, match="Cannot save state"):
                manager.save_state(sample_pipeline_state)

    def test_load_state_corrupted_file(self, processing_context):
        """Test loading corrupted state file."""
        manager = StateManager(context=processing_context)

        # Create corrupted state file
        manager.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        manager.state_file_path.write_text("invalid json content")

        with pytest.raises(StateLoadError, match="Cannot load state"):
            manager.load_state()

    def test_calculate_video_hash(self, processing_context):
        """Test video hash calculation."""
        manager = StateManager(context=processing_context)

        # Calculate hash twice - should be consistent
        hash1 = manager._calculate_video_hash()
        hash2 = manager._calculate_video_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length
        assert isinstance(hash1, str)

    def test_backup_and_restore(self, processing_context, sample_pipeline_state):
        """Test backup and restore functionality."""
        manager = StateManager(context=processing_context)

        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(processing_context.video_path)

            # Save initial state
            sample_pipeline_state.current_step = "initialization"
            manager.save_state(sample_pipeline_state)

            # Save updated state (should create backup)
            sample_pipeline_state.current_step = "frame_extraction"
            manager.save_state(sample_pipeline_state)

            # Verify current state
            loaded_state = manager.load_state()
            assert loaded_state.current_step == "frame_extraction"
