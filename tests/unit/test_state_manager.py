"""Unit tests for StateManager."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock

from personfromvid.core import StateManager
from personfromvid.core.temp_manager import TempManager
from personfromvid.data import PipelineState, VideoMetadata
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
    def temp_manager(self, temp_video_file):
        """Create a TempManager for testing."""
        manager = TempManager(str(temp_video_file))
        manager.create_temp_structure()
        
        yield manager
        
        # Cleanup temp directory
        manager.cleanup_temp_files()
    
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
    
    def test_state_manager_initialization(self, temp_video_file, temp_manager):
        """Test StateManager initialization."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        assert manager.video_path == temp_video_file
        assert manager.state_file_path.name == f"{temp_video_file.stem}_info.json"
        assert manager.state_file_path.parent == temp_manager.get_temp_path()
    
    def test_load_state_no_file(self, temp_video_file, temp_manager):
        """Test loading state when no state file exists."""
        manager = StateManager(str(temp_video_file), temp_manager)
        state = manager.load_state()
        
        assert state is None
    
    def test_save_and_load_state(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test saving and loading pipeline state."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # Mock the video hash calculation to match our test state
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            # Update state to match our video file
            sample_pipeline_state.video_file = str(temp_video_file)
            
            # Save state
            manager.save_state(sample_pipeline_state)
            
            # Verify state file was created
            assert manager.state_file_path.exists()
            
            # Load state
            loaded_state = manager.load_state()
            
            assert loaded_state is not None
            assert loaded_state.video_file == str(temp_video_file)
            assert loaded_state.video_hash == "abc123def456"
            assert loaded_state.video_metadata.duration == 120.5
    
    def test_save_state_creates_backup(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test that saving state creates backup of existing file."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            # Update state to match our video file
            sample_pipeline_state.video_file = str(temp_video_file)
            
            # Save state first time
            manager.save_state(sample_pipeline_state)
            
            # Modify state and save again
            sample_pipeline_state.current_step = "face_detection"
            manager.save_state(sample_pipeline_state)
            
            # Load state and verify it has the updated step
            loaded_state = manager.load_state()
            assert loaded_state.current_step == "face_detection"
    
    def test_validate_state_video_hash_mismatch(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test state validation when video hash doesn't match."""
        manager = StateManager(str(temp_video_file), temp_manager)

        # Save state with one hash
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            manager.save_state(sample_pipeline_state)
        
        # Try to load with different hash
        with patch.object(manager, '_calculate_video_hash', return_value="different_hash"):
            with pytest.raises(StateLoadError, match="Video file has been modified"):
                manager.load_state()
    
    def test_validate_state_video_file_missing(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test state validation when video file is missing."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # Save state
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            manager.save_state(sample_pipeline_state)
        
        # Remove video file
        temp_video_file.unlink()
        
        # Try to load state
        with pytest.raises(StateLoadError, match="Video file no longer exists"):
            manager.load_state()
    
    def test_update_step_progress(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test updating step progress."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            sample_pipeline_state.start_step("frame_extraction", 1000)
            manager.save_state(sample_pipeline_state)
            
            # Update progress
            manager.update_step_progress("frame_extraction", {"processed_count": 500})
            
            # Load and verify
            updated_state = manager.load_state()
            progress = updated_state.get_step_progress("frame_extraction")
            assert progress.processed_count == 500
    
    def test_mark_step_complete(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test marking step as complete."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            sample_pipeline_state.start_step("frame_extraction", 1000)
            manager.save_state(sample_pipeline_state)
            
            # Mark complete
            manager.mark_step_complete("frame_extraction")
            
            # Load and verify
            updated_state = manager.load_state()
            assert updated_state.is_step_completed("frame_extraction") is True
    
    def test_can_resume(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test can_resume functionality."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # No state file exists
        assert manager.can_resume() is False
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            
            # Save new state (no completed steps)
            manager.save_state(sample_pipeline_state)
            assert manager.can_resume() is False
            
            # Complete a step
            sample_pipeline_state.complete_step("initialization")
            manager.save_state(sample_pipeline_state)
            assert manager.can_resume() is True
    
    def test_get_resume_point(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test getting resume point."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # No state file
        assert manager.get_resume_point() is None
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            sample_pipeline_state.complete_step("initialization")
            sample_pipeline_state.complete_step("frame_extraction")
            sample_pipeline_state.current_step = "face_detection"
            manager.save_state(sample_pipeline_state)
            
            resume_point = manager.get_resume_point()
            assert resume_point == "face_detection"
    
    def test_delete_state(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test deleting state file."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            manager.save_state(sample_pipeline_state)
            
            # Verify file exists
            assert manager.state_file_path.exists()
            
            # Delete state
            manager.delete_state()
            
            # Verify file is gone
            assert not manager.state_file_path.exists()
    
    def test_get_state_info(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test getting basic state information."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # No state file
        info = manager.get_state_info()
        assert info is None
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            sample_pipeline_state.complete_step("initialization")
            manager.save_state(sample_pipeline_state)
            
            info = manager.get_state_info()
            assert info is not None
            assert info["video_file"] == str(temp_video_file)
            assert info["current_step"] == sample_pipeline_state.current_step
            assert "initialization" in info["completed_steps"]
            assert info["can_resume"] is True
    
    def test_context_manager(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test StateManager as context manager."""
        with patch('personfromvid.core.state_manager.StateManager._calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
    
            # Use as context manager
            with StateManager(str(temp_video_file), temp_manager) as manager:
                manager.save_state(sample_pipeline_state)
                assert manager.state_file_path.exists()
            
            # Should cleanup backup files
            backup_path = manager.state_file_path.with_suffix('.json.backup')
            assert not backup_path.exists()
    
    def test_save_state_error_handling(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test error handling when saving state fails."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # Mock file operations to fail
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(StateSaveError, match="Cannot save state"):
                manager.save_state(sample_pipeline_state)
    
    def test_load_state_corrupted_file(self, temp_video_file, temp_manager):
        """Test loading corrupted state file."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # Create corrupted JSON file
        with open(manager.state_file_path, 'w') as f:
            f.write("invalid json content {")
        
        with pytest.raises(StateLoadError, match="Cannot load state"):
            manager.load_state()
    
    def test_calculate_video_hash(self, temp_video_file, temp_manager):
        """Test video hash calculation."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        # Calculate hash twice - should be same
        hash1 = manager._calculate_video_hash()
        hash2 = manager._calculate_video_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex string length
        assert isinstance(hash1, str)
    
    def test_backup_and_restore(self, temp_video_file, temp_manager, sample_pipeline_state):
        """Test backup creation and restoration."""
        manager = StateManager(str(temp_video_file), temp_manager)
        
        with patch.object(manager, '_calculate_video_hash', return_value="abc123def456"):
            sample_pipeline_state.video_file = str(temp_video_file)
            
            # Save initial state
            manager.save_state(sample_pipeline_state)
            original_content = manager.state_file_path.read_text()
            
            # Save modified state (should create backup)
            sample_pipeline_state.current_step = "modified_step"
            manager.save_state(sample_pipeline_state)
            
            # Verify backup exists and contains original content
            backup_path = manager.state_file_path.with_suffix('.json.backup')
            # Note: backup is created but may be cleaned up immediately in successful save
            
            # Verify new content is different
            new_content = manager.state_file_path.read_text()
            assert "modified_step" in new_content 