"""Unit tests for the ProgressManager class."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from personfromvid.utils.progress import ProgressManager, ProgressStats, create_progress_manager
from personfromvid.data import PipelineState, StepProgress, VideoMetadata
from personfromvid.core.pipeline import ProcessingResult
from rich.console import Console


@pytest.fixture
def mock_console():
    """Create a mock console for testing."""
    return Mock(spec=Console)


@pytest.fixture
def sample_pipeline_state():
    """Create sample pipeline state for testing."""
    video_metadata = VideoMetadata(
        duration=120.0, fps=30.0, width=1920, height=1080,
        codec="h264", total_frames=3600, file_size_bytes=1000000, format="mp4"
    )
    
    state = PipelineState(
        video_file="/test/video.mp4",
        video_hash="test_hash_123",
        video_metadata=video_metadata,
        model_versions={},
        created_at=datetime.now(),
        last_updated=datetime.now()
    )
    
    # Add some progress
    state.start_step("frame_extraction", 100)
    state.update_step_progress("frame_extraction", 50)
    state.get_step_progress("frame_extraction").set_data("extracted_count", 50)
    
    return state


@pytest.fixture  
def sample_processing_result():
    """Create sample processing result for testing."""
    return ProcessingResult(
        success=True,
        total_frames_extracted=100,
        faces_found=25,
        poses_found={"standing": 10, "sitting": 8, "squatting": 5},
        head_angles_found={"front": 12, "profile_left": 6, "profile_right": 4},
        output_files=["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"],
        processing_time_seconds=45.5
    )


class TestProgressStats:
    """Test ProgressStats data class."""
    
    def test_progress_stats_creation(self):
        """Test creating ProgressStats instance."""
        stats = ProgressStats()
        assert stats.items_per_second == 0.0
        assert stats.eta_seconds is None
        assert stats.elapsed_seconds == 0.0
        assert stats.peak_rate == 0.0
    
    def test_eta_formatted_property(self):
        """Test ETA formatting."""
        stats = ProgressStats()
        
        # Test unknown ETA
        assert stats.eta_formatted == "Unknown"
        
        # Test seconds
        stats.eta_seconds = 45.0
        assert stats.eta_formatted == "45s"
        
        # Test minutes and seconds
        stats.eta_seconds = 125.0  # 2m 5s
        assert stats.eta_formatted == "2m 5s"
        
        # Test exactly 1 minute
        stats.eta_seconds = 60.0
        assert stats.eta_formatted == "1m 0s"


class TestProgressManager:
    """Test ProgressManager class."""
    
    def test_progress_manager_initialization(self, mock_console):
        """Test progress manager initialization."""
        manager = ProgressManager(mock_console)
        
        assert manager.console == mock_console
        assert manager.main_progress is None
        assert manager.step_progress is None
        assert manager.current_task is None
        assert manager.live_display is None
        assert manager.is_active is False
        assert manager.current_step == "initialization"
        assert manager.pipeline_state is None
    
    def test_progress_manager_initialization_default_console(self):
        """Test progress manager initialization with default console."""
        manager = ProgressManager()
        
        assert manager.console is not None
        assert isinstance(manager.console, Console)
    
    @patch('personfromvid.utils.progress.Live')
    @patch('personfromvid.utils.progress.Progress')
    @patch('personfromvid.utils.progress.Layout')
    def test_start_pipeline_progress(self, mock_layout, mock_progress, mock_live, 
                                   mock_console, sample_pipeline_state):
        """Test starting pipeline progress display."""
        manager = ProgressManager(mock_console)
        
        # Mock progress instances
        mock_main_progress = Mock()
        mock_step_progress = Mock()
        mock_progress.side_effect = [mock_main_progress, mock_step_progress]
        
        # Mock layout
        mock_layout_instance = Mock()
        mock_layout.return_value = mock_layout_instance
        
        # Mock live display
        mock_live_instance = Mock()
        mock_live.return_value = mock_live_instance
        
        # Start progress
        manager.start_pipeline_progress(sample_pipeline_state)
        
        # Verify setup
        assert manager.pipeline_state == sample_pipeline_state
        assert manager.is_active is True
        assert manager.overall_start_time is not None
        assert manager.main_progress == mock_main_progress
        assert manager.step_progress == mock_step_progress
        assert manager.live_display == mock_live_instance
        
        # Verify layout creation
        mock_layout_instance.split_column.assert_called_once()
        
        # Verify live display creation
        mock_live.assert_called_once()
    
    def test_update_pipeline_state_inactive(self, mock_console, sample_pipeline_state):
        """Test updating pipeline state when inactive."""
        manager = ProgressManager(mock_console)
        
        # Should not crash when inactive
        manager.update_pipeline_state(sample_pipeline_state)
        
        assert manager.pipeline_state == sample_pipeline_state
    
    @patch('personfromvid.utils.progress.Live')
    @patch('personfromvid.utils.progress.Progress')
    @patch('personfromvid.utils.progress.Layout')
    def test_start_step_progress(self, mock_layout, mock_progress, mock_live, 
                               mock_console):
        """Test starting step progress tracking."""
        manager = ProgressManager(mock_console)
        
        # Mock step progress
        mock_step_progress = Mock()
        mock_task_id = "test_task"
        mock_step_progress.add_task.return_value = mock_task_id
        mock_step_progress.tasks = []
        manager.step_progress = mock_step_progress
        
        # Start step progress
        step_name = "frame_extraction"
        total_items = 100
        description = "Extracting Frames"
        
        manager.start_step_progress(step_name, total_items, description)
        
        # Verify state
        assert manager.current_step == step_name
        assert step_name in manager.step_start_times
        assert step_name in manager.step_stats
        assert manager.current_task == mock_task_id
        
        # Verify task creation
        mock_step_progress.add_task.assert_called_once_with(
            description,
            total=total_items,
            rate="0.0/s"
        )
    
    def test_update_step_progress_inactive(self, mock_console):
        """Test updating step progress when inactive."""
        manager = ProgressManager(mock_console)
        
        # Should not crash when inactive
        manager.update_step_progress("test_step", 50)
    
    def test_update_step_progress_with_rate_calculation(self, mock_console):
        """Test step progress update with rate calculation."""
        manager = ProgressManager(mock_console)
        manager.is_active = True
        
        # Mock step progress and task with proper task list
        mock_step_progress = Mock()
        mock_task = Mock()
        mock_task.total = 100
        # Make tasks behave like a list
        mock_step_progress.tasks = [mock_task]
        manager.step_progress = mock_step_progress
        manager.current_task = "test_task"
        
        # Set start time in the past
        step_name = "test_step"
        manager.step_start_times[step_name] = time.time() - 10.0  # 10 seconds ago
        manager.step_stats[step_name] = ProgressStats()
        
        # Update progress
        processed_count = 50
        manager.update_step_progress(step_name, processed_count)
        
        # Verify rate calculation
        stats = manager.step_stats[step_name]
        assert stats.items_per_second > 0  # Should have calculated a rate
        assert stats.elapsed_seconds > 0
        assert stats.eta_seconds is not None
        
        # Verify progress update
        mock_step_progress.update.assert_called_once()
    
    def test_complete_step_progress(self, mock_console):
        """Test completing step progress."""
        manager = ProgressManager(mock_console)
        
        # Mock step progress and task
        mock_step_progress = Mock()
        mock_task = Mock()
        mock_task.id = "test_task"
        mock_task.total = 100
        mock_step_progress.tasks = [mock_task]
        manager.step_progress = mock_step_progress
        manager.current_task = "test_task"
        
        # Set start time
        step_name = "test_step"
        manager.step_start_times[step_name] = time.time() - 5.0
        manager.step_stats[step_name] = ProgressStats()
        
        # Complete step
        manager.complete_step_progress(step_name)
        
        # Verify completion
        assert manager.step_stats[step_name].elapsed_seconds > 0
        mock_step_progress.update.assert_called_once_with(
            "test_task",
            completed=100,
            rate="Complete"
        )
    
    def test_add_statistics_panel(self, mock_console, sample_pipeline_state):
        """Test adding statistics panel."""
        manager = ProgressManager(mock_console)
        manager.pipeline_state = sample_pipeline_state
        
        # Add some step stats
        manager.step_stats["frame_extraction"] = ProgressStats(
            items_per_second=5.2,
            eta_seconds=30.0,
            peak_rate=6.1
        )
        
        # Add statistics
        custom_stats = {"Custom Metric": "Custom Value"}
        manager.add_statistics_panel(custom_stats)
        
        # Verify table creation
        assert manager.stats_table is not None
        # Note: We can't easily test Rich table contents, but we can verify it was created
    
    def test_display_final_summary(self, mock_console, sample_processing_result):
        """Test displaying final summary."""
        manager = ProgressManager(mock_console)
        
        # Display summary
        manager.display_final_summary(sample_processing_result)
        
        # Verify console was used to print
        assert mock_console.print.call_count >= 2  # At least summary table and files
    
    def test_stop_progress(self, mock_console):
        """Test stopping progress display."""
        manager = ProgressManager(mock_console)
        
        # Mock live display
        mock_live_display = Mock()
        manager.live_display = mock_live_display
        manager.is_active = True
        
        # Stop progress
        manager.stop_progress()
        
        # Verify cleanup
        assert manager.is_active is False
        assert manager.current_task is None
        assert manager.main_progress is None
        assert manager.step_progress is None
        assert manager.live_display is None
        mock_live_display.stop.assert_called_once()
    
    def test_context_manager(self, mock_console):
        """Test context manager functionality."""
        manager = ProgressManager(mock_console)
        
        # Mock live display
        mock_live_display = Mock()
        mock_live_display.is_started = False
        manager.live_display = mock_live_display
        
        # Test context manager
        with manager as ctx_manager:
            assert ctx_manager == manager
            mock_live_display.start.assert_called_once()
        
        # Verify exit cleanup
        mock_live_display.stop.assert_called_once()
    
    def test_context_manager_already_started(self, mock_console):
        """Test context manager when display already started."""
        manager = ProgressManager(mock_console)
        
        # Mock live display that's already started
        mock_live_display = Mock()
        mock_live_display.is_started = True
        manager.live_display = mock_live_display
        
        # Test context manager
        with manager:
            # Should not call start again
            mock_live_display.start.assert_not_called()
    
    def test_update_step_progress_with_extra_info(self, mock_console, sample_pipeline_state):
        """Test updating step progress with extra information."""
        manager = ProgressManager(mock_console)
        manager.pipeline_state = sample_pipeline_state
        manager.is_active = True
        
        # Mock step progress with proper tasks behavior
        mock_step_progress = Mock()
        mock_task = Mock()
        mock_task.total = 100
        mock_step_progress.tasks = [mock_task]  # Make it behave like a list
        manager.step_progress = mock_step_progress
        manager.current_task = "test_task"
        
        # Set up step stats
        step_name = "frame_extraction"
        manager.step_start_times[step_name] = time.time() - 1.0
        manager.step_stats[step_name] = ProgressStats()
        
        # Update with extra info
        extra_info = {"faces_found": 10, "extracted_count": 50}
        manager.update_step_progress(step_name, 50, extra_info)
        
        # Verify extra info was added to pipeline state
        step_progress = sample_pipeline_state.get_step_progress(step_name)
        assert step_progress.get_data("faces_found") == 10
        assert step_progress.get_data("extracted_count") == 50
    
    def test_update_statistics_panel_with_step_data(self, mock_console, sample_pipeline_state):
        """Test updating statistics panel with step-specific data."""
        manager = ProgressManager(mock_console)
        manager.pipeline_state = sample_pipeline_state
        manager.overall_start_time = time.time() - 30.0  # 30 seconds ago
        
        # Add step data
        step_progress = sample_pipeline_state.get_step_progress("frame_extraction")
        step_progress.set_data("extracted_count", 75)
        step_progress.set_data("faces_found", 20)
        step_progress.set_data("poses_found", {"standing": 5, "sitting": 3})
        
        # Update statistics panel
        manager._update_statistics_panel()
        
        # Verify table was created (can't easily test contents)
        assert manager.stats_table is not None


class TestProgressUtilities:
    """Test utility functions."""
    
    def test_create_progress_manager_default(self):
        """Test creating progress manager with default console."""
        manager = create_progress_manager()
        
        assert isinstance(manager, ProgressManager)
        assert isinstance(manager.console, Console)
    
    def test_create_progress_manager_custom_console(self, mock_console):
        """Test creating progress manager with custom console."""
        manager = create_progress_manager(mock_console)
        
        assert isinstance(manager, ProgressManager)
        assert manager.console == mock_console


class TestProgressIntegration:
    """Test progress manager integration scenarios."""
    
    @patch('personfromvid.utils.progress.Live')
    @patch('personfromvid.utils.progress.Progress')
    @patch('personfromvid.utils.progress.Layout')
    def test_full_pipeline_simulation(self, mock_layout, mock_progress, mock_live,
                                    mock_console, sample_pipeline_state):
        """Test simulating a full pipeline run with progress updates."""
        manager = ProgressManager(mock_console)
        
        # Mock components
        mock_main_progress = Mock()
        mock_main_progress.tasks = []  # Make tasks behave like a list
        mock_step_progress = Mock()
        mock_step_progress.tasks = []  # Make tasks behave like a list
        mock_progress.side_effect = [mock_main_progress, mock_step_progress]
        
        mock_layout_instance = Mock()
        # Make layout subscriptable with proper mock behavior
        mock_main_layout = Mock()
        mock_step_layout = Mock() 
        mock_stats_layout = Mock()
        mock_layout_instance.__getitem__ = Mock(side_effect=lambda key: {
            "main": mock_main_layout,
            "step": mock_step_layout, 
            "stats": mock_stats_layout
        }[key])
        mock_layout.return_value = mock_layout_instance
        
        mock_live_instance = Mock()
        mock_live_instance.renderable = mock_layout_instance  # Set renderable property
        mock_live.return_value = mock_live_instance
        
        mock_task_id = "test_task"
        mock_step_progress.add_task.return_value = mock_task_id
        
        # Simulate pipeline progression
        manager.start_pipeline_progress(sample_pipeline_state)
        
        # Simulate frame extraction step
        manager.start_step_progress("frame_extraction", 100, "Extracting Frames")
        
        # Simulate progress updates
        for i in range(0, 101, 25):
            manager.update_step_progress("frame_extraction", i)
            sample_pipeline_state.update_step_progress("frame_extraction", i)
            manager.update_pipeline_state(sample_pipeline_state)
        
        # Complete step
        manager.complete_step_progress("frame_extraction")
        sample_pipeline_state.complete_step("frame_extraction")
        
        # Update final state
        manager.update_pipeline_state(sample_pipeline_state)
        
        # Stop progress
        manager.stop_progress()
        
        # Verify the simulation completed without errors
        assert not manager.is_active
        assert mock_step_progress.add_task.called
        assert mock_step_progress.update.called
    
    def test_progress_manager_with_interruption(self, mock_console, sample_pipeline_state):
        """Test progress manager behavior during interruption."""
        manager = ProgressManager(mock_console)
        
        # Start progress
        manager.pipeline_state = sample_pipeline_state
        manager.is_active = True
        
        # Simulate interruption by stopping
        manager.stop_progress()
        
        # Verify clean shutdown
        assert not manager.is_active
        assert manager.live_display is None
    
    def test_resume_from_partial_progress(self, mock_console, sample_pipeline_state):
        """Test resuming progress from a partially completed state."""
        manager = ProgressManager(mock_console)
        
        # Set up partial completion state
        sample_pipeline_state.complete_step("initialization")
        sample_pipeline_state.start_step("frame_extraction", 100)
        sample_pipeline_state.update_step_progress("frame_extraction", 60)
        
        # Update manager with resumed state
        manager.pipeline_state = sample_pipeline_state
        manager.is_active = True
        
        # Should handle resumed state gracefully
        manager._update_statistics_panel()
        assert manager.stats_table is not None 