"""Tests for PersonBuildingStep pipeline step."""

from unittest.mock import Mock, patch

import pytest

from personfromvid.core.steps.person_building import PersonBuildingStep
from personfromvid.data.detection_results import FaceDetection, PoseDetection
from personfromvid.data.frame_data import FrameData
from personfromvid.data.person import Person


class TestPersonBuildingStep:
    """Test cases for PersonBuildingStep."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock ProcessingPipeline."""
        pipeline = Mock()
        pipeline.config = Mock()
        pipeline.state = Mock()
        pipeline.logger = Mock()
        pipeline.formatter = None
        pipeline._interrupted = False
        return pipeline

    @pytest.fixture
    def person_building_step(self, mock_pipeline):
        """Create PersonBuildingStep instance."""
        return PersonBuildingStep(mock_pipeline)

    def test_init(self, person_building_step, mock_pipeline):
        """Test PersonBuildingStep initialization."""
        assert person_building_step.pipeline == mock_pipeline
        assert person_building_step.config == mock_pipeline.config
        assert person_building_step.state == mock_pipeline.state
        assert person_building_step.logger == mock_pipeline.logger
        assert person_building_step.formatter == mock_pipeline.formatter

    def test_step_name(self, person_building_step):
        """Test step_name property."""
        assert person_building_step.step_name == "person_building"

    def test_execute_empty_frames(self, person_building_step):
        """Test execute method with no frames containing detections."""
        # Setup
        person_building_step.state.frames = []
        mock_progress = Mock()
        person_building_step.state.get_step_progress.return_value = mock_progress

        # Execute
        person_building_step.execute()

        # Verify
        person_building_step.state.start_step.assert_called_once_with("person_building")
        mock_progress.start.assert_called_once_with(0)
        person_building_step.state.update_step_progress.assert_called_once_with("person_building", 0)

    def test_execute_basic_call(self, person_building_step):
        """Test basic execute method call with mock frames."""
        # Setup mock frames with detections
        frame1 = Mock(spec=FrameData)
        frame1.has_faces.return_value = True
        frame1.has_poses.return_value = True
        frame1.face_detections = [Mock(spec=FaceDetection)]
        frame1.pose_detections = [Mock(spec=PoseDetection)]
        frame1.head_poses = []
        frame1.frame_number = 1

        person_building_step.state.frames = [frame1]
        mock_progress = Mock()
        person_building_step.state.get_step_progress.return_value = mock_progress

        # Mock PersonBuilder
        with patch('personfromvid.core.steps.person_building.PersonBuilder') as mock_person_builder_class:
            mock_person_builder = Mock()
            mock_person_builder_class.return_value = mock_person_builder
            mock_person_builder.build_persons.return_value = [Mock(spec=Person)]

            # Execute
            person_building_step.execute()

            # Verify
            person_building_step.state.start_step.assert_called_once_with("person_building")
            mock_progress.start.assert_called_once_with(1)
            mock_person_builder.build_persons.assert_called_once()
            assert hasattr(frame1, 'persons')

    def test_execute_error_handling(self, person_building_step):
        """Test execute method error handling."""
        # Setup - cause error in get_step_progress to trigger outer exception handling
        frame1 = Mock(spec=FrameData)
        frame1.has_faces.return_value = True
        frame1.has_poses.return_value = False

        person_building_step.state.frames = [frame1]
        person_building_step.state.get_step_progress.side_effect = Exception("Test error")

        # Execute and verify exception is raised
        with pytest.raises(Exception, match="Test error"):
            person_building_step.execute()

        # Verify error handling
        person_building_step.state.fail_step.assert_called_once_with("person_building", "Test error")

    def test_execute_frame_error_handling(self, person_building_step):
        """Test individual frame error handling within execute method."""
        # Setup - cause error in individual frame processing
        frame1 = Mock(spec=FrameData)
        frame1.has_faces.return_value = True
        frame1.has_poses.return_value = True
        frame1.face_detections = [Mock(spec=FaceDetection)]
        frame1.pose_detections = [Mock(spec=PoseDetection)]
        frame1.head_poses = []
        frame1.frame_number = 1

        person_building_step.state.frames = [frame1]
        mock_progress = Mock()
        person_building_step.state.get_step_progress.return_value = mock_progress

        # Mock PersonBuilder to raise exception for this frame
        with patch('personfromvid.core.steps.person_building.PersonBuilder') as mock_person_builder_class:
            mock_person_builder = Mock()
            mock_person_builder_class.return_value = mock_person_builder
            mock_person_builder.build_persons.side_effect = Exception("Frame processing error")

            # Execute - should NOT raise exception due to error handling
            person_building_step.execute()

            # Verify frame gets empty persons list on error
            assert frame1.persons == []

            # Verify error was logged
            person_building_step.logger.error.assert_called_with(
                "Failed to build persons for frame 1: Frame processing error"
            )

    def test_get_frames_with_detections_faces_only(self, person_building_step):
        """Test _get_frames_with_detections with face detections only."""
        # Setup frames
        frame_with_faces = Mock(spec=FrameData)
        frame_with_faces.has_faces.return_value = True
        frame_with_faces.has_poses.return_value = False

        frame_empty = Mock(spec=FrameData)
        frame_empty.has_faces.return_value = False
        frame_empty.has_poses.return_value = False

        person_building_step.state.frames = [frame_with_faces, frame_empty]

        # Execute
        result = person_building_step._get_frames_with_detections()

        # Verify
        assert len(result) == 1
        assert result[0] == frame_with_faces

    def test_get_frames_with_detections_poses_only(self, person_building_step):
        """Test _get_frames_with_detections with pose detections only."""
        # Setup frames
        frame_with_poses = Mock(spec=FrameData)
        frame_with_poses.has_faces.return_value = False
        frame_with_poses.has_poses.return_value = True

        frame_empty = Mock(spec=FrameData)
        frame_empty.has_faces.return_value = False
        frame_empty.has_poses.return_value = False

        person_building_step.state.frames = [frame_with_poses, frame_empty]

        # Execute
        result = person_building_step._get_frames_with_detections()

        # Verify
        assert len(result) == 1
        assert result[0] == frame_with_poses

    def test_get_frames_with_detections_both(self, person_building_step):
        """Test _get_frames_with_detections with both face and pose detections."""
        # Setup frames
        frame_with_both = Mock(spec=FrameData)
        frame_with_both.has_faces.return_value = True
        frame_with_both.has_poses.return_value = True

        frame_empty = Mock(spec=FrameData)
        frame_empty.has_faces.return_value = False
        frame_empty.has_poses.return_value = False

        person_building_step.state.frames = [frame_with_both, frame_empty]

        # Execute
        result = person_building_step._get_frames_with_detections()

        # Verify
        assert len(result) == 1
        assert result[0] == frame_with_both

    def test_get_frames_with_detections_empty(self, person_building_step):
        """Test _get_frames_with_detections with no detections."""
        # Setup frames with no detections
        frame_empty = Mock(spec=FrameData)
        frame_empty.has_faces.return_value = False
        frame_empty.has_poses.return_value = False

        person_building_step.state.frames = [frame_empty]

        # Execute
        result = person_building_step._get_frames_with_detections()

        # Verify
        assert len(result) == 0
