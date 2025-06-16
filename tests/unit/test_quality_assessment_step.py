"""Unit tests for QualityAssessmentStep."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from personfromvid.core.steps.quality_assessment import QualityAssessmentStep
from personfromvid.data.frame_data import FrameData, SourceInfo, ImageProperties, SelectionInfo
from personfromvid.data.detection_results import QualityMetrics, FaceDetection, PoseDetection


class TestQualityAssessmentStep:
    """Test cases for QualityAssessmentStep class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock pipeline with necessary attributes
        self.mock_pipeline = Mock()
        self.mock_pipeline.state = Mock()
        self.mock_pipeline.formatter = None
        self.mock_pipeline.logger = Mock()
        self.mock_pipeline.config = Mock()
        
        # Initialize the step
        self.step = QualityAssessmentStep(self.mock_pipeline)

    def _create_test_frame(self, frame_id: str, overall_quality: float = None, has_faces: bool = True, has_poses: bool = True) -> FrameData:
        """Helper to create a test FrameData object."""
        frame = FrameData(
            frame_id=frame_id,
            file_path=Path(f"/test/{frame_id}.jpg"),
            source_info=SourceInfo(
                video_timestamp=1.0,
                extraction_method='test',
                original_frame_number=30,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=640,
                height=480,
                channels=3,
                file_size_bytes=100000,
                format='JPEG'
            )
        )
        
        # Add face and pose detections if requested
        if has_faces:
            frame.face_detections = [FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9)]
        
        if has_poses:
            frame.pose_detections = [PoseDetection(
                bbox=(80, 50, 220, 400), 
                confidence=0.92, 
                keypoints={'nose': (150, 120, 0.9)}, 
                pose_classifications=[("standing", 0.92)]
            )]
        
        # Add quality metrics if overall_quality is specified
        if overall_quality is not None:
            frame.quality_metrics = QualityMetrics(
                laplacian_variance=1000.0,
                sobel_variance=800.0,
                brightness_score=128.0,
                contrast_score=50.0,
                overall_quality=overall_quality,
                quality_issues=[],
                usable=True
            )
        
        return frame

    def test_quality_ranking_is_correct(self):
        """Test that quality_rank is correctly populated on frames with varying quality scores."""
        # Create test frames with different quality scores
        frames = [
            self._create_test_frame("frame_001", overall_quality=0.95),  # Should get rank 1 (highest)
            self._create_test_frame("frame_002", overall_quality=0.45),  # Should get rank 4 (lowest assessed)
            self._create_test_frame("frame_003", overall_quality=0.80),  # Should get rank 2
            self._create_test_frame("frame_004", overall_quality=0.60),  # Should get rank 3
            self._create_test_frame("frame_005", has_faces=False),       # Should get rank None (not assessed)
            self._create_test_frame("frame_006", has_poses=False),       # Should get rank None (not assessed)
        ]
        
        # Call the ranking method directly
        self.step._rank_frames_by_quality(frames)
        
        # Verify ranking is correct
        assert frames[0].selections.quality_rank == 1    # 0.95 quality
        assert frames[1].selections.quality_rank == 4    # 0.45 quality  
        assert frames[2].selections.quality_rank == 2    # 0.80 quality
        assert frames[3].selections.quality_rank == 3    # 0.60 quality
        assert frames[4].selections.quality_rank is None # No faces
        assert frames[5].selections.quality_rank is None # No poses

    def test_quality_ranking_handles_ties(self):
        """Test that tied quality scores are handled gracefully with stable sort."""
        # Create frames with identical quality scores
        frames = [
            self._create_test_frame("frame_001", overall_quality=0.80),  # Should get rank 1 (first in list)
            self._create_test_frame("frame_002", overall_quality=0.90),  # Should get rank 1 (highest)
            self._create_test_frame("frame_003", overall_quality=0.80),  # Should get rank 2 (second with 0.80)
            self._create_test_frame("frame_004", overall_quality=0.80),  # Should get rank 3 (third with 0.80)
        ]
        
        # Call the ranking method
        self.step._rank_frames_by_quality(frames)
        
        # Verify the highest score gets rank 1
        assert frames[1].selections.quality_rank == 1    # 0.90 quality (highest)
        
        # Verify tied scores get sequential ranks in original order (stable sort)
        tied_ranks = [frame.selections.quality_rank for frame in [frames[0], frames[2], frames[3]]]
        assert tied_ranks == [2, 3, 4]  # Sequential ranks for tied scores

    def test_quality_ranking_empty_list(self):
        """Test that empty frame list is handled without errors."""
        # Call with empty list - should not raise any exceptions
        self.step._rank_frames_by_quality([])
        
        # Test passes if no exception is raised

    def test_quality_ranking_no_assessed_frames(self):
        """Test ranking when no frames have been quality assessed."""
        # Create frames that won't be quality assessed (no faces or poses)
        frames = [
            self._create_test_frame("frame_001", has_faces=False),
            self._create_test_frame("frame_002", has_poses=False),
            self._create_test_frame("frame_003", has_faces=False, has_poses=False),
        ]
        
        # Call the ranking method
        self.step._rank_frames_by_quality(frames)
        
        # All frames should have None rank
        for frame in frames:
            assert frame.selections.quality_rank is None

    def test_quality_ranking_mixed_assessed_unassessed(self):
        """Test ranking with mix of assessed and unassessed frames."""
        frames = [
            self._create_test_frame("frame_001", overall_quality=0.75),  # Should get rank 2
            self._create_test_frame("frame_002", has_faces=False),       # Should get rank None
            self._create_test_frame("frame_003", overall_quality=0.90),  # Should get rank 1
            self._create_test_frame("frame_004", has_poses=False),       # Should get rank None
            self._create_test_frame("frame_005", overall_quality=0.40),  # Should get rank 3
        ]
        
        # Call the ranking method
        self.step._rank_frames_by_quality(frames)
        
        # Verify only assessed frames get ranks
        assert frames[0].selections.quality_rank == 2    # 0.75 quality
        assert frames[1].selections.quality_rank is None # No faces
        assert frames[2].selections.quality_rank == 1    # 0.90 quality (highest)
        assert frames[3].selections.quality_rank is None # No poses  
        assert frames[4].selections.quality_rank == 3    # 0.40 quality (lowest)

    def test_quality_ranking_preserves_selection_info(self):
        """Test that ranking doesn't interfere with existing SelectionInfo data."""
        # Create a frame with existing selection data
        frame = self._create_test_frame("frame_001", overall_quality=0.85)
        frame.selections.selected_for_poses = ["standing"]
        frame.selections.final_output = True
        frame.selections.output_files = ["test.jpg"]
        
        # Call ranking
        self.step._rank_frames_by_quality([frame])
        
        # Verify existing data is preserved and rank is added
        assert frame.selections.quality_rank == 1
        assert frame.selections.selected_for_poses == ["standing"]
        assert frame.selections.final_output is True
        assert frame.selections.output_files == ["test.jpg"] 