"""Unit tests for frame selection logic.

Tests for the FrameSelector class, focusing on the new enhanced transparency
features and separation of scoring from selection logic.
"""

from pathlib import Path
from unittest.mock import Mock

from personfromvid.analysis.frame_selector import (
    FrameSelector,
    SelectionCriteria,
)
from personfromvid.data.detection_results import (
    FaceDetection,
    HeadPoseResult,
    PoseDetection,
    QualityMetrics,
)
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo


def create_test_frame(
    frame_id: str,
    quality_score: float = 0.8,
    pose_confidence: float = 0.9,
    face_confidence: float = 0.85,
    head_pose_confidence: float = 0.7,
    timestamp: float = 0.0
) -> FrameData:
    """Create a test frame with specified quality and confidence scores."""
    source_info = SourceInfo(
        video_timestamp=timestamp,
        extraction_method="test",
        original_frame_number=0,
        video_fps=30.0
    )

    image_properties = ImageProperties(
        width=1920,
        height=1080,
        channels=3,
        file_size_bytes=100000,
        format="JPG"
    )

    quality_metrics = QualityMetrics(
        laplacian_variance=100.0,
        sobel_variance=50.0,
        brightness_score=0.8,
        contrast_score=0.7,
        overall_quality=quality_score,
        quality_issues=[],
        usable=True
    )

    # Create pose detection
    pose_detection = PoseDetection(
        bbox=(100, 100, 300, 400),
        confidence=pose_confidence,
        keypoints={},
        pose_classifications=[("standing", 0.9)]
    )

    # Create face detection
    face_detection = FaceDetection(
        bbox=(150, 120, 250, 220),
        confidence=face_confidence,
        landmarks=None
    )

    # Create head pose
    head_pose = HeadPoseResult(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        confidence=head_pose_confidence,
        direction="front",
        face_id=0
    )

    frame = FrameData(
        frame_id=frame_id,
        file_path=Path(f"/test/{frame_id}.jpg"),
        source_info=source_info,
        image_properties=image_properties,
        quality_metrics=quality_metrics,
        face_detections=[face_detection],
        pose_detections=[pose_detection],
        head_poses=[head_pose]
    )

    return frame


class TestFrameSelectorScoringAndRanking:
    """Test the new scoring and ranking method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.criteria = SelectionCriteria(
            min_frames_per_category=2,
            max_frames_per_category=5,
            min_quality_threshold=0.5,
            face_size_weight=0.3,
            quality_weight=0.6,
            diversity_threshold=0.8,
            temporal_diversity_threshold=2.0
        )
        self.selector = FrameSelector(self.criteria)

    def test_score_and_rank_candidates_for_category_populates_transparency_fields(self):
        """Test that the new method populates all transparency fields correctly."""
        # Create test frames with different quality scores
        frames = [
            create_test_frame("frame1", quality_score=0.9, pose_confidence=0.8),
            create_test_frame("frame2", quality_score=0.7, pose_confidence=0.9),
            create_test_frame("frame3", quality_score=0.8, pose_confidence=0.7)
        ]

        # Call the new method
        scored_frames = self.selector._score_and_rank_candidates_for_category(
            frames=frames,
            category_name="standing",
            category_type="pose",
            score_function=self.selector._calculate_pose_frame_score
        )

        # Verify return structure
        assert len(scored_frames) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in scored_frames)
        assert all(isinstance(frame, FrameData) and isinstance(score, float) for frame, score in scored_frames)

        # Verify sorting (highest score first)
        scores = [score for _, score in scored_frames]
        assert scores == sorted(scores, reverse=True)

        # Verify transparency fields are populated
        category_key = "pose_standing"
        for frame, expected_score in scored_frames:
            # Check category_scores
            assert category_key in frame.selections.category_scores
            assert frame.selections.category_scores[category_key] == expected_score

            # Check category_score_breakdowns
            assert category_key in frame.selections.category_score_breakdowns
            breakdown = frame.selections.category_score_breakdowns[category_key]
            assert isinstance(breakdown, dict)
            assert "quality" in breakdown
            assert "pose_confidence" in breakdown

            # Check category_ranks
            assert category_key in frame.selections.category_ranks
            assert isinstance(frame.selections.category_ranks[category_key], int)
            assert 1 <= frame.selections.category_ranks[category_key] <= 3

    def test_score_and_rank_candidates_ranking_logic(self):
        """Test that ranking logic assigns correct ranks based on scores."""
        # Create frames with known quality scores to test ranking
        frames = [
            create_test_frame("high_quality", quality_score=0.9, pose_confidence=0.9),   # Should be rank 1
            create_test_frame("medium_quality", quality_score=0.7, pose_confidence=0.7), # Should be rank 3
            create_test_frame("high_medium", quality_score=0.8, pose_confidence=0.8)     # Should be rank 2
        ]

        scored_frames = self.selector._score_and_rank_candidates_for_category(
            frames=frames,
            category_name="standing",
            category_type="pose",
            score_function=self.selector._calculate_pose_frame_score
        )

        category_key = "pose_standing"

        # Verify ranks are assigned correctly (1 = highest score)
        ranks_by_frame_id = {
            frame.frame_id: frame.selections.category_ranks[category_key]
            for frame, _ in scored_frames
        }

        assert ranks_by_frame_id["high_quality"] == 1
        assert ranks_by_frame_id["high_medium"] == 2
        assert ranks_by_frame_id["medium_quality"] == 3

    def test_score_and_rank_candidates_handles_ties(self):
        """Test that ranking handles tied scores gracefully."""
        # Create frames with identical scores
        frames = [
            create_test_frame("frame1", quality_score=0.8, pose_confidence=0.8),
            create_test_frame("frame2", quality_score=0.8, pose_confidence=0.8),
            create_test_frame("frame3", quality_score=0.9, pose_confidence=0.9)  # Higher score
        ]

        scored_frames = self.selector._score_and_rank_candidates_for_category(
            frames=frames,
            category_name="standing",
            category_type="pose",
            score_function=self.selector._calculate_pose_frame_score
        )

        category_key = "pose_standing"
        ranks = [frame.selections.category_ranks[category_key] for frame, _ in scored_frames]

        # Should have ranks 1, 2, 3 (stable sort preserves input order for ties)
        assert sorted(ranks) == [1, 2, 3]

        # Frame with highest score should get rank 1
        highest_score_frame = scored_frames[0][0]
        assert highest_score_frame.frame_id == "frame3"
        assert highest_score_frame.selections.category_ranks[category_key] == 1

    def test_score_and_rank_candidates_head_angle_category(self):
        """Test the method works correctly for head angle categories."""
        frames = [
            create_test_frame("frame1", quality_score=0.9, face_confidence=0.8),
            create_test_frame("frame2", quality_score=0.7, face_confidence=0.9)
        ]

        scored_frames = self.selector._score_and_rank_candidates_for_category(
            frames=frames,
            category_name="front",
            category_type="head_angle",
            score_function=self.selector._calculate_head_angle_frame_score
        )

        category_key = "head_angle_front"

        # Verify fields are populated for head angle category
        for frame, _score in scored_frames:
            assert category_key in frame.selections.category_scores
            assert category_key in frame.selections.category_score_breakdowns
            assert category_key in frame.selections.category_ranks

            # Head angle breakdown should have different fields
            breakdown = frame.selections.category_score_breakdowns[category_key]
            assert "quality" in breakdown
            assert "face_size" in breakdown
            assert "head_pose_confidence" in breakdown

    def test_score_and_rank_candidates_empty_frames_list(self):
        """Test the method handles empty frames list correctly."""
        scored_frames = self.selector._score_and_rank_candidates_for_category(
            frames=[],
            category_name="standing",
            category_type="pose",
            score_function=self.selector._calculate_pose_frame_score
        )

        assert scored_frames == []

    def test_score_and_rank_candidates_interruption_check(self):
        """Test that interruption check is called during processing."""
        frames = [create_test_frame(f"frame{i}") for i in range(10)]

        interruption_mock = Mock()

        self.selector._score_and_rank_candidates_for_category(
            frames=frames,
            category_name="standing",
            category_type="pose",
            score_function=self.selector._calculate_pose_frame_score,
            interruption_check=interruption_mock
        )

        # Should be called every 5 frames: at i=0 and i=5
        assert interruption_mock.call_count == 2

    def test_score_and_rank_candidates_fallback_for_no_breakdown(self):
        """Test fallback when score function doesn't return breakdown."""
        frames = [create_test_frame("frame1")]

        # Mock score function that doesn't return breakdown
        def simple_score_function(frame, return_breakdown=False):
            return 0.8  # Always return just score, ignore return_breakdown

        scored_frames = self.selector._score_and_rank_candidates_for_category(
            frames=frames,
            category_name="standing",
            category_type="pose",
            score_function=simple_score_function
        )

        category_key = "pose_standing"
        frame = scored_frames[0][0]

        # Should still populate fields with empty breakdown
        assert frame.selections.category_scores[category_key] == 0.8
        assert frame.selections.category_score_breakdowns[category_key] == {}
        assert frame.selections.category_ranks[category_key] == 1


class TestFrameSelectorIntegration:
    """Integration tests for the complete enhanced frame selection workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.criteria = SelectionCriteria(
            min_frames_per_category=2,
            max_frames_per_category=2,
            min_quality_threshold=0.5,
            face_size_weight=0.3,
            quality_weight=0.6,
            diversity_threshold=0.8,
            temporal_diversity_threshold=2.0
        )
        self.selector = FrameSelector(self.criteria)

    def test_complete_selection_workflow_with_transparency(self):
        """Test the complete selection workflow populates all new transparency fields."""
        # Create frames with different characteristics
        frames = [
            # Pose frames - standing category
            create_test_frame("standing_high", quality_score=0.9, pose_confidence=0.9, timestamp=0.0),
            create_test_frame("standing_medium", quality_score=0.7, pose_confidence=0.8, timestamp=5.0),
            create_test_frame("standing_low", quality_score=0.6, pose_confidence=0.7, timestamp=10.0),

            # Head angle frames - front category
            create_test_frame("front_high", quality_score=0.8, face_confidence=0.9, timestamp=15.0),
            create_test_frame("front_medium", quality_score=0.7, face_confidence=0.8, timestamp=20.0),
        ]

        # Add pose and head angle classifications to frames
        frames[0].pose_detections[0].pose_classifications = [("standing", 0.9)]
        frames[1].pose_detections[0].pose_classifications = [("standing", 0.8)]
        frames[2].pose_detections[0].pose_classifications = [("standing", 0.7)]
        frames[3].head_poses[0].direction = "front"
        frames[4].head_poses[0].direction = "front"

        # Execute the selection workflow
        summary = self.selector.select_best_frames(frames)

        # Verify summary structure
        assert summary.total_selected >= 2  # Should select at least 2 frames
        assert "standing" in summary.pose_selections
        assert "front" in summary.head_angle_selections

        # Verify transparency fields are populated for all relevant frames
        standing_frames = [f for f in frames if f.has_pose_classification("standing")]
        for frame in standing_frames:
            category_key = "pose_standing"

            # Check category_scores
            assert category_key in frame.selections.category_scores
            assert isinstance(frame.selections.category_scores[category_key], float)
            assert 0.0 <= frame.selections.category_scores[category_key] <= 1.0

            # Check category_score_breakdowns
            assert category_key in frame.selections.category_score_breakdowns
            breakdown = frame.selections.category_score_breakdowns[category_key]
            assert "quality" in breakdown
            assert "pose_confidence" in breakdown

            # Check category_ranks
            assert category_key in frame.selections.category_ranks
            assert 1 <= frame.selections.category_ranks[category_key] <= len(standing_frames)

        front_frames = [f for f in frames if f.has_head_direction("front")]
        for frame in front_frames:
            category_key = "head_angle_front"

            # Check category_scores
            assert category_key in frame.selections.category_scores
            assert isinstance(frame.selections.category_scores[category_key], float)

            # Check category_score_breakdowns
            assert category_key in frame.selections.category_score_breakdowns
            breakdown = frame.selections.category_score_breakdowns[category_key]
            assert "quality" in breakdown
            assert "face_size" in breakdown
            assert "head_pose_confidence" in breakdown

            # Check category_ranks
            assert category_key in frame.selections.category_ranks
            assert 1 <= frame.selections.category_ranks[category_key] <= len(front_frames)

        # Verify selected frames have proper metadata
        selected_frames = [f for f in frames if f.selections.final_output]
        assert len(selected_frames) > 0

        for frame in selected_frames:
            # Check primary_selection_category is set
            assert frame.selections.primary_selection_category is not None
            assert frame.selections.primary_selection_category.startswith(("pose_", "head_angle_"))

            # Check final_selection_score matches the category score
            primary_category = frame.selections.primary_selection_category
            assert frame.selections.final_selection_score is not None
            assert frame.selections.final_selection_score == frame.selections.category_scores[primary_category]

            # Check selection_rank is set
            assert frame.selections.selection_rank is not None
            assert frame.selections.selection_rank >= 1

        # Verify ranking consistency (higher scores get lower ranks)
        for category_frames in [standing_frames, front_frames]:
            if len(category_frames) < 2:
                continue

            # Get category key from first frame
            if category_frames[0].has_pose_classification("standing"):
                category_key = "pose_standing"
            else:
                category_key = "head_angle_front"

            # Sort by rank and verify scores are in descending order
            ranked_frames = sorted(category_frames, key=lambda f: f.selections.category_ranks[category_key])
            scores = [f.selections.category_scores[category_key] for f in ranked_frames]

            # Scores should be in descending order (rank 1 = highest score)
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], f"Ranking inconsistency: scores {scores}"

    def test_rejection_reasons_are_assigned(self):
        """Test that non-selected frames get proper rejection reasons."""
        # Create many frames so some will be rejected
        frames = [create_test_frame(f"frame{i}", quality_score=0.8 - i*0.1, timestamp=i*1.0) for i in range(5)]

        # All frames have standing pose
        for frame in frames:
            frame.pose_detections[0].pose_classifications = [("standing", 0.8)]

        self.selector.select_best_frames(frames)

        # Verify non-selected frames have rejection reasons
        non_selected_frames = [f for f in frames if not f.selections.final_output]
        for frame in non_selected_frames:
            assert frame.selections.rejection_reason is not None
            assert frame.selections.rejection_reason in [
                "not_selected",
                "insufficient_diversity",
                "exceeded_max_frames_per_category",
                "below_min_quality_threshold",  # Added this valid rejection reason
                "not_top_ranked"  # New granular rejection reason from Phase 3.2
            ]

    def test_priority_based_selection_with_competition_tracking(self):
        """Test that higher priority categories win in multi-category competition."""
        # Create a frame that qualifies for multiple pose categories
        frame = create_test_frame("multi_category", quality_score=0.9, pose_confidence=0.9)

        # Add multiple pose classifications (closeup has higher priority than standing)
        frame.pose_detections[0].pose_classifications = [("standing", 0.8), ("closeup", 0.9)]
        frame.closeup_detections = [
            type('CloseupDetection', (), {
                'is_closeup': True,
                'shot_type': 'closeup',
                'confidence': 0.9,
                'face_area_ratio': 0.3,
                'inter_ocular_distance': None,
                'estimated_distance': None,
                'shoulder_width_ratio': None
            })()
        ]

        frames = [frame]
        self.selector.select_best_frames(frames)

        # Frame should be scored for standing (from pose classification)
        assert "pose_standing" in frame.selections.category_scores
        assert "pose_standing" in frame.selections.category_ranks

        # Frame should also be scored for head angle (since it has "front" direction)
        assert "head_angle_front" in frame.selections.category_scores
        assert "head_angle_front" in frame.selections.category_ranks

        # Verify transparency fields have correct structure
        for category_key in frame.selections.category_scores:
            assert category_key in frame.selections.category_score_breakdowns
            assert category_key in frame.selections.category_ranks
            assert isinstance(frame.selections.category_scores[category_key], float)
            assert isinstance(frame.selections.category_ranks[category_key], int)

        # Note: In Phase 3.2, we'll add selection_competition tracking
        # For now, just verify the scoring transparency is working

    def test_phase_3_2_competition_tracking_validation(self):
        """Test comprehensive Phase 3.2 competition tracking and priority-based selection."""
        # Create frames that qualify for multiple categories with clear priority order
        frames = [
            # Frame 1: High quality, standing pose (will be selected for standing)
            create_test_frame("frame1", quality_score=0.9, pose_confidence=0.9, timestamp=0.0),

            # Frame 2: Medium quality, also standing pose (will lose to higher priority)
            create_test_frame("frame2", quality_score=0.7, pose_confidence=0.8, timestamp=5.0),

            # Frame 3: Lower quality, standing pose (will be marked not_top_ranked)
            create_test_frame("frame3", quality_score=0.5, pose_confidence=0.7, timestamp=10.0),

            # Frame 4: Good quality, but will lose to higher priority categories
            create_test_frame("frame4", quality_score=0.8, pose_confidence=0.8, timestamp=15.0),
        ]

        # All frames have standing pose classification
        for frame in frames:
            frame.pose_detections[0].pose_classifications = [("standing", 0.8)]

        # Add closeup classification to frame1 (closeup has higher priority than standing)
        frames[0].closeup_detections = [
            type('CloseupDetection', (), {
                'is_closeup': True,
                'shot_type': 'closeup',
                'confidence': 0.9,
                'face_area_ratio': 0.3,
                'inter_ocular_distance': None,
                'estimated_distance': None,
                'shoulder_width_ratio': None
            })()
        ]

        self.selector.select_best_frames(frames)

        # Validate that all frames were scored for standing category
        for frame in frames:
            assert "pose_standing" in frame.selections.category_scores
            assert "pose_standing" in frame.selections.category_score_breakdowns
            assert "pose_standing" in frame.selections.category_ranks
            assert "pose_standing" in frame.selections.selection_competition

        # Validate frame selection and competition tracking
        # Frame 1 should be selected for some category (pose takes priority over head angle)
        selected_frames = [f for f in frames if f.selections.final_output]
        assert len(selected_frames) > 0  # At least one frame should be selected

        for frame in selected_frames:
            # Each selected frame should have proper metadata
            assert frame.selections.primary_selection_category is not None
            assert frame.selections.final_selection_score is not None
            assert frame.selections.selection_rank is not None

            # The primary category should show "selected" in competition tracking
            primary_cat = frame.selections.primary_selection_category
            assert frame.selections.selection_competition[primary_cat] == "selected"

        # Some frames should be marked as not_top_ranked or rejected
        non_selected_frames = [f for f in frames if not f.selections.final_output]
        for frame in non_selected_frames:
            standing_competition = frame.selections.selection_competition.get("pose_standing")
            assert standing_competition in ["not_top_ranked", "rejected_insufficient_diversity"]
            assert frame.selections.rejection_reason is not None

        # Verify granular rejection reasons are being set
        rejection_reasons = [f.selections.rejection_reason for f in frames if f.selections.rejection_reason]
        assert len(rejection_reasons) > 0
        valid_reasons = {"not_top_ranked", "insufficient_diversity", "not_selected"}
        for reason in rejection_reasons:
            assert reason in valid_reasons
