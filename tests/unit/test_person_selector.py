"""Unit tests for PersonSelector class."""

from unittest.mock import Mock

import pytest

from personfromvid.analysis.person_selector import (
    PersonCandidate,
    PersonSelection,
    PersonSelector,
)
from personfromvid.data.config import PersonSelectionCriteria
from personfromvid.data.detection_results import FaceDetection, PoseDetection
from personfromvid.data.frame_data import FrameData
from personfromvid.data.person import Person, PersonQuality


class TestPersonCandidate:
    """Test PersonCandidate dataclass."""

    def test_person_candidate_creation(self):
        """Test PersonCandidate creation and properties."""
        # Create mock frame and person
        frame = Mock(spec=FrameData)
        frame.timestamp = 10.5

        person = Mock(spec=Person)
        person.person_id = 2
        person.quality = Mock()
        person.quality.overall_quality = 0.75

        candidate = PersonCandidate(frame=frame, person=person)

        assert candidate.frame is frame
        assert candidate.person is person
        assert candidate.person_id == 2
        assert candidate.quality_score == 0.75
        assert candidate.timestamp == 10.5


class TestPersonSelection:
    """Test PersonSelection dataclass."""

    def test_person_selection_creation(self):
        """Test PersonSelection creation and properties."""
        # Create mock frame and person
        frame_data = Mock(spec=FrameData)
        frame_data.timestamp = 15.2

        person = Mock(spec=Person)

        selection = PersonSelection(
            frame_data=frame_data,
            person_id=1,
            person=person,
            selection_score=0.85,
            category="minimum"
        )

        assert selection.frame_data is frame_data
        assert selection.person_id == 1
        assert selection.person is person
        assert selection.selection_score == 0.85
        assert selection.category == "minimum"
        assert selection.timestamp == 15.2


class TestPersonSelector:
    """Test PersonSelector class."""

    def test_person_selector_initialization_default(self):
        """Test PersonSelector initialization with default criteria."""
        selector = PersonSelector()

        assert selector.criteria is not None
        assert selector.criteria.min_instances_per_person == 3
        assert selector.criteria.max_instances_per_person == 10
        assert selector.criteria.min_quality_threshold == 0.3
        assert selector.logger is not None

    def test_person_selector_initialization_custom(self):
        """Test PersonSelector initialization with custom criteria."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=2,
            max_instances_per_person=5,
            min_quality_threshold=0.5
        )

        selector = PersonSelector(criteria)

        assert selector.criteria is criteria
        assert selector.criteria.min_instances_per_person == 2
        assert selector.criteria.max_instances_per_person == 5
        assert selector.criteria.min_quality_threshold == 0.5

    def test_select_persons_empty_input(self):
        """Test select_persons with empty input."""
        selector = PersonSelector()
        result = selector.select_persons([])

        assert result == []

    def test_select_persons_no_valid_persons(self):
        """Test select_persons with frames containing no valid persons."""
        selector = PersonSelector()

        # Create frame with no persons
        frame1 = Mock(spec=FrameData)
        frame1.persons = []

        # Create frame with persons below quality threshold
        frame2 = Mock(spec=FrameData)
        frame2.frame_id = 2

        low_quality_person = Mock(spec=Person)
        low_quality_person.person_id = 1
        low_quality_person.quality = Mock()
        low_quality_person.quality.overall_quality = 0.1  # Below default 0.3 threshold

        frame2.persons = [low_quality_person]

        result = selector.select_persons([frame1, frame2])

        assert result == []

    def test_extract_and_group_persons_basic(self):
        """Test extract_and_group_persons with basic input."""
        selector = PersonSelector()

        # Create frames with persons
        frame1 = self._create_frame_with_persons(1, [(0, 0.8), (1, 0.7)])
        frame2 = self._create_frame_with_persons(2, [(0, 0.9), (1, 0.6)])

        groups = selector.extract_and_group_persons([frame1, frame2])

        assert len(groups) == 2
        assert 0 in groups
        assert 1 in groups
        assert len(groups[0]) == 2  # Person 0 appears in both frames
        assert len(groups[1]) == 2  # Person 1 appears in both frames

        # Check person_id grouping
        assert all(candidate.person_id == 0 for candidate in groups[0])
        assert all(candidate.person_id == 1 for candidate in groups[1])

    def test_extract_and_group_persons_quality_filtering(self):
        """Test extract_and_group_persons filters by quality threshold."""
        criteria = PersonSelectionCriteria(min_quality_threshold=0.5)
        selector = PersonSelector(criteria)

        # Create frame with mixed quality persons
        frame = self._create_frame_with_persons(1, [
            (0, 0.8),  # Above threshold
            (1, 0.3),  # Below threshold
            (2, 0.6),  # Above threshold
        ])

        groups = selector.extract_and_group_persons([frame])

        assert len(groups) == 2  # Only persons 0 and 2 should be included
        assert 0 in groups
        assert 1 not in groups  # Filtered out
        assert 2 in groups

    def test_select_best_instances_for_person_minimum_only(self):
        """Test select_best_instances_for_person with only minimum instances."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=2,
            max_instances_per_person=5,
            temporal_diversity_threshold=0.0,  # Disable temporal diversity
            enable_pose_categories=False,  # Disable category-based selection
            enable_head_angle_categories=False  # Disable category-based selection
        )
        selector = PersonSelector(criteria)

        # Create candidates with different quality scores
        candidates = [
            self._create_person_candidate(1, 0, 0.9, 10.0),  # Best quality
            self._create_person_candidate(2, 0, 0.7, 20.0),  # Second best
            self._create_person_candidate(3, 0, 0.5, 30.0),  # Third best
        ]

        selections = selector.select_best_instances_for_person(0, candidates)

        # With temporal diversity disabled and category selection disabled, should select all 3 candidates (up to max_instances_per_person=5)
        assert len(selections) == 3  # All candidates selected when temporal diversity disabled
        assert len([s for s in selections if s.category == "minimum"]) == 2  # First 2 are minimum
        assert len([s for s in selections if s.category == "additional"]) == 1  # Third is additional
        assert selections[0].selection_score == 0.9  # Best quality first
        assert selections[1].selection_score == 0.7  # Second best quality
        assert selections[2].selection_score == 0.5  # Third best quality

    def test_select_best_instances_for_person_with_temporal_diversity(self):
        """Test select_best_instances_for_person with temporal diversity applied to all selections."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=3,
            max_instances_per_person=8,
            temporal_diversity_threshold=5.0,  # 5 second minimum gap
            enable_pose_categories=False,  # Disable category-based selection
            enable_head_angle_categories=False  # Disable category-based selection
        )
        selector = PersonSelector(criteria)

        # Create candidates with timestamps that don't respect temporal diversity
        candidates = [
            self._create_person_candidate(0, 0, 0.9, 0.0),   # Best quality
            self._create_person_candidate(1, 0, 0.8, 2.0),   # Too close (2s gap)
            self._create_person_candidate(2, 0, 0.7, 3.0),   # Too close (3s gap)
            self._create_person_candidate(3, 0, 0.6, 6.0),   # Good gap (6s from first)
            self._create_person_candidate(4, 0, 0.5, 12.0),  # Good gap (6s from previous)
            self._create_person_candidate(5, 0, 0.4, 15.0),  # Too close (3s gap)
            self._create_person_candidate(6, 0, 0.3, 18.0),  # Good gap (6s from candidate 4)
        ]

        selections = selector.select_best_instances_for_person(0, candidates)

        # With temporal diversity filtering applied to all selections,
        # only candidates with sufficient temporal gaps should be selected
        assert len(selections) == 4  # Candidates 0, 3, 4, 6
        selected_frame_ids = [int(s.frame_data.frame_id) for s in selections]
        assert selected_frame_ids == [0, 3, 4, 6]

        # Verify all selections respect temporal diversity
        timestamps = [s.timestamp for s in selections]
        for i in range(len(timestamps)):
            for j in range(i + 1, len(timestamps)):
                time_diff = abs(timestamps[i] - timestamps[j])
                assert time_diff >= 5.0  # All selections should respect temporal diversity

    def test_select_best_instances_for_person_max_limit(self):
        """Test select_best_instances_for_person respects max instances limit."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=2,
            max_instances_per_person=3,  # Low max limit
            temporal_diversity_threshold=0.0,  # Disable temporal diversity
            enable_pose_categories=False,  # Disable category-based selection
            enable_head_angle_categories=False  # Disable category-based selection
        )
        selector = PersonSelector(criteria)

        # Create many candidates
        candidates = [
            self._create_person_candidate(i, 0, 0.9 - i * 0.1, i * 10.0)
            for i in range(10)  # 10 candidates
        ]

        selections = selector.select_best_instances_for_person(0, candidates)

        # With temporal_diversity_threshold=0.0, should select best candidates up to max limit
        # Since max_instances_per_person=3, should select top 3 candidates by quality
        assert len(selections) == 3  # Should select up to max limit when temporal diversity disabled
        assert len([s for s in selections if s.category == "minimum"]) == 2  # First 2 are minimum
        assert len([s for s in selections if s.category == "additional"]) == 1  # Third is additional

        # Should be ordered by quality (descending)
        assert selections[0].selection_score == 0.9  # Best quality
        assert selections[1].selection_score == 0.8  # Second best
        assert selections[2].selection_score == 0.7  # Third best

    def test_integrated_temporal_diversity_filtering(self):
        """Test integrated temporal diversity filtering in select_best_instances_for_person."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=2,
            max_instances_per_person=5,
            temporal_diversity_threshold=5.0,
            enable_pose_categories=False,  # Disable category-based selection
            enable_head_angle_categories=False  # Disable category-based selection
        )
        selector = PersonSelector(criteria)

        # Create candidates with various timestamps and quality scores
        candidates = [
            self._create_person_candidate(0, 0, 0.9, 0.0),   # Best quality, first
            self._create_person_candidate(1, 0, 0.8, 2.0),   # Too close to first (within 5s)
            self._create_person_candidate(2, 0, 0.7, 6.0),   # Good gap from first
            self._create_person_candidate(3, 0, 0.6, 8.0),   # Too close to second (within 5s)
            self._create_person_candidate(4, 0, 0.5, 12.0),  # Good gap from second
        ]

        selections = selector.select_best_instances_for_person(0, candidates)

        # Should only select candidates with sufficient temporal diversity
        assert len(selections) == 3  # Candidates 0, 2, and 4
        selected_frame_ids = [int(s.frame_data.frame_id) for s in selections]
        assert selected_frame_ids == [0, 2, 4]

        # Verify timestamps have sufficient diversity
        timestamps = [s.timestamp for s in selections]
        for i in range(len(timestamps)):
            for j in range(i + 1, len(timestamps)):
                time_diff = abs(timestamps[i] - timestamps[j])
                assert time_diff >= 5.0  # All selections should respect temporal diversity

    def test_select_persons_integration(self):
        """Test complete select_persons integration."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=2,
            max_instances_per_person=4,
            min_quality_threshold=0.4,
            temporal_diversity_threshold=3.0,
            max_total_selections=10
        )
        selector = PersonSelector(criteria)

        # Create frames with multiple persons
        frames = [
            self._create_frame_with_persons(1, [(0, 0.9), (1, 0.8)], 10.0),
            self._create_frame_with_persons(2, [(0, 0.7), (1, 0.6)], 12.0),  # Too close
            self._create_frame_with_persons(3, [(0, 0.8), (1, 0.9)], 15.0),  # Good gap
            self._create_frame_with_persons(4, [(0, 0.6), (1, 0.7)], 18.0),  # Good gap
            self._create_frame_with_persons(5, [(0, 0.5), (1, 0.3)], 20.0),  # Person 1 below threshold
        ]

        selections = selector.select_persons(frames)

        # Should have selections for both persons
        person_0_selections = [s for s in selections if s.person_id == 0]
        person_1_selections = [s for s in selections if s.person_id == 1]

        assert len(person_0_selections) >= 2  # At least minimum
        assert len(person_1_selections) >= 2  # At least minimum
        assert len(selections) <= 10  # Respects global limit

        # Check quality threshold filtering
        assert all(s.selection_score >= 0.4 for s in selections)

    def test_select_persons_global_max_limit(self):
        """Test select_persons respects global max_total_selections limit."""
        criteria = PersonSelectionCriteria(
            min_instances_per_person=5,
            max_instances_per_person=10,
            max_total_selections=15  # Low global limit (min is 10)
        )
        selector = PersonSelector(criteria)

        # Create many high-quality persons
        frames = []
        for frame_id in range(20):
            persons_data = [(person_id, 0.9) for person_id in range(3)]  # 3 persons per frame
            frame = self._create_frame_with_persons(frame_id, persons_data, frame_id * 1.0)
            frames.append(frame)

        selections = selector.select_persons(frames)

        assert len(selections) == 15  # Respects global limit
        # Should prioritize highest quality selections
        assert all(s.selection_score >= 0.8 for s in selections)

    def _create_frame_with_persons(self, frame_id: int, persons_data: list, timestamp: float = 0.0) -> FrameData:
        """Helper to create FrameData with persons.

        Args:
            frame_id: Frame ID
            persons_data: List of (person_id, quality_score) tuples
            timestamp: Frame timestamp

        Returns:
            FrameData with populated persons
        """
        from pathlib import Path

        from personfromvid.data.frame_data import ImageProperties, SourceInfo

        # Create proper SourceInfo with timestamp
        source_info = SourceInfo(
            video_timestamp=timestamp,
            extraction_method="test",
            original_frame_number=frame_id,
            video_fps=30.0
        )

        # Create basic image properties
        image_properties = ImageProperties(
            width=640,
            height=480,
            channels=3,
            file_size_bytes=1024,
            format="JPEG"
        )

        frame = FrameData(
            frame_id=str(frame_id),
            file_path=Path(f"test_frame_{frame_id}.jpg"),
            source_info=source_info,
            image_properties=image_properties
        )
        frame.persons = []

        for person_id, quality_score in persons_data:
            # Create basic face and body detections
            face = FaceDetection(bbox=(10, 10, 50, 50), confidence=0.9, landmarks=None)
            body = PoseDetection(bbox=(5, 5, 60, 100), confidence=0.8, keypoints={}, pose_classifications=[])

            # Create person quality
            quality = PersonQuality(face_quality=quality_score, body_quality=quality_score)

            # Create person
            person = Person(
                person_id=person_id,
                face=face,
                body=body,
                head_pose=None,
                quality=quality
            )

            frame.persons.append(person)

        return frame

    def _create_person_candidate(self, frame_id: int, person_id: int,
                                                              quality_score: float, timestamp: float) -> PersonCandidate:
        """Helper to create PersonCandidate.

        Args:
            frame_id: Frame ID
            person_id: Person ID
            quality_score: Quality score
            timestamp: Frame timestamp

        Returns:
            PersonCandidate object
        """
        frame = Mock(spec=FrameData)
        frame.frame_id = frame_id
        frame.timestamp = timestamp

        person = Mock(spec=Person)
        person.person_id = person_id
        person.quality = Mock()
        person.quality.overall_quality = quality_score

        # Configure body attribute to avoid AttributeError
        person.body = Mock()
        person.body.pose_classifications = [("standing", 0.9)]  # Default pose

        # Configure head_pose attribute to avoid AttributeError
        person.head_pose = Mock()
        person.head_pose.direction = "front"  # Default direction

        return PersonCandidate(frame=frame, person=person)


if __name__ == "__main__":
    pytest.main([__file__])
