"""Unit tests for PersonSelectionStep pipeline step."""

from unittest.mock import Mock, patch

import pytest

from personfromvid.analysis.person_selector import PersonSelection
from personfromvid.core.steps.person_selection import (
    ALL_SELECTED_PERSONS_KEY,
    PersonSelectionStep,
)
from personfromvid.data.config import PersonSelectionCriteria
from personfromvid.data.constants import QualityMethod
from personfromvid.data.detection_results import (
    FaceDetection,
    PoseDetection,
    QualityMetrics,
)
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo
from personfromvid.data.person import BodyUnknown, FaceUnknown, Person, PersonQuality


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = Mock()
    pipeline.logger = Mock()
    pipeline.formatter = None

    # Mock config with person selection criteria
    pipeline.config = Mock()
    pipeline.config.person_selection = PersonSelectionCriteria(
        enabled=True,
        min_instances_per_person=2,
        max_instances_per_person=5,
        min_quality_threshold=0.3,
        temporal_diversity_threshold=1.0,
        max_total_selections=50
    )

    # Mock state
    pipeline.state = Mock()
    mock_step_progress = Mock()
    pipeline.state.get_step_progress.return_value = mock_step_progress

    return pipeline


@pytest.fixture
def sample_frames_with_persons():
    """Create sample FrameData objects with Person objects for testing."""
    frames = []

    # Create test persons
    for frame_idx in range(3):
        from pathlib import Path
        frame = FrameData(
            frame_id=f"frame_{frame_idx}",
            file_path=Path(f"/tmp/frame_{frame_idx}.jpg"),
            source_info=SourceInfo(
                video_timestamp=float(frame_idx),
                extraction_method="temporal_sampling",
                original_frame_number=frame_idx,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=1920,
                height=1080,
                channels=3,
                file_size_bytes=1024000,
                format="JPEG"
            ),
            quality_metrics=QualityMetrics(
                laplacian_variance=150.0,
                sobel_variance=120.0,
                brightness_score=100.0,
                contrast_score=50.0,
                overall_quality=0.8,
                method=QualityMethod.DIRECT
            )
        )

        # Add persons to frame
        persons = []
        for person_idx in range(2):  # 2 persons per frame
            person = Person(
                person_id=person_idx,
                face=FaceUnknown() if person_idx == 1 else FaceDetection(
                    bbox=[100 + person_idx * 50, 100, 150 + person_idx * 50, 150],
                    confidence=0.9,
                    landmarks=None
                ),
                body=BodyUnknown() if person_idx == 0 else PoseDetection(
                    bbox=[90 + person_idx * 50, 90, 160 + person_idx * 50, 160],
                    confidence=0.8,
                    keypoints=[]
                ),
                head_pose=None,
                quality=PersonQuality(
                    face_quality=0.7 if person_idx == 0 else 0.0,
                    body_quality=0.0 if person_idx == 0 else 0.8
                )
            )
            persons.append(person)

        frame.persons = persons
        frames.append(frame)

    return frames


def test_person_selection_step_initialization(mock_pipeline):
    """Test PersonSelectionStep initialization."""
    step = PersonSelectionStep(mock_pipeline)

    assert step.step_name == "person_selection"
    assert step.pipeline == mock_pipeline
    assert step.logger == mock_pipeline.logger


def test_person_selection_step_empty_frames(mock_pipeline):
    """Test PersonSelectionStep behavior with no candidate frames."""
    step = PersonSelectionStep(mock_pipeline)
    mock_pipeline.state.frames = []

    step.execute()

    # Should start with 0 progress and return early
    mock_pipeline.state.get_step_progress().start.assert_called_with(0)

    # Should log warning about no frames
    mock_pipeline.logger.warning.assert_called_with(
        "⚠️ No frames with persons and quality assessments for selection"
    )


def test_person_selection_step_no_persons(mock_pipeline):
    """Test PersonSelectionStep behavior with frames that have no persons."""
    step = PersonSelectionStep(mock_pipeline)

    # Create frames without persons
    from pathlib import Path
    frames = [
        FrameData(
            frame_id="frame_0",
            file_path=Path("/tmp/frame_0.jpg"),
            source_info=SourceInfo(
                video_timestamp=0.0,
                extraction_method="temporal_sampling",
                original_frame_number=0,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=1920,
                height=1080,
                channels=3,
                file_size_bytes=1024000,
                format="JPEG"
            ),
            quality_metrics=QualityMetrics(
                laplacian_variance=150.0,
                sobel_variance=120.0,
                brightness_score=100.0,
                contrast_score=50.0,
                overall_quality=0.8,
                method=QualityMethod.DIRECT
            )
        )
    ]
    frames[0].persons = []  # No persons

    mock_pipeline.state.frames = frames

    step.execute()

    # Should start with 0 progress and return early
    mock_pipeline.state.get_step_progress().start.assert_called_with(0)


@patch('personfromvid.core.steps.person_selection.PersonSelector')
def test_person_selection_step_successful_execution(mock_person_selector_class, mock_pipeline, sample_frames_with_persons):
    """Test successful PersonSelectionStep execution."""
    # Setup mock PersonSelector
    mock_person_selector = Mock()
    mock_person_selector_class.return_value = mock_person_selector

    # Create mock PersonSelection objects
    mock_selections = [
        PersonSelection(
            frame_data=sample_frames_with_persons[0],
            person_id=0,
            person=sample_frames_with_persons[0].persons[0],
            selection_score=0.49,
            category="minimum"
        ),
        PersonSelection(
            frame_data=sample_frames_with_persons[1],
            person_id=1,
            person=sample_frames_with_persons[1].persons[1],
            selection_score=0.56,
            category="additional"
        )
    ]
    mock_person_selector.select_persons.return_value = mock_selections

    # Setup pipeline state
    mock_pipeline.state.frames = sample_frames_with_persons

    step = PersonSelectionStep(mock_pipeline)
    step.execute()

    # Verify PersonSelector was created with correct criteria
    mock_person_selector_class.assert_called_once_with(mock_pipeline.config.person_selection)

    # Verify person selection was called with candidate frames
    mock_person_selector.select_persons.assert_called_once()
    called_frames = mock_person_selector.select_persons.call_args[0][0]
    assert len(called_frames) == 3  # All frames have persons and quality metrics

    # Verify progress tracking was set up correctly
    total_candidates = sum(len(f.persons) for f in sample_frames_with_persons)  # 6 total persons
    mock_pipeline.state.get_step_progress().start.assert_called_with(total_candidates)

    # Verify selection results were stored
    mock_step_progress = mock_pipeline.state.get_step_progress()
    mock_step_progress.set_data.assert_called()

    # Check that PersonSelection objects were stored
    call_args = mock_step_progress.set_data.call_args_list
    persons_call = None
    for call in call_args:
        if call[0][0] == ALL_SELECTED_PERSONS_KEY:
            persons_call = call
            break

    assert persons_call is not None
    # The actual implementation stores serialized dictionaries, not PersonSelection objects
    stored_data = persons_call[0][1]
    assert len(stored_data) == len(mock_selections)
    # Check that the stored data has the expected structure (serialized PersonSelection data)
    expected_keys = {"frame_id", "person_id", "selection_score", "category", "timestamp"}
    for item in stored_data:
        assert set(item.keys()) == expected_keys


@patch('personfromvid.core.steps.person_selection.PersonSelector')
def test_person_selection_step_with_formatter(mock_person_selector_class, mock_pipeline, sample_frames_with_persons):
    """Test PersonSelectionStep execution with rich formatter."""
    # Setup rich formatter
    mock_formatter = Mock()
    mock_progress_bar = Mock()
    mock_formatter.create_progress_bar.return_value.__enter__ = Mock(return_value=mock_progress_bar)
    mock_formatter.create_progress_bar.return_value.__exit__ = Mock(return_value=None)
    mock_pipeline.formatter = mock_formatter

    # Setup mock PersonSelector
    mock_person_selector = Mock()
    mock_person_selector_class.return_value = mock_person_selector
    mock_person_selector.select_persons.return_value = []

    mock_pipeline.state.frames = sample_frames_with_persons

    step = PersonSelectionStep(mock_pipeline)
    step.execute()

    # Verify rich formatter was used
    mock_formatter.print_info.assert_called_with("🎯 Optimizing person selection...", "targeting")
    mock_formatter.create_progress_bar.assert_called_with("Selecting persons", 6)  # 6 total persons


@patch('personfromvid.core.steps.person_selection.PersonSelector')
def test_person_selection_step_error_handling(mock_person_selector_class, mock_pipeline, sample_frames_with_persons):
    """Test PersonSelectionStep error handling."""
    # Setup PersonSelector to raise an exception
    mock_person_selector = Mock()
    mock_person_selector_class.return_value = mock_person_selector
    mock_person_selector.select_persons.side_effect = Exception("Selection failed")

    mock_pipeline.state.frames = sample_frames_with_persons

    step = PersonSelectionStep(mock_pipeline)

    with pytest.raises(Exception, match="Selection failed"):
        step.execute()

    # Verify error was logged and step was marked as failed
    mock_pipeline.logger.error.assert_called_with("❌ Person selection failed: Selection failed")
    mock_pipeline.state.fail_step.assert_called_with("person_selection", "Selection failed")


def test_store_selection_results(mock_pipeline, sample_frames_with_persons):
    """Test _store_selection_results method."""
    step = PersonSelectionStep(mock_pipeline)

    # Create mock selections
    selections = [
        PersonSelection(
            frame_data=sample_frames_with_persons[0],
            person_id=0,
            person=sample_frames_with_persons[0].persons[0],
            selection_score=0.49,
            category="minimum"
        ),
        PersonSelection(
            frame_data=sample_frames_with_persons[1],
            person_id=0,
            person=sample_frames_with_persons[1].persons[0],
            selection_score=0.52,
            category="additional"
        ),
        PersonSelection(
            frame_data=sample_frames_with_persons[2],
            person_id=1,
            person=sample_frames_with_persons[2].persons[1],
            selection_score=0.56,
            category="minimum"
        )
    ]

    step._store_selection_results(selections, total_candidates=6)

    # Verify detailed results were stored
    mock_step_progress = mock_pipeline.state.get_step_progress()
    mock_step_progress.set_data.assert_called()

    # Check that summary data includes expected fields
    call_args = mock_step_progress.set_data.call_args_list
    summary_call = None
    for call in call_args:
        if call[0][0] == "person_selections":
            summary_call = call
            break

    assert summary_call is not None
    summary_data = summary_call[0][1]

    assert summary_data["summary"]["total_candidates"] == 6
    assert summary_data["summary"]["total_selected"] == 3
    assert summary_data["summary"]["unique_persons"] == 2
    assert "person_0" in summary_data["person_groups"]
    assert "person_1" in summary_data["person_groups"]


def test_format_and_log_results_with_formatter(mock_pipeline):
    """Test _format_and_log_results with rich formatter."""
    mock_formatter = Mock()
    mock_pipeline.formatter = mock_formatter

    step = PersonSelectionStep(mock_pipeline)

    # Create sample selections
    from pathlib import Path
    frame = FrameData(
        frame_id="test",
        file_path=Path("/tmp/test.jpg"),
        source_info=SourceInfo(
            video_timestamp=0.0,
            extraction_method="temporal_sampling",
            original_frame_number=0,
            video_fps=30.0
        ),
        image_properties=ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=1024000,
            format="JPEG"
        )
    )
    person = Person(
        person_id=0,
        face=FaceDetection(
            bbox=[100, 100, 150, 150],
            confidence=0.9,
            landmarks=None
        ),
        body=BodyUnknown(),
        head_pose=None,
        quality=PersonQuality(face_quality=0.7, body_quality=0.0)
    )

    selections = [
        PersonSelection(
            frame_data=frame,
            person_id=0,
            person=person,
            selection_score=0.5,
            category="minimum"
        )
    ]

    step._format_and_log_results(selections, total_candidates=5)

    # Verify step results were stored for rich formatter
    mock_step_progress = mock_pipeline.state.get_step_progress()
    mock_step_progress.set_data.assert_called_with("step_results", {
        "candidates_summary": "📊 Candidates: 5 person instances",
        "selected_summary": "✅ Selected 1 person instances",
        "person_breakdown": "👥 Persons: person_0 (1)",
        "category_breakdown": "📂 Categories: minimum (1)",
    })


def test_format_and_log_results_without_formatter(mock_pipeline):
    """Test _format_and_log_results without rich formatter."""
    mock_pipeline.formatter = None

    step = PersonSelectionStep(mock_pipeline)
    step._format_and_log_results([], total_candidates=5)

    # Verify basic logging was used
    mock_pipeline.logger.info.assert_called_with(
        "✅ Person selection completed: 0 person instances from 5 candidates"
    )
    mock_pipeline.logger.warning.assert_called_with(
        "⚠️  No person instances were selected - check quality thresholds!"
    )
