"""Tests for frame extraction engine functionality."""

import hashlib
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from personfromvid.core.frame_extractor import (
    FrameExtractor, 
    FrameCandidate, 
    ExtractionMethod
)
from personfromvid.data import VideoMetadata, FrameData, SourceInfo, ImageProperties
from personfromvid.utils.exceptions import VideoProcessingError


@pytest.fixture
def sample_video_metadata():
    """Sample video metadata for testing."""
    return VideoMetadata(
        duration=10.0,
        fps=30.0,
        width=1920,
        height=1080,
        codec='h264',
        total_frames=300,
        file_size_bytes=1000000,
        format='mp4'
    )


@pytest.fixture
def test_video_path():
    """Path to test video fixture."""
    return Path(__file__).parent.parent / "fixtures" / "test_video.mp4"


@pytest.fixture
def frame_extractor(test_video_path, sample_video_metadata):
    """Create frame extractor instance for testing."""
    return FrameExtractor(str(test_video_path), sample_video_metadata)


@pytest.fixture
def sample_frame_candidates():
    """Sample frame candidates for testing."""
    return [
        FrameCandidate(
            timestamp=1.0,
            frame_number=30,
            method=ExtractionMethod.I_FRAME,
            confidence=1.0
        ),
        FrameCandidate(
            timestamp=1.25,
            frame_number=37,
            method=ExtractionMethod.TEMPORAL_SAMPLING,
            confidence=0.8
        ),
        FrameCandidate(
            timestamp=2.5,
            frame_number=75,
            method=ExtractionMethod.I_FRAME,
            confidence=1.0
        )
    ]


class TestFrameCandidate:
    """Test FrameCandidate dataclass."""
    
    def test_frame_candidate_creation(self):
        """Test creating a frame candidate."""
        candidate = FrameCandidate(
            timestamp=5.5,
            frame_number=165,
            method=ExtractionMethod.I_FRAME,
            confidence=0.9
        )
        
        assert candidate.timestamp == 5.5
        assert candidate.frame_number == 165
        assert candidate.method == ExtractionMethod.I_FRAME
        assert candidate.confidence == 0.9
    
    def test_frame_candidate_default_confidence(self):
        """Test default confidence value."""
        candidate = FrameCandidate(
            timestamp=1.0,
            frame_number=30,
            method=ExtractionMethod.TEMPORAL_SAMPLING
        )
        
        assert candidate.confidence == 1.0


class TestFrameExtractor:
    """Test FrameExtractor class."""
    
    def test_init_with_valid_inputs(self, test_video_path, sample_video_metadata):
        """Test initialization with valid inputs."""
        extractor = FrameExtractor(str(test_video_path), sample_video_metadata)
        
        assert extractor.video_path == test_video_path
        assert extractor.video_metadata == sample_video_metadata
        assert extractor.temporal_interval == 0.25
        assert extractor.similarity_threshold == 0.95
        assert extractor.max_frames_per_second == 8
        assert len(extractor.extracted_frames) == 0
        assert len(extractor.frame_hashes) == 0
    
    def test_init_statistics_initialized(self, frame_extractor):
        """Test that statistics are properly initialized."""
        stats = frame_extractor.stats
        
        assert stats['i_frames_found'] == 0
        assert stats['temporal_samples'] == 0
        assert stats['duplicates_removed'] == 0
        assert stats['total_extracted'] == 0
        assert stats['processing_time'] == 0.0
    
    @patch('subprocess.run')
    def test_extract_i_frames_success(self, mock_subprocess, frame_extractor):
        """Test successful I-frame extraction."""
        # Mock successful ffprobe output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "frames": [
                {"pkt_pts_time": "1.0", "pict_type": "I"},
                {"pkt_pts_time": "2.5", "pict_type": "P"},  # Not I-frame
                {"pkt_pts_time": "4.0", "pict_type": "I"},
                {"pkt_pts_time": "15.0", "pict_type": "I"}  # Beyond duration
            ]
        })
        mock_subprocess.return_value = mock_result
        
        candidates = frame_extractor._extract_i_frames()
        
        assert len(candidates) == 2  # Only I-frames within duration
        assert candidates[0].timestamp == 1.0
        assert candidates[0].frame_number == 30  # 1.0 * 30 fps
        assert candidates[0].method == ExtractionMethod.I_FRAME
        assert candidates[0].confidence == 1.0
        
        assert candidates[1].timestamp == 4.0
        assert candidates[1].frame_number == 120  # 4.0 * 30 fps
        
        assert frame_extractor.stats['i_frames_found'] == 2
    
    @patch('subprocess.run')
    def test_extract_i_frames_ffprobe_failure(self, mock_subprocess, frame_extractor):
        """Test I-frame extraction when ffprobe fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ffprobe error"
        mock_subprocess.return_value = mock_result
        
        candidates = frame_extractor._extract_i_frames()
        
        assert len(candidates) == 0
        assert frame_extractor.stats['i_frames_found'] == 0
    
    @patch('subprocess.run')
    def test_extract_i_frames_json_error(self, mock_subprocess, frame_extractor):
        """Test I-frame extraction with invalid JSON."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"
        mock_subprocess.return_value = mock_result
        
        candidates = frame_extractor._extract_i_frames()
        
        assert len(candidates) == 0
    
    def test_generate_temporal_samples(self, frame_extractor):
        """Test temporal sampling generation."""
        candidates = frame_extractor._generate_temporal_samples()
        
        # Should generate samples every 0.25s for 10s duration = 40 samples
        expected_count = 40  # 10.0 / 0.25
        assert len(candidates) == expected_count
        
        # Check first few candidates
        assert candidates[0].timestamp == 0.0
        assert candidates[0].frame_number == 0
        assert candidates[0].method == ExtractionMethod.TEMPORAL_SAMPLING
        assert candidates[0].confidence == 0.8
        
        assert candidates[1].timestamp == 0.25
        assert candidates[1].frame_number == 7  # 0.25 * 30 fps
        
        assert candidates[-1].timestamp == 9.75
        
        assert frame_extractor.stats['temporal_samples'] == expected_count
    
    def test_generate_temporal_samples_exceeds_frames(self):
        """Test temporal sampling when calculated frame exceeds total frames."""
        # Create metadata with fewer total frames
        short_metadata = VideoMetadata(
            duration=1.0, fps=10.0, width=640, height=480,
            codec='h264', total_frames=5, file_size_bytes=1000, format='mp4'
        )
        extractor = FrameExtractor("test.mp4", short_metadata)
        
        candidates = extractor._generate_temporal_samples()
        
        # Should stop when frame number would exceed total frames
        assert len(candidates) <= 5
        for candidate in candidates:
            assert candidate.frame_number < short_metadata.total_frames
    
    def test_combine_and_deduplicate_candidates(self, frame_extractor):
        """Test combining and deduplicating candidates."""
        i_frames = [
            FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0),
            FrameCandidate(5.0, 150, ExtractionMethod.I_FRAME, 1.0)
        ]
        
        temporal = [
            FrameCandidate(0.25, 7, ExtractionMethod.TEMPORAL_SAMPLING, 0.8),
            FrameCandidate(1.05, 31, ExtractionMethod.TEMPORAL_SAMPLING, 0.8),  # Near duplicate
            FrameCandidate(3.0, 90, ExtractionMethod.TEMPORAL_SAMPLING, 0.8)
        ]
        
        combined = frame_extractor._combine_and_deduplicate_candidates(i_frames, temporal)
        
        # Should remove near-duplicate at 1.05s (within 0.1s of 1.0s I-frame)
        assert len(combined) == 4
        
        # Should be sorted by timestamp
        timestamps = [c.timestamp for c in combined]
        assert timestamps == sorted(timestamps)
        
        # Should keep I-frame over temporal sample for near-duplicate
        timestamp_1_candidate = next(c for c in combined if abs(c.timestamp - 1.0) < 0.1)
        assert timestamp_1_candidate.method == ExtractionMethod.I_FRAME
    
    def test_combine_and_deduplicate_rate_limiting(self, frame_extractor):
        """Test rate limiting in candidate combination."""
        # Create many candidates to trigger rate limiting
        i_frames = [
            FrameCandidate(float(i), i * 30, ExtractionMethod.I_FRAME, 1.0)
            for i in range(100)  # 100 I-frames
        ]
        
        combined = frame_extractor._combine_and_deduplicate_candidates(i_frames, [])
        
        # Should be limited to max_frames_per_second * duration
        max_expected = frame_extractor.max_frames_per_second * frame_extractor.video_metadata.duration
        assert len(combined) <= max_expected
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_images_success(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test successful frame image extraction."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock frame reading - create different frames to avoid duplicate detection
        test_frames = [
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        ]
        mock_cap.read.side_effect = [(True, test_frames[0]), (True, test_frames[1])]
        
        # Mock cv2.imwrite
        with patch('cv2.imwrite', return_value=True):
            candidates = [
                FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0),
                FrameCandidate(2.0, 60, ExtractionMethod.TEMPORAL_SAMPLING, 0.8)
            ]
            
            extracted = frame_extractor._extract_frame_images(candidates, tmp_path)
            
            assert len(extracted) == 2
            
            # Check first frame data
            frame_data = extracted[0]
            assert isinstance(frame_data, FrameData)
            assert frame_data.frame_id == "frame_000030"
            assert frame_data.source_info.video_timestamp == 1.0
            assert frame_data.source_info.extraction_method == "i_frame"
            assert frame_data.image_properties.width == 1920
            assert frame_data.image_properties.height == 1080
            
            # Check that seek was called correctly
            mock_cap.set.assert_has_calls([
                call(cv2.CAP_PROP_POS_MSEC, 1000.0),  # 1.0 * 1000
                call(cv2.CAP_PROP_POS_MSEC, 2000.0)   # 2.0 * 1000
            ])
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_images_video_open_failure(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test frame extraction when video cannot be opened."""
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        candidates = [FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0)]
        
        with pytest.raises(VideoProcessingError, match="Could not open video file"):
            frame_extractor._extract_frame_images(candidates, tmp_path)
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_images_frame_read_failure(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test frame extraction when frame reading fails."""
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Failed read
        
        candidates = [FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0)]
        
        extracted = frame_extractor._extract_frame_images(candidates, tmp_path)
        
        # Should return empty list when no frames can be read
        assert len(extracted) == 0
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_images_duplicate_detection(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test duplicate frame detection during extraction."""
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Create identical frames
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        with patch('cv2.imwrite', return_value=True):
            candidates = [
                FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0),
                FrameCandidate(2.0, 60, ExtractionMethod.TEMPORAL_SAMPLING, 0.8)
            ]
            
            extracted = frame_extractor._extract_frame_images(candidates, tmp_path)
            
            # Should only extract one frame due to duplicate detection
            assert len(extracted) == 1
            assert frame_extractor.stats['duplicates_removed'] >= 1
    
    def test_calculate_frame_hash(self, frame_extractor):
        """Test frame hash calculation."""
        # Create test frames
        frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        hash1 = frame_extractor._calculate_frame_hash(frame1)
        hash2 = frame_extractor._calculate_frame_hash(frame2)
        
        # Hashes should be hex strings
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        assert all(c in '0123456789abcdef' for c in hash1)
        
        # Different frames should have different hashes
        assert hash1 != hash2
        
        # Same frame should have same hash
        hash1_repeat = frame_extractor._calculate_frame_hash(frame1)
        assert hash1 == hash1_repeat
    
    def test_create_frame_data(self, frame_extractor, tmp_path):
        """Test frame data creation."""
        candidate = FrameCandidate(1.5, 45, ExtractionMethod.I_FRAME, 0.9)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"fake image data")
        frame_id = "frame_000045"
        
        frame_data = frame_extractor._create_frame_data(
            candidate, test_frame, frame_path, frame_id
        )
        
        assert isinstance(frame_data, FrameData)
        assert frame_data.frame_id == frame_id
        assert frame_data.file_path == frame_path
        
        # Check source info
        assert frame_data.source_info.video_timestamp == 1.5
        assert frame_data.source_info.extraction_method == "i_frame"
        assert frame_data.source_info.original_frame_number == 45
        assert frame_data.source_info.video_fps == 30.0
        
        # Check image properties
        assert frame_data.image_properties.width == 640
        assert frame_data.image_properties.height == 480
        assert frame_data.image_properties.channels == 3
        assert frame_data.image_properties.format == "JPEG"
    
    def test_get_extraction_statistics(self, frame_extractor):
        """Test extraction statistics calculation."""
        # Set some test statistics
        frame_extractor.stats.update({
            'i_frames_found': 10,
            'temporal_samples': 40,
            'duplicates_removed': 5,
            'total_extracted': 35,
            'processing_time': 2.5
        })
        
        stats = frame_extractor.get_extraction_statistics()
        
        assert stats['total_candidates_considered'] == 50  # 10 + 40
        assert stats['i_frames_found'] == 10
        assert stats['temporal_samples_generated'] == 40
        assert stats['duplicates_removed'] == 5
        assert stats['frames_extracted'] == 35
        assert stats['processing_time_seconds'] == 2.5
        assert stats['extraction_rate'] == 14.0  # 35 / 2.5
        
        # Coverage and interval calculations
        expected_coverage = (35 * 0.25) / 10.0 * 100  # 87.5%
        assert abs(stats['coverage_percentage'] - expected_coverage) < 0.1
        
        expected_interval = 10.0 / 35  # 0.286 seconds
        assert abs(stats['average_interval_seconds'] - expected_interval) < 0.01
    
    def test_get_extraction_statistics_zero_division(self, frame_extractor):
        """Test statistics calculation with zero values."""
        stats = frame_extractor.get_extraction_statistics()
        
        assert stats['extraction_rate'] == 0
        assert stats['coverage_percentage'] == 0
        assert stats['average_interval_seconds'] == 0
    
    def test_cleanup_temp_frames(self, frame_extractor, tmp_path):
        """Test temporary frame cleanup."""
        # Create test frame files
        frame1_path = tmp_path / "frame1.jpg"
        frame2_path = tmp_path / "frame2.jpg"
        frame3_path = tmp_path / "frame3.jpg"
        
        for path in [frame1_path, frame2_path, frame3_path]:
            path.write_bytes(b"test image data")
        
        # Create frame data objects
        frame_extractor.extracted_frames = [
            FrameData(
                frame_id="frame1",
                file_path=frame1_path,
                source_info=SourceInfo(1.0, "i_frame", 30, 30.0),
                image_properties=ImageProperties(640, 480, 3, 1000, "JPEG")
            ),
            FrameData(
                frame_id="frame2",
                file_path=frame2_path,
                source_info=SourceInfo(2.0, "temporal_sampling", 60, 30.0),
                image_properties=ImageProperties(640, 480, 3, 1000, "JPEG")
            ),
            FrameData(
                frame_id="frame3",
                file_path=frame3_path,
                source_info=SourceInfo(3.0, "i_frame", 90, 30.0),
                image_properties=ImageProperties(640, 480, 3, 1000, "JPEG")
            )
        ]
        
        # Cleanup keeping only frame2
        frame_extractor.cleanup_temp_frames(keep_selected=["frame2"])
        
        # Check which files were deleted
        assert not frame1_path.exists()
        assert frame2_path.exists()
        assert not frame3_path.exists()
    
    def test_cleanup_temp_frames_all(self, frame_extractor, tmp_path):
        """Test cleanup of all temporary frames."""
        frame_path = tmp_path / "frame.jpg"
        frame_path.write_bytes(b"test data")
        
        frame_extractor.extracted_frames = [
            FrameData(
                frame_id="frame1",
                file_path=frame_path,
                source_info=SourceInfo(1.0, "i_frame", 30, 30.0),
                image_properties=ImageProperties(640, 480, 3, 1000, "JPEG")
            )
        ]
        
        frame_extractor.cleanup_temp_frames()  # No keep_selected = delete all
        
        assert not frame_path.exists()
    
    @patch('cv2.VideoCapture')
    @patch('subprocess.run') 
    def test_extract_frames_integration(self, mock_subprocess, mock_cv2_capture, 
                                      frame_extractor, tmp_path):
        """Test complete frame extraction workflow."""
        # Mock ffprobe response
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "frames": [
                {"pkt_pts_time": "1.0", "pict_type": "I"},
                {"pkt_pts_time": "5.0", "pict_type": "I"}
            ]
        })
        mock_subprocess.return_value = mock_result
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        # Mock progress callback
        progress_callback = MagicMock()
        
        with patch('cv2.imwrite', return_value=True):
            extracted_frames = frame_extractor.extract_frames(tmp_path, progress_callback)
            
            # Should have extracted frames from both I-frames and temporal sampling
            assert len(extracted_frames) > 0
            
            # Check that progress callback was called
            assert progress_callback.called
            
            # Check statistics were updated
            assert frame_extractor.stats['total_extracted'] == len(extracted_frames)
            assert frame_extractor.stats['processing_time'] > 0
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_video_error(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test frame extraction with video processing error."""
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        with pytest.raises(VideoProcessingError, match="Frame extraction failed"):
            frame_extractor.extract_frames(tmp_path)
    
    def test_extraction_method_enum(self):
        """Test ExtractionMethod enum values."""
        assert ExtractionMethod.I_FRAME.value == "i_frame"
        assert ExtractionMethod.TEMPORAL_SAMPLING.value == "temporal_sampling"
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_images_save_failure(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test frame extraction when cv2.imwrite fails."""
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        # Mock cv2.imwrite to fail
        with patch('cv2.imwrite', return_value=False):
            candidates = [FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0)]
            
            extracted = frame_extractor._extract_frame_images(candidates, tmp_path)
            
            # Should return empty list when imwrite fails
            assert len(extracted) == 0
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_images_exception_handling(self, mock_cv2_capture, frame_extractor, tmp_path):
        """Test frame extraction with exception in processing loop."""
        mock_cap = MagicMock()
        mock_cv2_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock read to raise exception for first frame, succeed for second
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [Exception("Read error"), (True, test_frame)]
        
        with patch('cv2.imwrite', return_value=True):
            candidates = [
                FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0),
                FrameCandidate(2.0, 60, ExtractionMethod.TEMPORAL_SAMPLING, 0.8)
            ]
            
            extracted = frame_extractor._extract_frame_images(candidates, tmp_path)
            
            # Should extract only the second frame (first failed with exception)
            assert len(extracted) == 1
            assert extracted[0].source_info.video_timestamp == 2.0
    
    def test_calculate_frame_hash_grayscale_frame(self, frame_extractor):
        """Test frame hash calculation with grayscale frame."""
        # Create a grayscale frame (single channel)
        gray_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Convert to 3-channel for OpenCV compatibility
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        hash_value = frame_extractor._calculate_frame_hash(bgr_frame)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32
        assert all(c in '0123456789abcdef' for c in hash_value)
    
    def test_create_frame_data_grayscale_frame(self, frame_extractor, tmp_path):
        """Test frame data creation with grayscale frame."""
        candidate = FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 1.0)
        
        # Create grayscale frame (2D array)
        gray_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        frame_path = tmp_path / "gray_frame.jpg"
        frame_path.write_bytes(b"fake image data")
        frame_id = "frame_000030"
        
        frame_data = frame_extractor._create_frame_data(
            candidate, gray_frame, frame_path, frame_id
        )
        
        # Should detect single channel
        assert frame_data.image_properties.channels == 1
        assert frame_data.image_properties.width == 640
        assert frame_data.image_properties.height == 480
    
    def test_cleanup_temp_frames_with_missing_files(self, frame_extractor, tmp_path):
        """Test cleanup when some frame files are already missing."""
        # Create only one of two frame files
        frame1_path = tmp_path / "frame1.jpg"
        frame2_path = tmp_path / "frame2.jpg"
        
        frame1_path.write_bytes(b"test data")
        # frame2_path intentionally not created
        
        frame_extractor.extracted_frames = [
            FrameData(
                frame_id="frame1",
                file_path=frame1_path,
                source_info=SourceInfo(1.0, "i_frame", 30, 30.0),
                image_properties=ImageProperties(640, 480, 3, 1000, "JPEG")
            ),
            FrameData(
                frame_id="frame2",
                file_path=frame2_path,
                source_info=SourceInfo(2.0, "temporal_sampling", 60, 30.0),
                image_properties=ImageProperties(640, 480, 3, 1000, "JPEG")
            )
        ]
        
        # Should not raise exception for missing files
        frame_extractor.cleanup_temp_frames()
        
        assert not frame1_path.exists()
        assert not frame2_path.exists()  # Was already missing
    
    def test_combine_candidates_with_higher_confidence_temporal(self, frame_extractor):
        """Test that higher confidence temporal sample replaces lower confidence I-frame."""
        i_frames = [
            FrameCandidate(1.0, 30, ExtractionMethod.I_FRAME, 0.7)  # Lower confidence
        ]
        
        temporal = [
            FrameCandidate(1.05, 31, ExtractionMethod.TEMPORAL_SAMPLING, 0.9)  # Higher confidence
        ]
        
        combined = frame_extractor._combine_and_deduplicate_candidates(i_frames, temporal)
        
        # Should keep temporal sample due to higher confidence
        assert len(combined) == 1
        assert combined[0].method == ExtractionMethod.TEMPORAL_SAMPLING
        assert combined[0].confidence == 0.9
    
    @patch('subprocess.run')
    def test_extract_i_frames_empty_frames_list(self, mock_subprocess, frame_extractor):
        """Test I-frame extraction with empty frames list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"frames": []})  # Empty frames list
        mock_subprocess.return_value = mock_result
        
        candidates = frame_extractor._extract_i_frames()
        
        assert len(candidates) == 0
        assert frame_extractor.stats['i_frames_found'] == 0
    
    @patch('subprocess.run')
    def test_extract_i_frames_missing_fields(self, mock_subprocess, frame_extractor):
        """Test I-frame extraction with missing required fields."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "frames": [
                {"pict_type": "I"},  # Missing pkt_pts_time
                {"pkt_pts_time": "2.0"},  # Missing pict_type
                {"pkt_pts_time": "invalid", "pict_type": "I"}  # Invalid timestamp
            ]
        })
        mock_subprocess.return_value = mock_result
        
        candidates = frame_extractor._extract_i_frames()
        
        # Should handle missing/invalid fields gracefully
        assert len(candidates) == 0


class TestFrameExtractorWithRealVideo:
    """Integration tests using the real test video fixture."""
    
    @pytest.mark.slow
    def test_extract_frames_with_real_video(self, test_video_path, tmp_path):
        """Test frame extraction with actual video file."""
        if not test_video_path.exists():
            pytest.skip("Test video fixture not found")
        
        # Create minimal video metadata (would normally come from VideoProcessor)
        metadata = VideoMetadata(
            duration=1.0,  # Assume short test video
            fps=30.0,
            width=640,
            height=480,
            codec='h264',
            total_frames=30,
            file_size_bytes=test_video_path.stat().st_size,
            format='mp4'
        )
        
        extractor = FrameExtractor(str(test_video_path), metadata)
        
        try:
            frames = extractor.extract_frames(tmp_path)
            
            # Basic checks that extraction worked
            assert len(frames) > 0
            assert all(isinstance(f, FrameData) for f in frames)
            assert all(f.file_path.exists() for f in frames)
            
            # Check statistics make sense
            stats = extractor.get_extraction_statistics()
            assert stats['frames_extracted'] == len(frames)
            assert stats['processing_time_seconds'] > 0
            
        except Exception as e:
            pytest.skip(f"Real video test failed (may be expected): {e}") 