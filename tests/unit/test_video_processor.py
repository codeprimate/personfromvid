"""Tests for video processor functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from personfromvid.core.video_processor import VideoProcessor
from personfromvid.data import VideoMetadata
from personfromvid.utils.exceptions import VideoProcessingError


class TestVideoProcessor:
    """Test cases for VideoProcessor class."""
    
    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file raises error."""
        with pytest.raises(VideoProcessingError, match="Video file not found"):
            VideoProcessor("nonexistent_file.mp4")
    
    def test_init_with_unreadable_file(self, tmp_path):
        """Test initialization with unreadable file raises error."""
        # Create an empty file 
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"")
        
        # Make it unreadable (this test may not work on all systems)
        try:
            test_file.chmod(0o000)
            with pytest.raises(VideoProcessingError, match="Cannot read video file"):
                VideoProcessor(str(test_file))
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)
    
    @patch('ffmpeg.probe')
    def test_extract_metadata_success(self, mock_probe, tmp_path):
        """Test successful metadata extraction."""
        # Create a temporary file
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content for testing")
        
        # Mock ffmpeg.probe response
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'codec_name': 'h264'
                }
            ],
            'format': {
                'duration': '60.5',
                'format_name': 'mov,mp4,m4a,3gp,3g2,mj2'
            }
        }
        
        processor = VideoProcessor(str(test_file))
        metadata = processor.extract_metadata()
        
        assert isinstance(metadata, VideoMetadata)
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.duration == 60.5
        assert metadata.codec == 'h264'
        assert metadata.format == 'mov'
        assert metadata.total_frames == 1815  # 60.5 * 30
    
    @patch('ffmpeg.probe')
    def test_extract_metadata_no_video_stream(self, mock_probe, tmp_path):
        """Test metadata extraction with no video stream."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake content")
        
        # Mock response with no video stream
        mock_probe.return_value = {
            'streams': [
                {'codec_type': 'audio'}  # Only audio stream
            ],
            'format': {'duration': '60.0'}
        }
        
        processor = VideoProcessor(str(test_file))
        
        with pytest.raises(VideoProcessingError, match="No video stream found"):
            processor.extract_metadata()
    
    @patch('ffmpeg.probe')
    def test_extract_metadata_invalid_dimensions(self, mock_probe, tmp_path):
        """Test metadata extraction with invalid dimensions."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake content")
        
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 0,  # Invalid width
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'codec_name': 'h264'
                }
            ],
            'format': {'duration': '60.0'}
        }
        
        processor = VideoProcessor(str(test_file))
        
        with pytest.raises(VideoProcessingError, match="Invalid video dimensions"):
            processor.extract_metadata()
    
    def test_calculate_hash(self, tmp_path):
        """Test hash calculation."""
        test_file = tmp_path / "test.mp4"
        test_content = b"test video content for hashing"
        test_file.write_bytes(test_content)
        
        processor = VideoProcessor(str(test_file))
        file_hash = processor.calculate_hash()
        
        # Should return a 64-character hex string (SHA256)
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64
        assert all(c in '0123456789abcdef' for c in file_hash)
        
        # Hash should be deterministic
        assert processor.calculate_hash() == file_hash
    
    @patch.object(VideoProcessor, 'extract_metadata')
    def test_validate_format_success(self, mock_extract, tmp_path):
        """Test successful format validation."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake content")
        
        # Mock metadata for a valid video
        mock_extract.return_value = VideoMetadata(
            duration=60.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec='h264',
            total_frames=1800,
            file_size_bytes=1000000,
            format='mp4'
        )
        
        processor = VideoProcessor(str(test_file))
        result = processor.validate_format()
        
        assert result is True
    
    @patch.object(VideoProcessor, 'extract_metadata')
    def test_validate_format_too_short(self, mock_extract, tmp_path):
        """Test format validation with too short video."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake content")
        
        # Mock metadata for a too-short video
        mock_extract.return_value = VideoMetadata(
            duration=0.05,  # Too short
            fps=30.0,
            width=1920,
            height=1080,
            codec='h264',
            total_frames=2,
            file_size_bytes=1000,
            format='mp4'
        )
        
        processor = VideoProcessor(str(test_file))
        
        with pytest.raises(VideoProcessingError, match="Video too short"):
            processor.validate_format()
    
    @patch.object(VideoProcessor, 'extract_metadata')
    def test_validate_format_low_resolution(self, mock_extract, tmp_path):
        """Test format validation with low resolution."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake content")
        
        # Mock metadata for a low resolution video
        mock_extract.return_value = VideoMetadata(
            duration=60.0,
            fps=30.0,
            width=50,   # Too low
            height=50,  # Too low
            codec='h264',
            total_frames=1800,
            file_size_bytes=1000000,
            format='mp4'
        )
        
        processor = VideoProcessor(str(test_file))
        
        with pytest.raises(VideoProcessingError, match="Video resolution too low"):
            processor.validate_format()
    
    @patch.object(VideoProcessor, 'extract_metadata')
    @patch.object(VideoProcessor, 'calculate_hash')
    def test_get_video_info_summary(self, mock_hash, mock_extract, tmp_path):
        """Test video info summary generation."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake content")
        
        # Mock metadata and hash
        mock_extract.return_value = VideoMetadata(
            duration=60.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec='h264',
            total_frames=1800,
            file_size_bytes=1048576,  # 1MB
            format='mp4'
        )
        mock_hash.return_value = 'abcd1234' * 8  # 64-char hash
        
        processor = VideoProcessor(str(test_file))
        info = processor.get_video_info_summary()
        
        assert info['file_name'] == 'test.mp4'
        assert info['file_size_mb'] == 1.0
        assert info['duration_seconds'] == 60.0
        assert info['fps'] == 30.0
        assert info['resolution'] == '1920x1080'
        assert info['codec'] == 'h264'
        assert info['format'] == 'mp4'
        assert info['total_frames'] == 1800
        assert 'estimated_processing_time' in info
        assert 'file_hash' in info 