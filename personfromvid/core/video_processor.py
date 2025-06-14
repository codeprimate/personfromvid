"""Video processing and metadata extraction.

This module implements the VideoProcessor class for analyzing video files
and extracting metadata using FFmpeg.
"""

import hashlib
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

import ffmpeg

from ..data import VideoMetadata
from ..utils.logging import get_logger
from ..utils.exceptions import VideoProcessingError


class VideoProcessor:
    """Video processor for metadata extraction and analysis."""

    def __init__(self, video_path: str):
        """Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self.logger = get_logger("video_processor")

        if not self.video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        if not self.video_path.is_file():
            raise VideoProcessingError(f"Path is not a file: {video_path}")

        # Check if file is readable
        try:
            with open(self.video_path, "rb") as f:
                f.read(1)
        except (IOError, OSError) as e:
            raise VideoProcessingError(f"Cannot read video file: {e}")

    def extract_metadata(self) -> VideoMetadata:
        """Extract comprehensive video metadata using FFmpeg.

        Returns:
            VideoMetadata object with complete video information

        Raises:
            VideoProcessingError: If metadata extraction fails
        """
        self.logger.info(f"Extracting metadata from: {self.video_path}")

        try:
            # Use ffmpeg.probe to get detailed metadata
            probe_data = ffmpeg.probe(str(self.video_path))

            # Find video stream
            video_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                raise VideoProcessingError("No video stream found in file")

            # Extract basic properties
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))

            if width <= 0 or height <= 0:
                raise VideoProcessingError(
                    f"Invalid video dimensions: {width}x{height}"
                )

            # Calculate duration
            format_info = probe_data.get("format", {})
            duration = float(format_info.get("duration", 0))

            if duration <= 0:
                # Try to get duration from video stream
                duration = float(video_stream.get("duration", 0))

            if duration <= 0:
                raise VideoProcessingError("Could not determine video duration")

            # Calculate FPS
            fps_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0
            else:
                fps = float(fps_str)

            if fps <= 0:
                # Try alternative FPS field
                fps_str = video_stream.get("avg_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 0
                else:
                    fps = float(fps_str)

            if fps <= 0:
                raise VideoProcessingError("Could not determine video frame rate")

            # Calculate total frames
            total_frames = int(duration * fps)

            # Get codec information
            codec = video_stream.get("codec_name", "unknown")

            # Get container format
            format_name = format_info.get("format_name", "unknown")
            # Extract primary format (e.g., "mp4" from "mov,mp4,m4a,3gp,3g2,mj2")
            if "," in format_name:
                format_name = format_name.split(",")[0]

            # Get file size
            file_size = self.video_path.stat().st_size

            # Create metadata object
            metadata = VideoMetadata(
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
                total_frames=total_frames,
                file_size_bytes=file_size,
                format=format_name,
            )

            self.logger.info(
                f"Video metadata extracted: {metadata.resolution_string}, "
                f"{duration:.1f}s, {fps:.1f}fps, {codec} codec"
            )

            return metadata

        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error during metadata extraction: {e}"
            if hasattr(e, "stderr") and e.stderr:
                error_msg += f"\nFFmpeg stderr: {e.stderr.decode()}"
            raise VideoProcessingError(error_msg)

        except (KeyError, ValueError, TypeError) as e:
            raise VideoProcessingError(f"Failed to parse video metadata: {e}")

        except Exception as e:
            raise VideoProcessingError(
                f"Unexpected error during metadata extraction: {e}"
            )

    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of video file for integrity checking.

        Returns:
            Hexadecimal string representation of SHA256 hash

        Raises:
            VideoProcessingError: If hash calculation fails
        """
        self.logger.debug(f"Calculating SHA256 hash for: {self.video_path}")

        try:
            hash_sha256 = hashlib.sha256()

            # Read file in chunks to handle large files efficiently
            with open(self.video_path, "rb") as f:
                chunk_size = 8192  # 8KB chunks
                while chunk := f.read(chunk_size):
                    hash_sha256.update(chunk)

            file_hash = hash_sha256.hexdigest()
            self.logger.debug(f"Video hash calculated: {file_hash[:16]}...")

            return file_hash

        except (IOError, OSError) as e:
            raise VideoProcessingError(f"Failed to read video file for hashing: {e}")

        except Exception as e:
            raise VideoProcessingError(f"Unexpected error during hash calculation: {e}")

    def validate_format(self) -> bool:
        """Validate video format and codec compatibility.

        Returns:
            True if video format is supported for processing

        Raises:
            VideoProcessingError: If format validation fails
        """
        self.logger.debug(f"Validating video format: {self.video_path}")

        try:
            metadata = self.extract_metadata()

            # Check for supported formats
            supported_formats = {
                "mp4",
                "mov",
                "avi",
                "mkv",
                "webm",
                "flv",
                "m4v",
                "3gp",
                "wmv",
                "mpg",
                "mpeg",
            }

            supported_codecs = {
                "h264",
                "h265",
                "hevc",
                "vp8",
                "vp9",
                "av1",
                "mpeg4",
                "xvid",
                "divx",
                "wmv3",
                "msmpeg4v3",
            }

            format_supported = metadata.format.lower() in supported_formats
            codec_supported = metadata.codec.lower() in supported_codecs

            if not format_supported:
                self.logger.warning(
                    f"Potentially unsupported format: {metadata.format}"
                )

            if not codec_supported:
                self.logger.warning(f"Potentially unsupported codec: {metadata.codec}")

            # Basic validation checks
            if metadata.duration < 0.1:
                raise VideoProcessingError(f"Video too short: {metadata.duration}s")

            if metadata.fps < 1 or metadata.fps > 240:
                raise VideoProcessingError(f"Invalid frame rate: {metadata.fps}fps")

            if metadata.width < 100 or metadata.height < 100:
                raise VideoProcessingError(
                    f"Video resolution too low: {metadata.resolution_string}"
                )

            if metadata.file_size_bytes < 1024:  # Less than 1KB
                raise VideoProcessingError(
                    "Video file appears to be corrupted (too small)"
                )

            self.logger.info(
                f"Video format validation passed: {metadata.format}/{metadata.codec}"
            )
            return True

        except VideoProcessingError:
            raise  # Re-raise VideoProcessingError as-is

        except Exception as e:
            raise VideoProcessingError(
                f"Unexpected error during format validation: {e}"
            )

    def get_duration(self) -> float:
        """Get video duration in seconds.

        Returns:
            Duration in seconds
        """
        metadata = self.extract_metadata()
        return metadata.duration

    def get_frame_count(self) -> int:
        """Get total frame count.

        Returns:
            Total number of frames in video
        """
        metadata = self.extract_metadata()
        return metadata.total_frames

    def get_video_info_summary(self) -> Dict[str, Any]:
        """Get comprehensive video information summary.

        Returns:
            Dictionary with detailed video information
        """
        metadata = self.extract_metadata()
        file_hash = self.calculate_hash()

        return {
            "file_path": str(self.video_path),
            "file_name": self.video_path.name,
            "file_size_mb": metadata.file_size_bytes / (1024 * 1024),
            "file_hash": file_hash,
            "duration_seconds": metadata.duration,
            "fps": metadata.fps,
            "resolution": metadata.resolution_string,
            "width": metadata.width,
            "height": metadata.height,
            "aspect_ratio": metadata.aspect_ratio,
            "codec": metadata.codec,
            "format": metadata.format,
            "total_frames": metadata.total_frames,
            "estimated_processing_time": self._estimate_processing_time(metadata),
        }

    def _estimate_processing_time(self, metadata: VideoMetadata) -> Dict[str, float]:
        """Estimate processing time based on video characteristics.

        Args:
            metadata: Video metadata

        Returns:
            Dictionary with time estimates for different processing steps
        """
        # Base processing rate estimates (frames per second)
        base_rates = {
            "frame_extraction": 100.0,  # frames/second
            "face_detection": 10.0,  # frames/second
            "pose_analysis": 5.0,  # frames/second
            "quality_assessment": 20.0,  # frames/second
        }

        # Adjust based on resolution
        resolution_factor = (metadata.width * metadata.height) / (
            1920 * 1080
        )  # Relative to 1080p

        estimates = {}
        for step, base_rate in base_rates.items():
            adjusted_rate = base_rate / max(1.0, resolution_factor)
            estimated_seconds = metadata.total_frames / adjusted_rate
            estimates[step] = estimated_seconds

        estimates["total_estimated"] = sum(estimates.values())

        return estimates
