"""Frame extraction engine for keyframe detection and temporal sampling.

This module implements the FrameExtractor class for extracting keyframes from videos
using a hybrid approach combining I-frame detection and temporal sampling.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import cv2
import ffmpeg
import numpy as np

from ..data import VideoMetadata, FrameData, SourceInfo, ImageProperties
from ..utils.logging import get_logger
from ..utils.exceptions import VideoProcessingError


class ExtractionMethod(Enum):
    """Frame extraction method types."""

    I_FRAME = "i_frame"
    TEMPORAL_SAMPLING = "temporal_sampling"


@dataclass
class FrameCandidate:
    """Candidate frame for extraction."""

    timestamp: float  # Time in seconds
    frame_number: int  # Original frame number in video
    method: ExtractionMethod  # How this frame was selected
    confidence: float = 1.0  # Confidence in frame quality/importance


class FrameExtractor:
    """Frame extraction engine for keyframe detection and temporal sampling.

    Implements hybrid extraction strategy:
    1. I-frame detection using FFmpeg metadata
    2. Temporal sampling at 0.25-second intervals
    3. Frame deduplication to avoid redundancy
    """

    def __init__(self, video_path: str, video_metadata: VideoMetadata):
        """Initialize frame extractor.

        Args:
            video_path: Path to video file
            video_metadata: Video metadata from VideoProcessor
        """
        self.video_path = Path(video_path)
        self.video_metadata = video_metadata
        self.logger = get_logger("frame_extractor")

        # Extraction configuration
        self.temporal_interval = 0.25  # Sample every 0.25 seconds
        self.similarity_threshold = 0.95  # For frame deduplication
        self.max_frames_per_second = 8  # Limit to prevent excessive extraction

        # Processing state
        self.extracted_frames: List[FrameData] = []
        self.frame_hashes: Set[str] = set()  # For deduplication

        # Statistics
        self.stats = {
            "i_frames_found": 0,
            "temporal_samples": 0,
            "duplicates_removed": 0,
            "total_extracted": 0,
            "processing_time": 0.0,
        }

    def extract_frames(
        self, output_dir: Path, progress_callback: Optional[callable] = None
    ) -> List[FrameData]:
        """Extract keyframes using hybrid approach.

        Args:
            output_dir: Directory to save extracted frames
            progress_callback: Optional callback for progress updates

        Returns:
            List of FrameData objects for extracted frames

        Raises:
            VideoProcessingError: If frame extraction fails
        """
        start_time = time.time()
        self.logger.info(f"Starting frame extraction from: {self.video_path}")

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Extract I-frames using FFmpeg metadata
            i_frame_candidates = self._extract_i_frames()
            self.logger.info(f"Found {len(i_frame_candidates)} I-frame candidates")

            # Step 2: Generate temporal sampling candidates
            temporal_candidates = self._generate_temporal_samples()
            self.logger.info(
                f"Generated {len(temporal_candidates)} temporal sampling candidates"
            )

            # Step 3: Combine and deduplicate candidates
            all_candidates = self._combine_and_deduplicate_candidates(
                i_frame_candidates, temporal_candidates
            )
            self.logger.info(
                f"Combined to {len(all_candidates)} unique frame candidates"
            )

            # Step 4: Extract actual frame images
            extracted_frames = self._extract_frame_images(
                all_candidates, output_dir, progress_callback
            )

            # Update statistics
            self.stats["total_extracted"] = len(extracted_frames)
            self.stats["processing_time"] = time.time() - start_time

            self.logger.info(
                f"Frame extraction completed: {len(extracted_frames)} frames "
                f"extracted in {self.stats['processing_time']:.1f}s"
            )

            self.extracted_frames = extracted_frames
            return extracted_frames

        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            raise VideoProcessingError(f"Frame extraction failed: {e}")

    def _extract_i_frames(self) -> List[FrameCandidate]:
        """Extract I-frame timestamps using FFmpeg.

        Returns:
            List of FrameCandidate objects for I-frames
        """
        self.logger.debug("Extracting I-frame positions using FFmpeg")

        try:
            # Use ffprobe to get frame information
            # Focus on keyframes (I-frames) which are compression-optimal
            probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",  # Video stream only
                "-show_frames",
                "-show_entries",
                "frame=pkt_pts_time,pict_type",
                "-of",
                "json",
                str(self.video_path),
            ]

            import subprocess

            result = subprocess.run(probe_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.warning(f"FFprobe failed: {result.stderr}")
                return []

            # Parse JSON output
            probe_data = json.loads(result.stdout)
            frames_data = probe_data.get("frames", [])

            i_frame_candidates = []

            for frame_data in frames_data:
                # Look for I-frames (keyframes)
                if frame_data.get("pict_type") == "I":
                    timestamp = float(frame_data.get("pkt_pts_time", 0))

                    # Skip frames outside video duration
                    if timestamp > self.video_metadata.duration:
                        continue

                    frame_number = int(timestamp * self.video_metadata.fps)

                    candidate = FrameCandidate(
                        timestamp=timestamp,
                        frame_number=frame_number,
                        method=ExtractionMethod.I_FRAME,
                        confidence=1.0,  # I-frames are high confidence
                    )

                    i_frame_candidates.append(candidate)

            self.stats["i_frames_found"] = len(i_frame_candidates)
            return i_frame_candidates

        except Exception as e:
            self.logger.warning(
                f"I-frame extraction failed: {e}, falling back to temporal sampling only"
            )
            return []

    def _generate_temporal_samples(self) -> List[FrameCandidate]:
        """Generate temporal sampling candidates at fixed intervals.

        Returns:
            List of FrameCandidate objects for temporal samples
        """
        self.logger.debug(
            f"Generating temporal samples every {self.temporal_interval}s"
        )

        temporal_candidates = []
        current_time = 0.0

        while current_time < self.video_metadata.duration:
            frame_number = int(current_time * self.video_metadata.fps)

            # Skip if frame number exceeds total frames
            if frame_number >= self.video_metadata.total_frames:
                break

            candidate = FrameCandidate(
                timestamp=current_time,
                frame_number=frame_number,
                method=ExtractionMethod.TEMPORAL_SAMPLING,
                confidence=0.8,  # Lower confidence than I-frames
            )

            temporal_candidates.append(candidate)
            current_time += self.temporal_interval

        self.stats["temporal_samples"] = len(temporal_candidates)
        return temporal_candidates

    def _combine_and_deduplicate_candidates(
        self, i_frames: List[FrameCandidate], temporal: List[FrameCandidate]
    ) -> List[FrameCandidate]:
        """Combine I-frame and temporal candidates, removing duplicates.

        Args:
            i_frames: I-frame candidates
            temporal: Temporal sampling candidates

        Returns:
            Deduplicated list of frame candidates
        """
        self.logger.debug("Combining and deduplicating frame candidates")

        # Combine all candidates
        all_candidates = i_frames + temporal

        # Sort by timestamp
        all_candidates.sort(key=lambda x: x.timestamp)

        # Remove near-duplicates (within 0.1 seconds)
        deduplicated = []
        last_timestamp = -1.0
        duplicate_threshold = 0.1  # 100ms threshold

        for candidate in all_candidates:
            if candidate.timestamp - last_timestamp >= duplicate_threshold:
                deduplicated.append(candidate)
                last_timestamp = candidate.timestamp
            else:
                # Keep the candidate with higher confidence
                if deduplicated and candidate.confidence > deduplicated[-1].confidence:
                    deduplicated[-1] = candidate
                self.stats["duplicates_removed"] += 1

        # Apply max frames per second limit
        if (
            len(deduplicated)
            > self.video_metadata.duration * self.max_frames_per_second
        ):
            # Keep frames with highest confidence, spread across time
            target_count = int(
                self.video_metadata.duration * self.max_frames_per_second
            )

            # Sort by confidence (descending) then by timestamp
            deduplicated.sort(key=lambda x: (-x.confidence, x.timestamp))
            deduplicated = deduplicated[:target_count]

            # Re-sort by timestamp
            deduplicated.sort(key=lambda x: x.timestamp)

            self.logger.info(
                f"Limited to {target_count} frames (max {self.max_frames_per_second}/sec)"
            )

        return deduplicated

    def _extract_frame_images(
        self,
        candidates: List[FrameCandidate],
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> List[FrameData]:
        """Extract actual frame images from video.

        Args:
            candidates: Frame candidates to extract
            output_dir: Directory to save frame images
            progress_callback: Optional progress callback

        Returns:
            List of FrameData objects for successfully extracted frames
        """
        self.logger.debug(f"Extracting {len(candidates)} frame images")

        extracted_frames = []

        # Open video capture
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {self.video_path}")

        try:
            total_candidates = len(candidates)

            for i, candidate in enumerate(candidates):
                try:
                    # Seek to specific timestamp
                    cap.set(cv2.CAP_PROP_POS_MSEC, candidate.timestamp * 1000)

                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning(
                            f"Could not read frame at {candidate.timestamp}s"
                        )
                        continue

                    # Generate frame ID and filename
                    frame_id = f"frame_{candidate.frame_number:06d}"
                    filename = f"{frame_id}.png"
                    frame_path = output_dir / filename

                    # Check for duplicate frames using perceptual hash
                    frame_hash = self._calculate_frame_hash(frame)
                    if frame_hash in self.frame_hashes:
                        self.logger.debug(f"Skipping duplicate frame: {frame_id}")
                        self.stats["duplicates_removed"] += 1
                        continue

                    self.frame_hashes.add(frame_hash)

                    # Save frame as PNG with maximum compression
                    success = cv2.imwrite(
                        str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 4]
                    )

                    if not success:
                        self.logger.warning(f"Failed to save frame: {frame_path}")
                        continue

                    # Create frame metadata
                    frame_data = self._create_frame_data(
                        candidate, frame, frame_path, frame_id
                    )

                    extracted_frames.append(frame_data)

                    # Progress callback
                    if progress_callback:
                        progress_callback(i + 1, total_candidates)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract frame at {candidate.timestamp}s: {e}"
                    )
                    continue

        finally:
            cap.release()

        return extracted_frames

    def _calculate_frame_hash(self, frame: np.ndarray) -> str:
        """Calculate perceptual hash for frame deduplication.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            Hash string for frame comparison
        """
        # Convert to grayscale and resize to small size for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16))

        # Calculate simple hash
        return hashlib.md5(resized.tobytes()).hexdigest()

    def _create_frame_data(
        self,
        candidate: FrameCandidate,
        frame: np.ndarray,
        frame_path: Path,
        frame_id: str,
    ) -> FrameData:
        """Create FrameData object for extracted frame.

        Args:
            candidate: Frame candidate information
            frame: OpenCV frame array
            frame_path: Path to saved frame file
            frame_id: Unique frame identifier

        Returns:
            FrameData object with complete frame metadata
        """
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        file_size = frame_path.stat().st_size if frame_path.exists() else 0

        source_info = SourceInfo(
            video_timestamp=candidate.timestamp,
            extraction_method=candidate.method.value,
            original_frame_number=candidate.frame_number,
            video_fps=self.video_metadata.fps,
        )

        image_properties = ImageProperties(
            width=width,
            height=height,
            channels=channels,
            file_size_bytes=file_size,
            format="PNG",
        )

        return FrameData(
            frame_id=frame_id,
            file_path=frame_path,
            source_info=source_info,
            image_properties=image_properties,
            # Other fields have default factories and will be populated later
        )

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get detailed extraction statistics.

        Returns:
            Dictionary with extraction statistics and metrics
        """
        return {
            "total_candidates_considered": (
                self.stats["i_frames_found"] + self.stats["temporal_samples"]
            ),
            "i_frames_found": self.stats["i_frames_found"],
            "temporal_samples_generated": self.stats["temporal_samples"],
            "duplicates_removed": self.stats["duplicates_removed"],
            "frames_extracted": self.stats["total_extracted"],
            "extraction_rate": (
                self.stats["total_extracted"] / self.stats["processing_time"]
                if self.stats["processing_time"] > 0
                else 0
            ),
            "processing_time_seconds": self.stats["processing_time"],
            "coverage_percentage": (
                (self.stats["total_extracted"] * self.temporal_interval)
                / self.video_metadata.duration
                * 100
                if self.video_metadata.duration > 0
                else 0
            ),
            "average_interval_seconds": (
                self.video_metadata.duration / self.stats["total_extracted"]
                if self.stats["total_extracted"] > 0
                else 0
            ),
        }

    def cleanup_temp_frames(self, keep_selected: List[str] = None) -> None:
        """Clean up temporary frame files, optionally keeping selected ones.

        Args:
            keep_selected: List of frame IDs to keep, delete others
        """
        if keep_selected is None:
            keep_selected = []

        frames_to_delete = []
        for frame_data in self.extracted_frames:
            if frame_data.frame_id not in keep_selected:
                frames_to_delete.append(frame_data.file_path)

        deleted_count = 0
        for frame_path in frames_to_delete:
            try:
                if frame_path.exists():
                    frame_path.unlink()
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to delete temp frame {frame_path}: {e}")

        self.logger.info(f"Cleaned up {deleted_count} temporary frame files")
