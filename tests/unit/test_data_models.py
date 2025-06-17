"""Unit tests for data model classes."""

import tempfile
from pathlib import Path

from personfromvid.data.detection_results import (
    FaceDetection,
    HeadPoseResult,
    PoseDetection,
    QualityMetrics,
)
from personfromvid.data.frame_data import (
    FrameData,
    ImageProperties,
    SelectionInfo,
    SourceInfo,
)


class TestFaceDetection:
    """Tests for FaceDetection dataclass."""

    def test_basic_creation(self):
        """Test basic face detection creation."""
        face = FaceDetection(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            landmarks=[(120, 120), (180, 120), (150, 150), (130, 180), (170, 180)]
        )

        assert face.bbox == (100, 100, 200, 200)
        assert face.confidence == 0.95
        assert len(face.landmarks) == 5

    def test_creation_without_landmarks(self):
        """Test face detection without landmarks."""
        face = FaceDetection(
            bbox=(100, 100, 200, 200),
            confidence=0.85
        )

        assert face.bbox == (100, 100, 200, 200)
        assert face.confidence == 0.85
        assert face.landmarks is None

    def test_bbox_properties(self):
        """Test bounding box property calculations."""
        face = FaceDetection(
            bbox=(100, 150, 300, 350),
            confidence=0.9
        )

        # Test calculated properties
        width = face.bbox[2] - face.bbox[0]
        height = face.bbox[3] - face.bbox[1]
        area = width * height

        assert width == 200
        assert height == 200
        assert area == 40000

    def test_high_confidence_detection(self):
        """Test high confidence detection."""
        face = FaceDetection(
            bbox=(50, 50, 150, 150),
            confidence=0.99
        )

        assert face.confidence > 0.9
        assert face.confidence <= 1.0

    def test_edge_case_bbox(self):
        """Test edge case bounding boxes."""
        # Minimum size bbox
        face = FaceDetection(
            bbox=(0, 0, 1, 1),
            confidence=0.5
        )

        assert face.bbox == (0, 0, 1, 1)

        # Large bbox
        face_large = FaceDetection(
            bbox=(0, 0, 1920, 1080),
            confidence=0.8
        )

        assert face_large.bbox == (0, 0, 1920, 1080)


class TestPoseDetection:
    """Tests for PoseDetection dataclass."""

    def test_basic_creation(self):
        """Test basic pose detection creation."""
        keypoints = {
            'nose': (500.0, 200.0, 0.9),
            'left_shoulder': (450.0, 300.0, 0.85),
            'right_shoulder': (550.0, 300.0, 0.88),
            'left_hip': (480.0, 500.0, 0.82),
            'right_hip': (520.0, 500.0, 0.84)
        }

        pose = PoseDetection(
            bbox=(400, 150, 600, 700),
            confidence=0.92,
            keypoints=keypoints,
            pose_classifications=[("standing", 0.9)]
        )

        assert pose.bbox == (400, 150, 600, 700)
        assert pose.confidence == 0.92
        assert len(pose.keypoints) == 5
        assert pose.pose_classifications == [("standing", 0.9)]

    def test_creation_without_classification(self):
        """Test pose detection without classification."""
        keypoints = {
            'nose': (500.0, 200.0, 0.9),
            'left_shoulder': (450.0, 300.0, 0.85)
        }

        pose = PoseDetection(
            bbox=(400, 150, 600, 700),
            confidence=0.8,
            keypoints=keypoints
        )

        assert not pose.pose_classifications

    def test_keypoint_confidence_filtering(self):
        """Test filtering keypoints by confidence."""
        keypoints = {
            'nose': (500.0, 200.0, 0.9),        # High confidence
            'left_shoulder': (450.0, 300.0, 0.3),  # Low confidence
            'right_shoulder': (550.0, 300.0, 0.85), # High confidence
        }

        pose = PoseDetection(
            bbox=(400, 150, 600, 700),
            confidence=0.8,
            keypoints=keypoints
        )

        # Filter high confidence keypoints (>0.5)
        high_conf_keypoints = {k: v for k, v in pose.keypoints.items() if v[2] > 0.5}
        assert len(high_conf_keypoints) == 2
        assert 'left_shoulder' not in high_conf_keypoints

    def test_pose_classifications(self):
        """Test different pose classifications."""
        classifications = [
            [("standing", 0.9)],
            [("sitting", 0.85)],
            [("squatting", 0.92), ("closeup", 0.8)],
        ]

        for classification_list in classifications:
            pose = PoseDetection(
                bbox=(400, 150, 600, 700),
                confidence=0.8,
                keypoints={'nose': (500.0, 200.0, 0.9)},
                pose_classifications=classification_list
            )

            assert pose.pose_classifications == classification_list

    def test_empty_keypoints(self):
        """Test pose detection with no keypoints."""
        pose = PoseDetection(
            bbox=(400, 150, 600, 700),
            confidence=0.5,
            keypoints={}
        )

        assert len(pose.keypoints) == 0
        assert not pose.pose_classifications


class TestHeadPoseResult:
    """Tests for HeadPoseResult dataclass."""

    def test_basic_creation(self):
        """Test basic head pose result creation."""
        head_pose = HeadPoseResult(
            yaw=15.5,
            pitch=-5.2,
            roll=2.1,
            confidence=0.88,
            direction="looking_left"
        )

        assert head_pose.yaw == 15.5
        assert head_pose.pitch == -5.2
        assert head_pose.roll == 2.1
        assert head_pose.confidence == 0.88
        assert head_pose.direction == "looking_left"

    def test_extreme_angles(self):
        """Test extreme angle values."""
        # Test maximum rotation angles
        extreme_pose = HeadPoseResult(
            yaw=180.0,
            pitch=90.0,
            roll=-90.0,
            confidence=0.7,
            direction="profile_left"
        )

        assert extreme_pose.yaw == 180.0
        assert extreme_pose.pitch == 90.0
        assert extreme_pose.roll == -90.0

    def test_direction_classifications(self):
        """Test different head direction classifications."""
        directions = [
            "front", "looking_left", "looking_right",
            "profile_left", "profile_right", "looking_up",
            "looking_down", "looking_up_left", "looking_up_right"
        ]

        for direction in directions:
            head_pose = HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                confidence=0.9,
                direction=direction
            )

            assert head_pose.direction == direction

    def test_low_confidence_detection(self):
        """Test low confidence head pose detection."""
        low_conf_pose = HeadPoseResult(
            yaw=45.0,
            pitch=0.0,
            roll=0.0,
            confidence=0.3,  # Low confidence
            direction="looking_left"
        )

        assert low_conf_pose.confidence < 0.5
        # Low confidence detections might be filtered out in practice

    def test_near_zero_angles(self):
        """Test near-zero angle values."""
        front_pose = HeadPoseResult(
            yaw=0.1,
            pitch=-0.2,
            roll=0.0,
            confidence=0.95,
            direction="front"
        )

        # Small angles should still be valid
        assert abs(front_pose.yaw) < 1.0
        assert abs(front_pose.pitch) < 1.0
        assert front_pose.roll == 0.0


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic quality metrics creation."""
        metrics = QualityMetrics(
            laplacian_variance=1250.5,
            sobel_variance=890.2,
            brightness_score=128.0,
            contrast_score=0.75,
            overall_quality=0.82
        )

        assert metrics.laplacian_variance == 1250.5
        assert metrics.sobel_variance == 890.2
        assert metrics.brightness_score == 128.0
        assert metrics.contrast_score == 0.75
        assert metrics.overall_quality == 0.82

    def test_high_quality_metrics(self):
        """Test high quality image metrics."""
        high_quality = QualityMetrics(
            laplacian_variance=2500.0,  # High sharpness
            sobel_variance=1800.0,      # High edge definition
            brightness_score=140.0,     # Good brightness
            contrast_score=0.9,         # High contrast
            overall_quality=0.95        # Excellent overall
        )

        assert high_quality.overall_quality > 0.9
        assert high_quality.laplacian_variance > 2000
        assert high_quality.contrast_score > 0.8

    def test_low_quality_metrics(self):
        """Test low quality image metrics."""
        low_quality = QualityMetrics(
            laplacian_variance=50.0,    # Low sharpness (blurry)
            sobel_variance=30.0,        # Poor edge definition
            brightness_score=50.0,      # Too dark
            contrast_score=0.2,         # Low contrast
            overall_quality=0.25        # Poor overall
        )

        assert low_quality.overall_quality < 0.5
        assert low_quality.laplacian_variance < 100
        assert low_quality.contrast_score < 0.5

    def test_extreme_values(self):
        """Test extreme quality metric values."""
        # Test with extreme values that might occur
        extreme_metrics = QualityMetrics(
            laplacian_variance=0.0,     # Completely blurred
            sobel_variance=10000.0,     # Very high edge content
            brightness_score=255.0,     # Maximum brightness
            contrast_score=1.0,         # Maximum contrast
            overall_quality=0.0         # Worst possible quality
        )

        assert extreme_metrics.laplacian_variance == 0.0
        assert extreme_metrics.sobel_variance == 10000.0
        assert extreme_metrics.overall_quality == 0.0

    def test_quality_score_normalization(self):
        """Test quality score normalization."""
        # Overall quality should typically be between 0 and 1
        metrics = QualityMetrics(
            laplacian_variance=1000.0,
            sobel_variance=800.0,
            brightness_score=120.0,
            contrast_score=0.7,
            overall_quality=0.73
        )

        assert 0.0 <= metrics.overall_quality <= 1.0
        assert 0.0 <= metrics.contrast_score <= 1.0


class TestSourceInfo:
    """Tests for SourceInfo dataclass."""

    def test_basic_creation(self):
        """Test basic source info creation."""
        source = SourceInfo(
            video_timestamp=45.67,
            extraction_method="i_frame",
            original_frame_number=1234,
            video_fps=30.0
        )

        assert source.video_timestamp == 45.67
        assert source.extraction_method == "i_frame"
        assert source.original_frame_number == 1234
        assert source.video_fps == 30.0

    def test_different_extraction_methods(self):
        """Test different frame extraction methods."""
        methods = ["i_frame", "temporal_sampling", "keyframe", "manual"]

        for method in methods:
            source = SourceInfo(
                video_timestamp=30.0,
                extraction_method=method,
                original_frame_number=900,
                video_fps=25.0
            )

            assert source.extraction_method == method

    def test_edge_timestamps(self):
        """Test edge case timestamps."""
        # Start of video
        start_source = SourceInfo(
            video_timestamp=0.0,
            extraction_method="temporal_sampling",
            original_frame_number=0,
            video_fps=30.0
        )

        assert start_source.video_timestamp == 0.0

        # Long video timestamp
        long_source = SourceInfo(
            video_timestamp=7200.5,  # 2 hours
            extraction_method="i_frame",
            original_frame_number=216015,  # 30fps * 2hrs
            video_fps=30.0
        )

        assert long_source.video_timestamp == 7200.5


class TestImageProperties:
    """Tests for ImageProperties dataclass."""

    def test_basic_creation(self):
        """Test basic image properties creation."""
        props = ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=245760,
            format="JPEG"
        )

        assert props.width == 1920
        assert props.height == 1080
        assert props.channels == 3
        assert props.file_size_bytes == 245760
        assert props.format == "JPEG"

    def test_different_resolutions(self):
        """Test different image resolutions."""
        resolutions = [
            (640, 480),    # VGA
            (1280, 720),   # 720p
            (1920, 1080),  # 1080p
            (3840, 2160),  # 4K
        ]

        for width, height in resolutions:
            props = ImageProperties(
                width=width,
                height=height,
                channels=3,
                file_size_bytes=width * height * 3,
                format="JPEG"
            )

            assert props.width == width
            assert props.height == height
            # Test aspect ratio calculation
            aspect_ratio = width / height
            assert aspect_ratio > 0

    def test_different_formats(self):
        """Test different image formats."""
        formats = ["JPEG", "PNG", "BMP", "TIFF"]

        for fmt in formats:
            props = ImageProperties(
                width=800,
                height=600,
                channels=3,
                file_size_bytes=800 * 600 * 3,
                format=fmt
            )

            assert props.format == fmt

    def test_grayscale_image(self):
        """Test grayscale image properties."""
        grayscale = ImageProperties(
            width=640,
            height=480,
            channels=1,  # Grayscale
            file_size_bytes=640 * 480,
            format="JPEG"
        )

        assert grayscale.channels == 1
        assert grayscale.file_size_bytes == 640 * 480

    def test_rgba_image(self):
        """Test RGBA image properties."""
        rgba = ImageProperties(
            width=512,
            height=512,
            channels=4,  # RGBA
            file_size_bytes=512 * 512 * 4,
            format="PNG"
        )

        assert rgba.channels == 4
        assert rgba.format == "PNG"  # PNG supports alpha


class TestSelectionInfo:
    """Tests for SelectionInfo dataclass."""

    def test_basic_creation(self):
        """Test basic selection info creation."""
        selection = SelectionInfo(
            selected_for_poses=["standing", "sitting"],
            selected_for_head_angles=["front", "looking_left"],
            final_output=True,
            output_files=["video_standing_001.jpg", "video_face_front_001.jpg"],
            crop_regions={"face_crop": (100, 50, 300, 250)}
        )

        assert "standing" in selection.selected_for_poses
        assert "front" in selection.selected_for_head_angles
        assert selection.final_output is True
        assert len(selection.output_files) == 2
        assert "face_crop" in selection.crop_regions

    def test_no_selections(self):
        """Test selection info with no selections."""
        no_selection = SelectionInfo(
            selected_for_poses=[],
            selected_for_head_angles=[],
            final_output=False,
            output_files=[],
            crop_regions={}
        )

        assert len(no_selection.selected_for_poses) == 0
        assert len(no_selection.selected_for_head_angles) == 0
        assert no_selection.final_output is False
        assert len(no_selection.output_files) == 0

    def test_multiple_crop_regions(self):
        """Test multiple crop regions."""
        multi_crop = SelectionInfo(
            selected_for_poses=["closeup"],
            selected_for_head_angles=["front", "profile_left"],
            final_output=True,
            output_files=["video_closeup_001.jpg"],
            crop_regions={
                "face_crop_front": (100, 50, 300, 250),
                "face_crop_profile": (150, 75, 350, 275),
                "body_crop": (50, 0, 400, 500)
            }
        )

        assert len(multi_crop.crop_regions) == 3
        assert "face_crop_front" in multi_crop.crop_regions
        assert "body_crop" in multi_crop.crop_regions


class TestFrameData:
    """Tests for FrameData dataclass integration."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.frame_file = self.temp_dir / "frame_001.jpg"
        self.frame_file.write_text("fake frame data")

    def test_complete_frame_data(self):
        """Test complete frame data with all components."""
        # Create sample data
        source = SourceInfo(
            video_timestamp=30.5,
            extraction_method="i_frame",
            original_frame_number=915,
            video_fps=30.0
        )

        image_props = ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=self.frame_file.stat().st_size,
            format="JPEG"
        )

        face_detection = FaceDetection(
            bbox=(500, 300, 700, 500),
            confidence=0.92,
            landmarks=[(520, 320), (680, 320), (600, 380), (540, 450), (660, 450)]
        )

        pose_detection = PoseDetection(
            bbox=(400, 200, 800, 900),
            confidence=0.88,
            keypoints={
                'nose': (600.0, 320.0, 0.9),
                'left_shoulder': (520.0, 420.0, 0.85),
                'right_shoulder': (680.0, 420.0, 0.87)
            },
            pose_classifications=[("standing", 0.88)]
        )

        head_pose = HeadPoseResult(
            yaw=10.5,
            pitch=-5.2,
            roll=1.8,
            confidence=0.89,
            direction="looking_left"
        )

        quality = QualityMetrics(
            laplacian_variance=1450.2,
            sobel_variance=920.5,
            brightness_score=135.0,
            contrast_score=0.78,
            overall_quality=0.84
        )

        selection = SelectionInfo(
            selected_for_poses=["standing"],
            selected_for_head_angles=["looking_left"],
            final_output=True,
            output_files=["video_standing_001.jpg", "video_face_looking_left_001.jpg"],
            crop_regions={"face_crop": (500, 300, 700, 500)}
        )

        # Create complete frame data
        frame = FrameData(
            frame_id="frame_001",
            file_path=self.frame_file,
            source_info=source,
            image_properties=image_props,
            face_detections=[face_detection],
            pose_detections=[pose_detection],
            head_poses=[head_pose],
            quality_metrics=quality,
            selections=selection
        )

        # Verify all components
        assert frame.frame_id == "frame_001"
        assert len(frame.face_detections) == 1
        assert len(frame.pose_detections) == 1
        assert len(frame.head_poses) == 1
        assert frame.quality_metrics.overall_quality == 0.84
        assert frame.selections.final_output is True

    def test_minimal_frame_data(self):
        """Test minimal frame data with required fields only."""
        frame = FrameData(
            frame_id="minimal_frame",
            file_path=self.frame_file,
            source_info=SourceInfo(
                video_timestamp=15.0,
                extraction_method="temporal_sampling",
                original_frame_number=450,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=640,
                height=480,
                channels=3,
                file_size_bytes=100000,
                format="JPEG"
            ),
            face_detections=[],
            pose_detections=[],
            head_poses=[],
            quality_metrics=None,
            selections=SelectionInfo([], [], False, [], {})
        )

        assert frame.frame_id == "minimal_frame"
        assert len(frame.face_detections) == 0
        assert len(frame.pose_detections) == 0
        assert frame.quality_metrics is None
        assert frame.selections.final_output is False

    def test_multiple_detections(self):
        """Test frame with multiple face and pose detections."""
        # Multiple faces in frame
        faces = [
            FaceDetection((100, 100, 200, 200), 0.9),
            FaceDetection((300, 150, 400, 250), 0.85),
            FaceDetection((500, 200, 600, 300), 0.88)
        ]

        # Multiple poses in frame
        poses = [
            PoseDetection((80, 50, 220, 400), 0.92, {'nose': (150, 120, 0.9)}, [("standing", 0.92)]),
            PoseDetection((280, 100, 420, 450), 0.87, {'nose': (350, 170, 0.85)}, [("sitting", 0.87), ("closeup", 0.8)])
        ]

        frame = FrameData(
            frame_id="multi_person",
            file_path=self.frame_file,
            source_info=SourceInfo(60.0, "i_frame", 1800, 30.0),
            image_properties=ImageProperties(1920, 1080, 3, 200000, "JPEG"),
            face_detections=faces,
            pose_detections=poses,
            head_poses=[],
            quality_metrics=QualityMetrics(1200, 800, 130, 0.7, 0.75),
            selections=SelectionInfo([], [], False, [], {})
        )

        assert len(frame.face_detections) == 3
        assert len(frame.pose_detections) == 2

        # Test filtering high confidence detections
        high_conf_faces = [f for f in frame.face_detections if f.confidence > 0.87]
        assert len(high_conf_faces) == 2

    def test_get_pose_classifications(self):
        """Test retrieval of unique pose classifications from multiple detections."""
        poses = [
            PoseDetection((80, 50, 220, 400), 0.92, {}, [("standing", 0.92), ("closeup", 0.85)]),
            PoseDetection((280, 100, 420, 450), 0.87, {}, [("sitting", 0.87), ("closeup", 0.90)])
        ]

        frame = FrameData(
            frame_id="multi_class",
            file_path=self.frame_file,
            source_info=SourceInfo(60.0, "i_frame", 1800, 30.0),
            image_properties=ImageProperties(1920, 1080, 3, 200000, "JPEG"),
            pose_detections=poses
        )

        classifications = frame.get_pose_classifications()
        assert len(classifications) == 3
        assert "standing" in classifications
        assert "sitting" in classifications
        assert "closeup" in classifications
