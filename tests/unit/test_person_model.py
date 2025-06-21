"""Unit tests for Person model components."""

import pytest

from personfromvid.data.detection_results import (
    FaceDetection,
    HeadPoseResult,
    PoseDetection,
)
from personfromvid.data.person import BodyUnknown, FaceUnknown, Person, PersonQuality


class TestPersonQuality:
    """Tests for PersonQuality dataclass."""

    def test_person_quality_creation(self):
        """Test PersonQuality creation with valid inputs."""
        quality = PersonQuality(face_quality=0.8, body_quality=0.6)

        assert quality.face_quality == 0.8
        assert quality.body_quality == 0.6
        # Test weighted calculation: 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert abs(quality.overall_quality - 0.74) < 0.001

    def test_person_quality_validation(self):
        """Test PersonQuality validation for invalid inputs."""
        # Test invalid face quality
        with pytest.raises(ValueError, match="face_quality must be between 0.0 and 1.0"):
            PersonQuality(face_quality=1.5, body_quality=0.5)

        with pytest.raises(ValueError, match="face_quality must be between 0.0 and 1.0"):
            PersonQuality(face_quality=-0.1, body_quality=0.5)

        # Test invalid body quality
        with pytest.raises(ValueError, match="body_quality must be between 0.0 and 1.0"):
            PersonQuality(face_quality=0.5, body_quality=1.5)

        with pytest.raises(ValueError, match="body_quality must be between 0.0 and 1.0"):
            PersonQuality(face_quality=0.5, body_quality=-0.1)

    def test_person_quality_properties(self):
        """Test PersonQuality property methods."""
        # Test high quality
        high_quality = PersonQuality(face_quality=0.9, body_quality=0.8)
        assert high_quality.is_high_quality
        assert high_quality.is_usable

        # Test medium quality
        medium_quality = PersonQuality(face_quality=0.5, body_quality=0.4)
        assert not medium_quality.is_high_quality
        assert medium_quality.is_usable

        # Test low quality
        low_quality = PersonQuality(face_quality=0.2, body_quality=0.1)
        assert not low_quality.is_high_quality
        assert not low_quality.is_usable

    def test_person_quality_factors(self):
        """Test PersonQuality quality_factors property."""
        quality = PersonQuality(face_quality=0.7, body_quality=0.5)
        factors = quality.quality_factors

        assert factors["face_quality"] == 0.7
        assert factors["body_quality"] == 0.5
        assert abs(factors["overall_quality"] - 0.64) < 0.001

    def test_person_quality_serialization(self):
        """Test PersonQuality serialization and deserialization."""
        quality = PersonQuality(face_quality=0.8, body_quality=0.6)

        # Test to_dict
        quality_dict = quality.to_dict()
        expected_dict = {
            "face_quality": 0.8,
            "body_quality": 0.6,
            "overall_quality": 0.74,
        }
        for key, value in expected_dict.items():
            if isinstance(value, float):
                assert abs(quality_dict[key] - value) < 0.001
            else:
                assert quality_dict[key] == value

        # Test from_dict
        reconstructed = PersonQuality.from_dict(quality_dict)
        assert reconstructed.face_quality == quality.face_quality
        assert reconstructed.body_quality == quality.body_quality
        assert abs(reconstructed.overall_quality - quality.overall_quality) < 0.001


class TestSentinelObjects:
    """Tests for FaceUnknown and BodyUnknown sentinel classes."""

    def test_face_unknown_singleton(self):
        """Test FaceUnknown singleton behavior."""
        face1 = FaceUnknown()
        face2 = FaceUnknown()

        assert face1 is face2  # Same instance
        assert isinstance(face1, FaceDetection)
        assert face1.confidence == 0.0
        assert face1.bbox == (0, 0, 0, 0)

    def test_body_unknown_singleton(self):
        """Test BodyUnknown singleton behavior."""
        body1 = BodyUnknown()
        body2 = BodyUnknown()

        assert body1 is body2  # Same instance
        assert isinstance(body1, PoseDetection)
        assert body1.confidence == 0.0
        assert body1.bbox == (0, 0, 0, 0)
        assert body1.keypoints == {}

    def test_sentinel_inheritance(self):
        """Test that sentinels inherit from detection classes."""
        face_unknown = FaceUnknown()
        body_unknown = BodyUnknown()

        assert isinstance(face_unknown, FaceDetection)
        assert isinstance(body_unknown, PoseDetection)

        # Test they have expected methods
        assert hasattr(face_unknown, 'center')
        assert hasattr(body_unknown, 'has_keypoint')


class TestPerson:
    """Tests for Person dataclass."""

    def create_test_face(self, confidence=0.8):
        """Helper to create test FaceDetection."""
        return FaceDetection(
            bbox=(100, 100, 200, 200),
            confidence=confidence,
            landmarks=[(150, 120), (150, 140)]
        )

    def create_test_body(self, confidence=0.7):
        """Helper to create test PoseDetection."""
        return PoseDetection(
            bbox=(80, 50, 220, 400),
            confidence=confidence,
            keypoints={'nose': (150, 120, 0.9)},
            pose_classifications=[("standing", 0.8)]
        )

    def create_test_head_pose(self):
        """Helper to create test HeadPoseResult."""
        return HeadPoseResult(
            yaw=15.0,
            pitch=-10.0,
            roll=5.0,
            confidence=0.85,
            face_id=0,
            direction="right"
        )

    def create_test_quality(self, face_quality=0.8, body_quality=0.6):
        """Helper to create test PersonQuality."""
        return PersonQuality(face_quality=face_quality, body_quality=body_quality)

    def test_person_creation_with_face_and_body(self):
        """Test Person creation with both face and body detections."""
        face = self.create_test_face()
        body = self.create_test_body()
        head_pose = self.create_test_head_pose()
        quality = self.create_test_quality()

        person = Person(
            person_id=0,
            face=face,
            body=body,
            head_pose=head_pose,
            quality=quality
        )

        assert person.person_id == 0
        assert person.face is face
        assert person.body is body
        assert person.head_pose is head_pose
        assert person.quality is quality

    def test_person_creation_with_face_only(self):
        """Test Person creation with only face detection."""
        face = self.create_test_face()
        body = BodyUnknown()
        quality = self.create_test_quality()

        person = Person(
            person_id=1,
            face=face,
            body=body,
            head_pose=None,
            quality=quality
        )

        assert person.has_face
        assert not person.has_body
        assert not person.has_head_pose

    def test_person_creation_with_body_only(self):
        """Test Person creation with only body detection."""
        face = FaceUnknown()
        body = self.create_test_body()
        quality = self.create_test_quality()

        person = Person(
            person_id=2,
            face=face,
            body=body,
            head_pose=None,
            quality=quality
        )

        assert not person.has_face
        assert person.has_body
        assert not person.has_head_pose

    def test_person_validation(self):
        """Test Person validation logic."""
        quality = self.create_test_quality()

        # Test negative person_id
        with pytest.raises(ValueError, match="person_id must be non-negative"):
            Person(
                person_id=-1,
                face=self.create_test_face(),
                body=self.create_test_body(),
                head_pose=None,
                quality=quality
            )

        # Test no detections
        with pytest.raises(ValueError, match="Person must have at least one detection"):
            Person(
                person_id=0,
                face=FaceUnknown(),
                body=BodyUnknown(),
                head_pose=None,
                quality=quality
            )

    def test_person_properties(self):
        """Test Person property methods."""
        face = self.create_test_face()
        body = self.create_test_body()
        head_pose = self.create_test_head_pose()
        quality = self.create_test_quality()

        person = Person(
            person_id=0,
            face=face,
            body=body,
            head_pose=head_pose,
            quality=quality
        )

        assert person.has_face
        assert person.has_body
        assert person.has_head_pose
        assert person.is_high_quality
        assert person.is_usable

    def test_person_center_calculation(self):
        """Test Person center point calculation."""
        face = self.create_test_face()  # bbox (100, 100, 200, 200), center (150, 150)
        body = self.create_test_body()  # bbox (80, 50, 220, 400), center (150, 225)
        quality = self.create_test_quality()

        # Test with both face and body (should prefer body)
        person = Person(
            person_id=0,
            face=face,
            body=body,
            head_pose=None,
            quality=quality
        )

        center = person.center
        assert center == (150.0, 225.0)  # Body center

        # Test with face only
        person_face_only = Person(
            person_id=1,
            face=face,
            body=BodyUnknown(),
            head_pose=None,
            quality=quality
        )

        center_face = person_face_only.center
        assert center_face == (150.0, 150.0)  # Face center

    def test_person_serialization(self):
        """Test Person serialization and deserialization."""
        face = self.create_test_face()
        body = self.create_test_body()
        head_pose = self.create_test_head_pose()
        quality = self.create_test_quality()

        person = Person(
            person_id=0,
            face=face,
            body=body,
            head_pose=head_pose,
            quality=quality
        )

        # Test to_dict
        person_dict = person.to_dict()

        assert person_dict["person_id"] == 0
        assert person_dict["face"]["type"] == "FaceDetection"
        assert person_dict["face"]["bbox"] == (100, 100, 200, 200)
        assert person_dict["body"]["type"] == "PoseDetection"
        assert person_dict["body"]["bbox"] == (80, 50, 220, 400)
        assert person_dict["head_pose"]["yaw"] == 15.0
        assert person_dict["quality"]["face_quality"] == 0.8

        # Test from_dict
        reconstructed = Person.from_dict(person_dict)

        assert reconstructed.person_id == person.person_id
        assert reconstructed.face.bbox == person.face.bbox
        assert reconstructed.body.bbox == person.body.bbox
        assert reconstructed.head_pose.yaw == person.head_pose.yaw
        assert reconstructed.quality.face_quality == person.quality.face_quality

    def test_person_serialization_with_sentinels(self):
        """Test Person serialization with sentinel objects."""
        face = FaceUnknown()
        body = BodyUnknown()
        quality = self.create_test_quality()

        # This should raise an error due to validation
        with pytest.raises(ValueError, match="Person must have at least one detection"):
            Person(
                person_id=0,
                face=face,
                body=body,
                head_pose=None,
                quality=quality
            )

        # Test with face only
        person_face_only = Person(
            person_id=0,
            face=self.create_test_face(),
            body=BodyUnknown(),
            head_pose=None,
            quality=quality
        )

        person_dict = person_face_only.to_dict()
        assert person_dict["face"]["type"] == "FaceDetection"
        assert person_dict["body"]["type"] == "BodyUnknown"

        # Test round-trip
        reconstructed = Person.from_dict(person_dict)
        assert isinstance(reconstructed.face, FaceDetection)
        assert isinstance(reconstructed.body, BodyUnknown)
        assert reconstructed.body is BodyUnknown()  # Should be same singleton

    def test_person_serialization_edge_cases(self):
        """Test Person serialization edge cases."""
        face = self.create_test_face()
        quality = self.create_test_quality()

        # Test without head pose
        person = Person(
            person_id=0,
            face=face,
            body=BodyUnknown(),
            head_pose=None,
            quality=quality
        )

        person_dict = person.to_dict()
        assert person_dict["head_pose"] is None

        reconstructed = Person.from_dict(person_dict)
        assert reconstructed.head_pose is None

    def test_person_quality_integration(self):
        """Test Person integration with quality assessment."""
        # Test high quality person
        high_quality = PersonQuality(face_quality=0.9, body_quality=0.8)
        person_high = Person(
            person_id=0,
            face=self.create_test_face(),
            body=self.create_test_body(),
            head_pose=None,
            quality=high_quality
        )

        assert person_high.is_high_quality
        assert person_high.is_usable

        # Test low quality person
        low_quality = PersonQuality(face_quality=0.1, body_quality=0.2)
        person_low = Person(
            person_id=1,
            face=self.create_test_face(),
            body=self.create_test_body(),
            head_pose=None,
            quality=low_quality
        )

        assert not person_low.is_high_quality
        assert not person_low.is_usable
