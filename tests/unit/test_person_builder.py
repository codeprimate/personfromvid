"""Unit tests for PersonBuilder class."""

from unittest.mock import Mock

from personfromvid.analysis.person_builder import PersonBuilder
from personfromvid.data.detection_results import (
    FaceDetection,
    HeadPoseResult,
    PoseDetection,
)
from personfromvid.data.person import BodyUnknown, FaceUnknown, Person, PersonQuality


class TestPersonBuilder:
    """Test cases for PersonBuilder class."""

    def test_init(self):
        """Test PersonBuilder initialization."""
        builder = PersonBuilder()
        assert builder is not None
        assert hasattr(builder, 'logger')

    def test_build_persons_empty_input(self):
        """Test build_persons with empty input lists."""
        builder = PersonBuilder()

        result = builder.build_persons([], [], [])

        assert result == []
        assert isinstance(result, list)

    def test_build_persons_error_handling(self):
        """Test build_persons handles errors gracefully."""
        builder = PersonBuilder()

        # Test with None inputs to trigger error
        result = builder.build_persons(None, None, None)

        # Should return empty list on error
        assert result == []
        assert isinstance(result, list)

    def test_build_persons_basic_call(self):
        """Test build_persons with mock detections."""
        builder = PersonBuilder()

        # Create mock detections
        face_detections = [Mock(spec=FaceDetection)]
        pose_detections = [Mock(spec=PoseDetection)]
        head_poses = [Mock(spec=HeadPoseResult)]

        result = builder.build_persons(face_detections, pose_detections, head_poses)

        # Should now return Person objects (not empty list)
        assert isinstance(result, list)
        # Result depends on spatial matching, but should be a list

    def create_test_face(self, x1=100, y1=100, x2=200, y2=200, confidence=0.9):
        """Create test FaceDetection with specified bbox."""
        return FaceDetection(
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            landmarks=None
        )

    def create_test_body(self, x1=80, y1=50, x2=220, y2=400, confidence=0.8):
        """Create test PoseDetection with specified bbox."""
        return PoseDetection(
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            keypoints={
                "nose": (150, 100, 0.9),
                "left_shoulder": (120, 150, 0.8),
                "right_shoulder": (180, 150, 0.8),
            }
        )

    def test_is_face_inside_body_bbox(self):
        """Test geometric containment checking."""
        builder = PersonBuilder()

        # Face center inside body bbox
        assert builder._is_face_inside_body_bbox((150, 150), (100, 100, 200, 200))

        # Face center outside body bbox - left side
        assert not builder._is_face_inside_body_bbox((50, 150), (100, 100, 200, 200))

        # Face center outside body bbox - right side
        assert not builder._is_face_inside_body_bbox((250, 150), (100, 100, 200, 200))

        # Face center outside body bbox - above
        assert not builder._is_face_inside_body_bbox((150, 50), (100, 100, 200, 200))

        # Face center outside body bbox - below
        assert not builder._is_face_inside_body_bbox((150, 250), (100, 100, 200, 200))

        # Face center on body bbox edge (should be inside)
        assert builder._is_face_inside_body_bbox((100, 100), (100, 100, 200, 200))
        assert builder._is_face_inside_body_bbox((200, 200), (100, 100, 200, 200))

    def test_calculate_distance(self):
        """Test Euclidean distance calculation."""
        builder = PersonBuilder()

        # Same point
        assert builder._calculate_distance((0, 0), (0, 0)) == 0.0

        # Horizontal distance
        assert builder._calculate_distance((0, 0), (3, 0)) == 3.0

        # Vertical distance
        assert builder._calculate_distance((0, 0), (0, 4)) == 4.0

        # Diagonal distance (3-4-5 triangle)
        assert builder._calculate_distance((0, 0), (3, 4)) == 5.0

        # Negative coordinates
        assert builder._calculate_distance((-1, -1), (2, 3)) == 5.0

    def test_spatial_proximity_matching_perfect_match(self):
        """Test spatial matching with perfect face-body alignment."""
        builder = PersonBuilder()

        # Create face centered in body bbox
        face = self.create_test_face(140, 140, 160, 160)  # Center at (150, 150)
        body = self.create_test_body(100, 100, 200, 200)  # Face center inside

        pairs, unmatched_faces, unmatched_bodies = builder._spatial_proximity_matching([face], [body])

        assert len(pairs) == 1
        assert pairs[0] == (face, body)
        assert len(unmatched_faces) == 0
        assert len(unmatched_bodies) == 0

    def test_spatial_proximity_matching_no_containment(self):
        """Test spatial matching when face is outside body bbox."""
        builder = PersonBuilder()

        # Create face completely outside body bbox
        face = self.create_test_face(300, 300, 400, 400)  # Center at (350, 350)
        body = self.create_test_body(100, 100, 200, 200)  # Face center outside

        pairs, unmatched_faces, unmatched_bodies = builder._spatial_proximity_matching([face], [body])

        assert len(pairs) == 0
        assert unmatched_faces == [face]
        assert unmatched_bodies == [body]

    def test_spatial_proximity_matching_multiple_candidates(self):
        """Test spatial matching with multiple valid pairs - should pick closest."""
        builder = PersonBuilder()

        # Face that could match either body
        face = self.create_test_face(140, 140, 160, 160)  # Center at (150, 150)

        # Two bodies that both contain the face center
        body1 = self.create_test_body(100, 100, 200, 200)  # Center at (150, 150) - distance 0
        body2 = self.create_test_body(120, 120, 180, 180)  # Center at (150, 150) - distance 0

        # Actually, let's make them have different centers to test distance selection
        body1 = self.create_test_body(100, 100, 200, 200)  # Center at (150, 150)
        body2 = self.create_test_body(140, 140, 240, 240)  # Center at (190, 190)

        pairs, unmatched_faces, unmatched_bodies = builder._spatial_proximity_matching([face], [body1, body2])

        assert len(pairs) == 1
        # Should pick body1 as it's closer to face center
        assert pairs[0] == (face, body1)
        assert len(unmatched_faces) == 0
        assert unmatched_bodies == [body2]

    def test_spatial_proximity_matching_greedy_selection(self):
        """Test greedy selection ensures each face/body used only once."""
        builder = PersonBuilder()

        # Two faces
        face1 = self.create_test_face(140, 140, 160, 160)  # Center at (150, 150)
        face2 = self.create_test_face(145, 145, 165, 165)  # Center at (155, 155)

        # One body that could match both faces
        body = self.create_test_body(100, 100, 200, 200)  # Contains both face centers

        pairs, unmatched_faces, unmatched_bodies = builder._spatial_proximity_matching([face1, face2], [body])

        # Should match only one face-body pair
        assert len(pairs) == 1
        assert len(unmatched_faces) == 1
        assert len(unmatched_bodies) == 0

        # Should pick the closer face (face1 is closer to body center)
        assert pairs[0][0] in [face1, face2]
        assert pairs[0][1] == body

    def test_spatial_proximity_matching_empty_inputs(self):
        """Test spatial matching with empty input lists."""
        builder = PersonBuilder()

        # Empty faces
        pairs, unmatched_faces, unmatched_bodies = builder._spatial_proximity_matching([], [self.create_test_body()])
        assert pairs == []
        assert unmatched_faces == []
        assert len(unmatched_bodies) == 1

        # Empty bodies
        pairs, unmatched_faces, unmatched_bodies = builder._spatial_proximity_matching([self.create_test_face()], [])
        assert pairs == []
        assert len(unmatched_faces) == 1
        assert unmatched_bodies == []

    def test_create_person_from_detections_face_and_body(self):
        """Test Person creation with both face and body."""
        builder = PersonBuilder()

        face = self.create_test_face()
        body = self.create_test_body()
        head_poses = []

        person = builder._create_person_from_detections(face, body, head_poses)

        assert isinstance(person, Person)
        assert person.face == face
        assert person.body == body
        assert person.head_pose is None
        assert isinstance(person.quality, PersonQuality)
        # Quality should be weighted average: 0.7 * 0.9 + 0.3 * 0.8 = 0.87
        assert abs(person.quality.overall_quality - 0.87) < 0.01

    def test_create_person_from_detections_face_only(self):
        """Test Person creation with face only."""
        builder = PersonBuilder()

        face = self.create_test_face()

        person = builder._create_person_from_detections(face, None, [])

        assert isinstance(person, Person)
        assert person.face == face
        assert isinstance(person.body, BodyUnknown)
        assert person.head_pose is None
        # Quality should be: 0.7 * 0.9 + 0.3 * 0.0 = 0.63
        assert abs(person.quality.overall_quality - 0.63) < 0.01

    def test_create_person_from_detections_body_only(self):
        """Test Person creation with body only."""
        builder = PersonBuilder()

        body = self.create_test_body()

        person = builder._create_person_from_detections(None, body, [])

        assert isinstance(person, Person)
        assert isinstance(person.face, FaceUnknown)
        assert person.body == body
        assert person.head_pose is None
        # Quality should be: 0.7 * 0.0 + 0.3 * 0.8 = 0.24
        assert abs(person.quality.overall_quality - 0.24) < 0.01

    def test_build_persons_integration_spatial_matching(self):
        """Test complete build_persons with spatial matching integration."""
        builder = PersonBuilder()

        # Create test scenario: 2 faces, 2 bodies, 1 spatial match possible
        face1 = self.create_test_face(140, 140, 160, 160)  # Center at (150, 150)
        face2 = self.create_test_face(300, 300, 320, 320)  # Center at (310, 310)

        body1 = self.create_test_body(100, 100, 200, 200)  # Contains face1 center
        body2 = self.create_test_body(250, 250, 350, 350)  # Contains face2 center

        persons = builder.build_persons([face1, face2], [body1, body2], [])

        # Should create 2 Person objects from spatial matches
        assert len(persons) == 2

        # Check that persons have expected associations
        person_faces = [p.face for p in persons if not isinstance(p.face, FaceUnknown)]
        person_bodies = [p.body for p in persons if not isinstance(p.body, BodyUnknown)]

        assert len(person_faces) == 2
        assert len(person_bodies) == 2
        assert face1 in person_faces
        assert face2 in person_faces
        assert body1 in person_bodies
        assert body2 in person_bodies

    def test_index_based_fallback_matching_equal_counts(self):
        """Test fallback matching with equal face and body counts."""
        builder = PersonBuilder()

        # Create faces and bodies sorted by x position
        face1 = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110)
        face2 = self.create_test_face(200, 100, 220, 120)  # Center at (210, 110)
        face3 = self.create_test_face(300, 100, 320, 120)  # Center at (310, 110)

        body1 = self.create_test_body(80, 50, 140, 200)    # Center at (110, 125)
        body2 = self.create_test_body(180, 50, 240, 200)   # Center at (210, 125)
        body3 = self.create_test_body(280, 50, 340, 200)   # Center at (310, 125)

        pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching(
            [face1, face2, face3], [body1, body2, body3]
        )

        # Should pair all faces with bodies by sorted index
        assert len(pairs) == 3
        assert len(remaining_faces) == 0
        assert len(remaining_bodies) == 0

        # Verify correct pairing by x-coordinate sorting
        assert pairs[0] == (face1, body1)  # Both leftmost
        assert pairs[1] == (face2, body2)  # Both middle
        assert pairs[2] == (face3, body3)  # Both rightmost

    def test_index_based_fallback_matching_more_faces(self):
        """Test fallback matching with more faces than bodies."""
        builder = PersonBuilder()

        # 3 faces, 2 bodies
        face1 = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110)
        face2 = self.create_test_face(200, 100, 220, 120)  # Center at (210, 110)
        face3 = self.create_test_face(300, 100, 320, 120)  # Center at (310, 110)

        body1 = self.create_test_body(180, 50, 240, 200)   # Center at (210, 125)
        body2 = self.create_test_body(80, 50, 140, 200)    # Center at (110, 125)

        pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching(
            [face1, face2, face3], [body1, body2]
        )

        # Should pair first 2 faces with bodies by sorted index
        assert len(pairs) == 2
        assert len(remaining_faces) == 1
        assert len(remaining_bodies) == 0

        # Verify sorting: body2 (x=110) should pair with face1 (x=110)
        # body1 (x=210) should pair with face2 (x=210)
        assert pairs[0] == (face1, body2)  # Both leftmost after sorting
        assert pairs[1] == (face2, body1)  # Both middle after sorting
        assert remaining_faces == [face3]  # Rightmost face unpaired

    def test_index_based_fallback_matching_more_bodies(self):
        """Test fallback matching with more bodies than faces."""
        builder = PersonBuilder()

        # 2 faces, 3 bodies
        face1 = self.create_test_face(200, 100, 220, 120)  # Center at (210, 110)
        face2 = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110)

        body1 = self.create_test_body(80, 50, 140, 200)    # Center at (110, 125)
        body2 = self.create_test_body(180, 50, 240, 200)   # Center at (210, 125)
        body3 = self.create_test_body(280, 50, 340, 200)   # Center at (310, 125)

        pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching(
            [face1, face2], [body1, body2, body3]
        )

        # Should pair first 2 bodies with faces by sorted index
        assert len(pairs) == 2
        assert len(remaining_faces) == 0
        assert len(remaining_bodies) == 1

        # Verify sorting: face2 (x=110) pairs with body1 (x=110)
        # face1 (x=210) pairs with body2 (x=210)
        assert pairs[0] == (face2, body1)  # Both leftmost after sorting
        assert pairs[1] == (face1, body2)  # Both middle after sorting
        assert remaining_bodies == [body3]  # Rightmost body unpaired

    def test_index_based_fallback_matching_deterministic_sorting(self):
        """Test that sorting is deterministic and stable."""
        builder = PersonBuilder()

        # Create faces with same x-coordinate to test stable sorting
        face1 = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110)
        face2 = self.create_test_face(100, 150, 120, 170)  # Center at (110, 160) - same x!
        face3 = self.create_test_face(200, 100, 220, 120)  # Center at (210, 110)

        body1 = self.create_test_body(80, 50, 140, 200)    # Center at (110, 125)
        body2 = self.create_test_body(180, 50, 240, 200)   # Center at (210, 125)

        # Run multiple times to verify deterministic behavior
        for _ in range(3):
            pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching(
                [face1, face2, face3], [body1, body2]
            )

            assert len(pairs) == 2
            assert len(remaining_faces) == 1

            # Should consistently pair by stable sort order
            # face1 and face2 both have x=110, but face1 should come first (stable sort)
            # face3 has x=210
            # body1 has x=110, body2 has x=210
            assert pairs[0][0] in [face1, face2]  # One of the x=110 faces
            assert pairs[0][1] == body1  # x=110 body
            assert pairs[1][0] in [face1, face2, face3]
            assert pairs[1][1] == body2  # x=210 body

    def test_index_based_fallback_matching_empty_inputs(self):
        """Test fallback matching with empty inputs."""
        builder = PersonBuilder()

        # Both empty
        pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching([], [])
        assert pairs == []
        assert remaining_faces == []
        assert remaining_bodies == []

        # Empty faces
        body = self.create_test_body()
        pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching([], [body])
        assert pairs == []
        assert remaining_faces == []
        assert remaining_bodies == [body]

        # Empty bodies
        face = self.create_test_face()
        pairs, remaining_faces, remaining_bodies = builder._index_based_fallback_matching([face], [])
        assert pairs == []
        assert remaining_faces == [face]
        assert remaining_bodies == []

    def test_build_persons_integration_with_fallback(self):
        """Test complete build_persons with both spatial and fallback matching."""
        builder = PersonBuilder()

        # Create scenario where spatial matching partially succeeds, requiring fallback
        # Face1 and Body1 will match spatially (face center inside body bbox)
        face1 = self.create_test_face(140, 140, 160, 160)  # Center at (150, 150)
        body1 = self.create_test_body(100, 100, 200, 200)  # Contains face1 center

        # Face2 and Body2 will NOT match spatially (face center outside body bbox)
        # but should match via fallback (both on right side)
        face2 = self.create_test_face(300, 50, 320, 70)    # Center at (310, 60)
        body2 = self.create_test_body(280, 100, 340, 300)  # Center at (310, 200) - face outside

        persons = builder.build_persons([face1, face2], [body1, body2], [])

        # Should create 2 Person objects: 1 from spatial + 1 from fallback
        assert len(persons) == 2

        # All detections should be used
        person_faces = [p.face for p in persons if not isinstance(p.face, FaceUnknown)]
        person_bodies = [p.body for p in persons if not isinstance(p.body, BodyUnknown)]

        assert len(person_faces) == 2
        assert len(person_bodies) == 2
        assert face1 in person_faces
        assert face2 in person_faces
        assert body1 in person_bodies
        assert body2 in person_bodies

    def test_get_person_x_coordinate_body_available(self):
        """Test x-coordinate extraction when body is available."""
        builder = PersonBuilder()

        # Create person with both face and body
        face = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110)
        body = self.create_test_body(200, 200, 220, 220)  # Center at (210, 210)

        person = builder._create_person_from_detections(face, body, [])

        # Should use body center x-coordinate (priority 1)
        x_coord = builder._get_person_x_coordinate(person)
        assert x_coord == 210.0  # Body center x

    def test_get_person_x_coordinate_face_fallback(self):
        """Test x-coordinate extraction when body is missing (face fallback)."""
        builder = PersonBuilder()

        # Create person with face only (body will be BodyUnknown)
        face = self.create_test_face(150, 150, 170, 170)  # Center at (160, 160)

        person = builder._create_person_from_detections(face, None, [])

        # Should use face center x-coordinate (priority 2)
        x_coord = builder._get_person_x_coordinate(person)
        assert x_coord == 160.0  # Face center x

    def test_get_person_x_coordinate_body_fallback(self):
        """Test x-coordinate extraction when face is missing (body available)."""
        builder = PersonBuilder()

        # Create person with body only (face will be FaceUnknown)
        body = self.create_test_body(250, 250, 270, 270)  # Center at (260, 260)

        person = builder._create_person_from_detections(None, body, [])

        # Should use body center x-coordinate (priority 1)
        x_coord = builder._get_person_x_coordinate(person)
        assert x_coord == 260.0  # Body center x

    def test_assign_person_ids_left_to_right_ordering(self):
        """Test person_id assignment with left-to-right ordering."""
        builder = PersonBuilder()

        # Create persons at different x positions (out of order initially)
        face1 = self.create_test_face(300, 100, 320, 120)  # Center at (310, 110) - rightmost
        face2 = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110) - leftmost
        face3 = self.create_test_face(200, 100, 220, 120)  # Center at (210, 110) - middle

        body1 = self.create_test_body(290, 50, 330, 150)   # Center at (310, 100) - rightmost
        body2 = self.create_test_body(90, 50, 130, 150)    # Center at (110, 100) - leftmost
        body3 = self.create_test_body(190, 50, 230, 150)   # Center at (210, 100) - middle

        # Create Person objects (initially in random order)
        person1 = builder._create_person_from_detections(face1, body1, [])  # Rightmost
        person2 = builder._create_person_from_detections(face2, body2, [])  # Leftmost
        person3 = builder._create_person_from_detections(face3, body3, [])  # Middle

        persons = [person1, person2, person3]  # Random initial order

        # Assign person IDs
        builder._assign_person_ids(persons)

        # Verify left-to-right ordering by person_id
        assert person2.person_id == 0  # Leftmost (x=110)
        assert person3.person_id == 1  # Middle (x=210)
        assert person1.person_id == 2  # Rightmost (x=310)

    def test_assign_person_ids_mixed_detections(self):
        """Test person_id assignment with mixed face/body availability."""
        builder = PersonBuilder()

        # Person 1: Face only (leftmost)
        face1 = self.create_test_face(100, 100, 120, 120)  # Center at (110, 110)
        person1 = builder._create_person_from_detections(face1, None, [])

        # Person 2: Body only (rightmost)
        body2 = self.create_test_body(290, 50, 330, 150)   # Center at (310, 100)
        person2 = builder._create_person_from_detections(None, body2, [])

        # Person 3: Both face and body (middle) - should use body x-coordinate
        face3 = self.create_test_face(180, 100, 200, 120)  # Center at (190, 110)
        body3 = self.create_test_body(210, 50, 250, 150)   # Center at (230, 100) - different from face!
        person3 = builder._create_person_from_detections(face3, body3, [])

        persons = [person2, person1, person3]  # Random initial order

        # Assign person IDs
        builder._assign_person_ids(persons)

        # Verify ordering: person1 (face x=110), person3 (body x=230), person2 (body x=310)
        assert person1.person_id == 0  # Leftmost (face center x=110)
        assert person3.person_id == 1  # Middle (body center x=230, not face x=190)
        assert person2.person_id == 2  # Rightmost (body center x=310)

    def test_assign_person_ids_identical_coordinates(self):
        """Test person_id assignment with identical x-coordinates (stable sort)."""
        builder = PersonBuilder()

        # Create persons with same x-coordinate
        face1 = self.create_test_face(150, 100, 170, 120)  # Center at (160, 110)
        face2 = self.create_test_face(150, 200, 170, 220)  # Center at (160, 210) - same x!

        body1 = self.create_test_body(140, 50, 180, 150)   # Center at (160, 100) - same x!
        body2 = self.create_test_body(140, 180, 180, 280)  # Center at (160, 230) - same x!

        person1 = builder._create_person_from_detections(face1, body1, [])
        person2 = builder._create_person_from_detections(face2, body2, [])

        persons = [person1, person2]  # Original order

        # Assign person IDs
        builder._assign_person_ids(persons)

        # Should maintain stable sort order (original order preserved for identical coordinates)
        assert person1.person_id == 0  # First in original order
        assert person2.person_id == 1  # Second in original order

    def test_assign_person_ids_empty_list(self):
        """Test person_id assignment with empty persons list."""
        builder = PersonBuilder()

        persons = []

        # Should handle empty list gracefully
        builder._assign_person_ids(persons)

        # No error should occur, list remains empty
        assert len(persons) == 0

    def test_assign_person_ids_single_person(self):
        """Test person_id assignment with single person."""
        builder = PersonBuilder()

        face = self.create_test_face()
        person = builder._create_person_from_detections(face, None, [])

        persons = [person]

        # Assign person IDs
        builder._assign_person_ids(persons)

        # Single person should get person_id = 0
        assert person.person_id == 0

    def test_build_persons_integration_with_person_id_assignment(self):
        """Test complete build_persons with person_id assignment integration."""
        builder = PersonBuilder()

        # Create faces and bodies at different x positions
        face1 = self.create_test_face(300, 140, 320, 160)  # Center at (310, 150) - rightmost
        face2 = self.create_test_face(100, 140, 120, 160)  # Center at (110, 150) - leftmost

        body1 = self.create_test_body(290, 100, 330, 200)  # Center at (310, 150) - rightmost
        body2 = self.create_test_body(90, 100, 130, 200)   # Center at (110, 150) - leftmost

        persons = builder.build_persons([face1, face2], [body1, body2], [])

        # Should create 2 Person objects with correct left-to-right person_id assignment
        assert len(persons) == 2

        # Find persons by their face association
        person_with_face1 = next(p for p in persons if p.face == face1)
        person_with_face2 = next(p for p in persons if p.face == face2)

        # Verify person_id assignment based on position
        assert person_with_face2.person_id == 0  # Leftmost (x=110)
        assert person_with_face1.person_id == 1  # Rightmost (x=310)
