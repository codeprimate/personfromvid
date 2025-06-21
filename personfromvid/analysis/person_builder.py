"""Person building logic for associating face and body detections.

This module provides the PersonBuilder class that implements a two-pass algorithm
to associate face detections with body pose detections within a frame, creating
Person objects with spatial proximity and index-based fallback matching.
"""

import math
from typing import List, Optional, Tuple

from ..data.detection_results import FaceDetection, HeadPoseResult, PoseDetection
from ..data.person import BodyUnknown, FaceUnknown, Person, PersonQuality
from ..utils.logging import get_logger


class PersonBuilder:
    """Builds Person objects from face and pose detections using association algorithms.

    The PersonBuilder implements a two-pass association strategy:
    1. Spatial proximity matching using geometric containment and distance
    2. Index-based fallback matching for remaining unmatched detections

    The resulting Person objects are assigned person_id values based on
    left-to-right ordering within the frame.
    """

    def __init__(self) -> None:
        """Initialize PersonBuilder with logging."""
        self.logger = get_logger("person_builder")

    def build_persons(
        self,
        face_detections: List[FaceDetection],
        pose_detections: List[PoseDetection],
        head_poses: List[HeadPoseResult],
    ) -> List[Person]:
        """Build Person objects from face and body detections.

        Args:
            face_detections: List of detected faces in the frame
            pose_detections: List of detected body poses in the frame
            head_poses: List of head pose results linked to faces

        Returns:
            List of Person objects with associated detections, ordered left-to-right

        Raises:
            ValueError: If input data is invalid or contains inconsistencies
        """
        try:
            self.logger.info(
                f"🔧 Building persons from {len(face_detections)} faces, "
                f"{len(pose_detections)} bodies, {len(head_poses)} head poses"
            )

            # Handle empty input case
            if not face_detections and not pose_detections:
                self.logger.info("No detections available, returning empty person list")
                return []

            # Pass 1: Spatial proximity matching
            (
                face_body_pairs,
                unmatched_faces,
                unmatched_bodies,
            ) = self._spatial_proximity_matching(face_detections, pose_detections)

            self.logger.debug(
                f"Spatial matching: {len(face_body_pairs)} pairs, "
                f"{len(unmatched_faces)} unmatched faces, {len(unmatched_bodies)} unmatched bodies"
            )

            # Pass 2: Index-based fallback matching
            (
                fallback_pairs,
                remaining_faces,
                remaining_bodies,
            ) = self._index_based_fallback_matching(unmatched_faces, unmatched_bodies)

            self.logger.debug(
                f"Fallback matching: {len(fallback_pairs)} additional pairs, "
                f"{len(remaining_faces)} remaining faces, {len(remaining_bodies)} remaining bodies"
            )

            # Create Person objects from all matches
            persons = []

            # Create Person objects from spatial matches
            self.logger.debug(
                f"Creating {len(face_body_pairs)} Person objects from spatial matches"
            )
            for face, body in face_body_pairs:
                person = self._create_person_from_detections(face, body, head_poses)
                persons.append(person)
                self.logger.debug(
                    f"Created Person from spatial match: face {face.center} + body {body.center}"
                )

            # Create Person objects from fallback matches
            self.logger.debug(
                f"Creating {len(fallback_pairs)} Person objects from fallback matches"
            )
            for face, body in fallback_pairs:
                person = self._create_person_from_detections(face, body, head_poses)
                persons.append(person)
                self.logger.debug(
                    f"Created Person from fallback match: face {face.center} + body {body.center}"
                )

            # Create Person objects from remaining unmatched faces
            if remaining_faces:
                self.logger.debug(
                    f"Creating {len(remaining_faces)} Person objects from unmatched faces"
                )
                for face in remaining_faces:
                    person = self._create_person_from_detections(face, None, head_poses)
                    persons.append(person)
                    self.logger.debug(
                        f"Created Person from unmatched face: {face.center} (no body)"
                    )

            # Create Person objects from remaining unmatched bodies
            if remaining_bodies:
                self.logger.debug(
                    f"Creating {len(remaining_bodies)} Person objects from unmatched bodies"
                )
                for body in remaining_bodies:
                    person = self._create_person_from_detections(None, body, head_poses)
                    persons.append(person)
                    self.logger.debug(
                        f"Created Person from unmatched body: {body.center} (no face)"
                    )

            # Assign person_id based on left-to-right ordering
            self._assign_person_ids(persons)

            self.logger.info(f"✅ Successfully built {len(persons)} Person objects")
            return persons

        except Exception as e:
            self.logger.error(f"❌ Person building failed with error: {e}")
            self.logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            # Return empty list on failure to maintain pipeline robustness
            return []

    def _spatial_proximity_matching(
        self, faces: List[FaceDetection], bodies: List[PoseDetection]
    ) -> Tuple[
        List[Tuple[FaceDetection, PoseDetection]],
        List[FaceDetection],
        List[PoseDetection],
    ]:
        """Perform spatial proximity matching between faces and bodies.

        Uses geometric containment (face center inside body bbox) and distance-based
        greedy selection to find optimal face-body pairs.

        Args:
            faces: List of face detections
            bodies: List of body pose detections

        Returns:
            Tuple of (matched_pairs, unmatched_faces, unmatched_bodies)
        """
        if not faces or not bodies:
            return [], faces.copy(), bodies.copy()

        # Find all valid face-body pairs where face center is inside body bbox
        valid_pairs = []
        containment_checks = 0

        for face in faces:
            face_center = face.center
            self.logger.debug(
                f"Checking face at {face_center} against {len(bodies)} bodies"
            )

            for body in bodies:
                containment_checks += 1
                body_bbox = body.bbox
                body_center = body.center

                if self._is_face_inside_body_bbox(face_center, body_bbox):
                    distance = self._calculate_distance(face_center, body_center)
                    valid_pairs.append((face, body, distance))
                    self.logger.debug(
                        f"Face {face_center} inside body bbox {body_bbox} "
                        f"(center {body_center}), distance: {distance:.1f}px"
                    )
                else:
                    self.logger.debug(
                        f"Face {face_center} outside body bbox {body_bbox} "
                        f"(center {body_center}) - skipping"
                    )

        self.logger.info(
            f"Spatial proximity: {containment_checks} containment checks, "
            f"{len(valid_pairs)} valid face-body pairs found"
        )

        # Sort pairs by distance (closest first)
        valid_pairs.sort(key=lambda x: x[2])

        if valid_pairs:
            self.logger.debug(
                f"Distance-sorted pairs: closest={valid_pairs[0][2]:.1f}px, "
                f"farthest={valid_pairs[-1][2]:.1f}px"
            )

        # Greedily select best pairs
        matched_pairs = []
        used_faces: List[FaceDetection] = []
        used_bodies: List[PoseDetection] = []
        rejected_pairs = 0

        for face, body, distance in valid_pairs:
            if face not in used_faces and body not in used_bodies:
                matched_pairs.append((face, body))
                used_faces.append(face)
                used_bodies.append(body)
                self.logger.debug(
                    f"✓ Selected face {face.center} → body {body.center}, distance: {distance:.1f}px"
                )
            else:
                rejected_pairs += 1
                self.logger.debug(
                    f"✗ Rejected face {face.center} → body {body.center} (already used)"
                )

        if rejected_pairs > 0:
            self.logger.debug(
                f"Rejected {rejected_pairs} pairs due to greedy selection constraints"
            )

        # Determine unmatched detections
        unmatched_faces = [f for f in faces if f not in used_faces]
        unmatched_bodies = [b for b in bodies if b not in used_bodies]

        self.logger.info(
            f"Spatial matching complete: {len(matched_pairs)} pairs selected, "
            f"{len(unmatched_faces)} faces unmatched, {len(unmatched_bodies)} bodies unmatched"
        )

        # Log unmatched detection positions for debugging
        if unmatched_faces:
            face_positions = [f.center for f in unmatched_faces]
            self.logger.debug(f"Unmatched face positions: {face_positions}")
        if unmatched_bodies:
            body_positions = [b.center for b in unmatched_bodies]
            self.logger.debug(f"Unmatched body positions: {body_positions}")

        return matched_pairs, unmatched_faces, unmatched_bodies

    def _index_based_fallback_matching(
        self, faces: List[FaceDetection], bodies: List[PoseDetection]
    ) -> Tuple[
        List[Tuple[FaceDetection, PoseDetection]],
        List[FaceDetection],
        List[PoseDetection],
    ]:
        """Perform index-based fallback matching for unmatched detections.

        Sorts remaining faces and bodies by x-coordinate (left-to-right) and pairs them
        by sorted index position. This provides deterministic fallback matching when
        spatial proximity heuristics fail.

        Args:
            faces: List of unmatched face detections from spatial matching
            bodies: List of unmatched body pose detections from spatial matching

        Returns:
            Tuple of (fallback_pairs, remaining_faces, remaining_bodies)
        """
        if not faces and not bodies:
            self.logger.debug("No faces or bodies for fallback matching")
            return [], [], []

        if not faces:
            self.logger.info(
                f"No unmatched faces - {len(bodies)} bodies remain unmatched"
            )
            return [], [], bodies.copy()

        if not bodies:
            self.logger.info(
                f"No unmatched bodies - {len(faces)} faces remain unmatched"
            )
            return [], faces.copy(), []

        # Sort faces and bodies by x-coordinate (left-to-right)
        sorted_faces = sorted(faces, key=lambda face: face.center[0])
        sorted_bodies = sorted(bodies, key=lambda body: body.center[0])

        self.logger.info(
            f"Index-based fallback: sorted {len(sorted_faces)} faces and {len(sorted_bodies)} bodies by x-coordinate"
        )

        # Log sorted positions for debugging
        if sorted_faces:
            face_x_coords = [f.center[0] for f in sorted_faces]
            self.logger.debug(f"Sorted face x-coordinates: {face_x_coords}")
        if sorted_bodies:
            body_x_coords = [b.center[0] for b in sorted_bodies]
            self.logger.debug(f"Sorted body x-coordinates: {body_x_coords}")

        # Pair sorted detections by index
        fallback_pairs = []
        min_count = min(len(sorted_faces), len(sorted_bodies))

        self.logger.debug(
            f"Creating {min_count} fallback pairs using index-based matching"
        )

        for i in range(min_count):
            face = sorted_faces[i]
            body = sorted_bodies[i]
            fallback_pairs.append((face, body))
            distance = self._calculate_distance(face.center, body.center)
            self.logger.debug(
                f"📍 Fallback pair {i}: face {face.center} ↔ body {body.center} "
                f"(distance: {distance:.1f}px)"
            )

        # Determine remaining unmatched detections
        remaining_faces = (
            sorted_faces[min_count:] if len(sorted_faces) > min_count else []
        )
        remaining_bodies = (
            sorted_bodies[min_count:] if len(sorted_bodies) > min_count else []
        )

        self.logger.info(
            f"Fallback matching complete: {len(fallback_pairs)} pairs created, "
            f"{len(remaining_faces)} faces remaining, {len(remaining_bodies)} bodies remaining"
        )

        # Log remaining detection positions for debugging
        if remaining_faces:
            remaining_face_positions = [f.center for f in remaining_faces]
            self.logger.debug(
                f"Remaining unmatched face positions: {remaining_face_positions}"
            )
        if remaining_bodies:
            remaining_body_positions = [b.center for b in remaining_bodies]
            self.logger.debug(
                f"Remaining unmatched body positions: {remaining_body_positions}"
            )

        return fallback_pairs, remaining_faces, remaining_bodies

    def _calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            point1: First point as (x, y) tuple
            point2: Second point as (x, y) tuple

        Returns:
            Euclidean distance between points
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _is_face_inside_body_bbox(
        self, face_center: Tuple[float, float], body_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if face center point is inside body bounding box.

        Args:
            face_center: Face center point as (x, y) tuple
            body_bbox: Body bounding box in (x1, y1, x2, y2) format

        Returns:
            True if face center is inside body bbox, False otherwise
        """
        face_x, face_y = face_center
        x1, y1, x2, y2 = body_bbox

        return x1 <= face_x <= x2 and y1 <= face_y <= y2

    def _create_person_from_detections(
        self,
        face: Optional[FaceDetection],
        body: Optional[PoseDetection],
        head_poses: List[HeadPoseResult],
    ) -> Person:
        """Create Person object from face and body detections.

        Args:
            face: Face detection or None
            body: Body detection or None
            head_poses: List of head pose results

        Returns:
            Person object with associated detections
        """
        # Use sentinel objects for missing detections
        person_face = face if face is not None else FaceUnknown()
        person_body = body if body is not None else BodyUnknown()

        self.logger.debug(
            f"Creating Person: face={'present' if face is not None else 'missing'}, "
            f"body={'present' if body is not None else 'missing'}"
        )

        # Find matching head pose for this face
        person_head_pose = None
        if face is not None:
            # Head poses are typically linked by face index, but we'll match by proximity
            # For now, use the first available head pose (simplified approach)
            if head_poses:
                person_head_pose = head_poses[0]  # TODO: Improve head pose association
                self.logger.debug("Assigned head pose to Person (simplified approach)")
            else:
                self.logger.debug("No head poses available for Person")

        # Calculate quality score (placeholder - will be improved in quality assessment phase)
        face_quality = face.confidence if face is not None else 0.0
        body_quality = body.confidence if body is not None else 0.0

        quality = PersonQuality(face_quality=face_quality, body_quality=body_quality)

        self.logger.debug(
            f"Person quality: face={face_quality:.3f}, body={body_quality:.3f}, "
            f"combined={quality.overall_quality:.3f}"
        )

        return Person(
            person_id=0,  # Will be assigned later based on ordering
            face=person_face,
            body=person_body,
            head_pose=person_head_pose,
            quality=quality,
        )

    def _get_person_x_coordinate(self, person: Person) -> float:
        """Extract x-coordinate for person ordering.

        Uses body center x-coordinate when available, falls back to face center
        x-coordinate when body is missing. This ensures consistent left-to-right
        ordering based on person position in the frame.

        Args:
            person: Person object to extract coordinate from

        Returns:
            X-coordinate for sorting (body center preferred, face center fallback)
        """
        # Priority 1: Use body center x-coordinate if body is available
        if not isinstance(person.body, BodyUnknown):
            return person.body.center[0]

        # Priority 2: Use face center x-coordinate if face is available
        if not isinstance(person.face, FaceUnknown):
            return person.face.center[0]

        # Fallback: Should not occur with current logic, but handle gracefully
        # Return 0.0 as leftmost position for rare edge case
        self.logger.warning(
            "Person has both unknown face and body - using fallback x-coordinate"
        )
        return 0.0

    def _assign_person_ids(self, persons: List[Person]) -> None:
        """Assign person_id values based on left-to-right ordering.

        Sorts Person objects by x-coordinate (body center preferred, face center fallback)
        and assigns person_id from 0 to N-1 based on sorted position.

        Args:
            persons: List of Person objects to assign IDs to
        """
        if not persons:
            self.logger.debug("No persons to assign IDs to")
            return

        # Sort persons by x-coordinate for left-to-right ordering
        sorted_persons = sorted(persons, key=lambda p: self._get_person_x_coordinate(p))

        # Assign person_id based on sorted position
        for i, person in enumerate(sorted_persons):
            person.person_id = i
            coord = self._get_person_x_coordinate(person)

            # Determine sorting basis for logging
            sort_basis = "body" if not isinstance(person.body, BodyUnknown) else "face"
            self.logger.debug(
                f"Assigned person_id {i} to person at x={coord:.1f} (sorted by {sort_basis})"
            )

        self.logger.info(
            f"🔢 Assigned person_id values to {len(persons)} persons using left-to-right ordering"
        )
