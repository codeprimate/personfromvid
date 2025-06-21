"""Image writing and processing for output generation.

This module implements the ImageWriter class that handles high-quality image
processing and file I/O operations for generating the final output images.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..data.context import ProcessingContext
from ..data.detection_results import FaceDetection
from ..data.frame_data import FrameData
from ..utils.exceptions import ImageWriteError
from ..utils.logging import get_logger
from .naming_convention import NamingConvention

if TYPE_CHECKING:
    from ..analysis.person_selector import PersonSelection
    from ..data.person import Person


class ImageWriter:
    """Handles image processing and file writing for output generation."""

    def __init__(self, context: ProcessingContext):
        """Initialize image writer.

        Args:
            context: ProcessingContext with unified pipeline data
        """
        self.config = context.config.output.image
        self.output_directory = context.output_directory
        self.video_base_name = context.video_base_name
        self.logger = get_logger(__name__)
        self.naming = NamingConvention(context=context)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._validate_config()

    def save_frame_outputs(self, frame: FrameData) -> List[str]:
        """Save all output images for a frame using enhanced selection metadata.

        Args:
            frame: Frame data to process with selection metadata

        Returns:
            List of output file paths that were created
        """
        # Guard clause: Check for new enhanced selection data
        primary_category = frame.selections.primary_selection_category
        rank = frame.selections.selection_rank

        if not primary_category or rank is None:
            # Fall back to legacy behavior for backward compatibility
            return self._save_frame_outputs_legacy(frame)

        output_files = []
        # Determine the correct file extension
        file_format = self.config.format.lower()
        extension = "jpg" if file_format == "jpeg" else file_format

        try:
            # Load the source image
            source_image = self._load_frame_image(frame)

            # Generate outputs for ALL pose categories the frame was selected for
            pose_categories = frame.selections.selected_for_poses
            if pose_categories:
                # Generate full frames if pose cropping is disabled OR if full_frames is enabled
                if not self.config.enable_pose_cropping or self.config.full_frames:
                    for category in pose_categories:
                        filename = self.naming.get_full_frame_filename(
                            frame, category, rank, extension
                        )
                        output_path = self.naming.get_full_output_path(filename)

                        # Apply resize if configured
                        resized_image = self._apply_resize(source_image)
                        self._save_image(resized_image, output_path)
                        output_files.append(str(output_path))
                        self.logger.debug(f"Saved full frame: {filename}")

                # Generate pose crop if enabled (replaces full frames unless full_frames is enabled)
                if self.config.enable_pose_cropping and frame.pose_detections:
                    best_pose = frame.get_best_pose()
                    if best_pose:
                        pose_crop = self._crop_region(
                            source_image, best_pose.bbox, self.config.pose_crop_padding
                        )

                        # Generate crop for the primary pose category only to avoid duplicates
                        primary_pose_category = next(
                            (
                                cat
                                for cat in pose_categories
                                if f"pose_{cat}" == primary_category
                            ),
                            pose_categories[0],  # fallback to first pose category
                        )
                        base_filename = self.naming.get_full_frame_filename(
                            frame, primary_pose_category, rank, extension
                        )
                        crop_filename = self.naming.get_crop_suffixed_filename(
                            base_filename
                        )
                        crop_output_path = self.naming.get_full_output_path(
                            crop_filename
                        )

                        self._save_image(pose_crop, crop_output_path)
                        output_files.append(str(crop_output_path))
                        self.logger.debug(f"Saved pose crop: {crop_filename}")

            # Generate outputs for ALL head angle categories the frame was selected for
            head_angle_categories = frame.selections.selected_for_head_angles
            if (
                self.config.face_crop_enabled
                and head_angle_categories
                and frame.face_detections
            ):
                best_face = frame.get_best_face()
                if best_face:
                    face_crop = self._crop_face(source_image, best_face)

                    # Only generate ONE face crop per frame using the primary head angle category
                    # to avoid duplicate face images with different category names
                    primary_head_angle = None
                    if "head_angle_" in primary_category:
                        # Extract the head angle part from primary_selection_category
                        primary_head_angle = primary_category.replace("head_angle_", "")

                    # Use primary head angle if available, otherwise first head angle category
                    face_category = (
                        primary_head_angle
                        if primary_head_angle in head_angle_categories
                        else head_angle_categories[0]
                    )

                    # Generate face crop with simplified naming (no shot type combinations)
                    filename = self.naming.get_face_crop_filename(
                        frame, face_category, rank, extension, shot_type=None
                    )
                    output_path = self.naming.get_full_output_path(filename)

                    self._save_image(face_crop, output_path)
                    output_files.append(str(output_path))
                    self.logger.debug(f"Saved face crop: {filename}")

            # Update frame's output files list and mark as final output if files were created
            if output_files:
                frame.selections.output_files.extend(output_files)
                frame.selections.final_output = True

            return output_files

        except Exception as e:
            error_msg = f"Failed to save outputs for frame {frame.frame_id}: {e}"
            self.logger.error(error_msg)
            raise ImageWriteError(error_msg) from e

    def save_person_outputs(self, person_selection: "PersonSelection") -> List[str]:
        """Save all output images for a PersonSelection object.

        Args:
            person_selection: PersonSelection object containing frame, person, and metadata

        Returns:
            List of output file paths that were created
        """

        frame = person_selection.frame_data
        person = person_selection.person
        person_id = person_selection.person_id

        output_files = []

        # Determine the correct file extension
        file_format = self.config.format.lower()
        extension = "jpg" if file_format == "jpeg" else file_format

        # Generate rank based on selection score (higher score = lower rank number)
        rank = 1  # For now, use rank 1 - could be enhanced later for multiple selections per person

        try:
            # Load the source image
            source_image = self._load_frame_image(frame)

            # Import sentinel classes for proper type checking
            from ..data.person import BodyUnknown, FaceUnknown

            # Generate person-specific outputs based on person detection types

            # Get the primary pose classification for this person
            primary_pose = self._get_primary_pose_classification(person)

            # Get the head direction for face crops
            head_direction = self._get_head_direction(person)

            # Get shot type from frame closeup detections
            shot_type = self._get_shot_type_from_frame(frame)

            # Person face crop (if person has face detection)
            if (
                not isinstance(person.face, FaceUnknown)
                and self.config.face_crop_enabled
            ):
                face_crop = self._crop_face(source_image, person.face)

                # Use head direction as category for person-based face crops (simplified - no shot type)
                filename = self.naming.get_face_crop_filename(
                    frame,
                    head_direction,
                    rank,
                    extension,
                    person_id=person_id,
                    shot_type=None,  # Removed shot type to avoid combinations
                )
                output_path = self.naming.get_full_output_path(filename)

                self._save_image(face_crop, output_path)
                output_files.append(str(output_path))
                self.logger.debug(
                    f"Saved person {person_id} face crop ({head_direction}): {filename}"
                )

            # Person body crop (if person has body detection and pose cropping enabled)
            if (
                not isinstance(person.body, BodyUnknown)
                and self.config.enable_pose_cropping
            ):
                body_crop = self._crop_region(
                    source_image, person.body.bbox, self.config.pose_crop_padding
                )

                # Use pose classification as category for person-based body crops with head direction and shot type
                base_filename = self.naming.get_full_frame_filename(
                    frame,
                    primary_pose,
                    rank,
                    extension,
                    person_id=person_id,
                    head_direction=head_direction,
                    shot_type=shot_type,
                )
                crop_filename = self.naming.get_crop_suffixed_filename(base_filename)
                crop_output_path = self.naming.get_full_output_path(crop_filename)

                self._save_image(body_crop, crop_output_path)
                output_files.append(str(crop_output_path))
                self.logger.debug(
                    f"Saved person {person_id} body crop ({primary_pose}, {head_direction}, {shot_type}): {crop_filename}"
                )

            # Full frame with person annotation (if pose cropping disabled OR full_frames enabled)
            if not self.config.enable_pose_cropping or self.config.full_frames:
                # Apply resize if configured
                resized_image = self._apply_resize(source_image)

                # Use pose classification as category for person-based full frames with head direction and shot type
                filename = self.naming.get_full_frame_filename(
                    frame,
                    primary_pose,
                    rank,
                    extension,
                    person_id=person_id,
                    head_direction=head_direction,
                    shot_type=shot_type,
                )
                output_path = self.naming.get_full_output_path(filename)

                self._save_image(resized_image, output_path)
                output_files.append(str(output_path))
                self.logger.debug(
                    f"Saved person {person_id} full frame ({primary_pose}, {head_direction}, {shot_type}): {filename}"
                )

            return output_files

        except Exception as e:
            error_msg = f"Failed to save outputs for person {person_id} in frame {frame.frame_id}: {e}"
            self.logger.error(error_msg)
            raise ImageWriteError(error_msg) from e

    def _get_primary_pose_classification(self, person: "Person") -> str:
        """Get the primary pose classification for a person.

        Args:
            person: Person object with body detection

        Returns:
            Primary pose classification (highest confidence) or "unknown"
        """
        from ..data.person import BodyUnknown

        if isinstance(person.body, BodyUnknown) or not person.body.pose_classifications:
            return "unknown"

        # Get pose classification with highest confidence
        primary_pose, confidence = max(
            person.body.pose_classifications, key=lambda x: x[1]
        )

        self.logger.debug(
            f"Primary pose for person {person.person_id}: {primary_pose} (confidence: {confidence:.3f})"
        )
        return primary_pose

    def _get_head_direction(self, person: "Person") -> str:
        """Get the head direction for a person.

        Args:
            person: Person object with head pose information

        Returns:
            Head direction formatted for filename or "unknown"
        """
        if person.head_pose and person.head_pose.direction:
            direction = person.head_pose.direction
            # Format for filename (lowercase with dashes)
            formatted_direction = direction.replace(" ", "-").lower()
            self.logger.debug(
                f"Head direction for person {person.person_id}: {direction} -> {formatted_direction}"
            )
            return formatted_direction

        # Fallback: try to infer from yaw angle if head_pose exists
        if person.head_pose:
            yaw = person.head_pose.yaw
            if abs(yaw) <= 30:
                direction = "front"
            elif yaw > 30:
                direction = "profile-right"
            elif yaw < -30:
                direction = "profile-left"
            else:
                direction = "unknown"

            self.logger.debug(
                f"Inferred head direction for person {person.person_id} from yaw {yaw:.1f}°: {direction}"
            )
            return direction

        self.logger.debug(
            f"No head pose information for person {person.person_id}, using 'unknown'"
        )
        return "unknown"

    def _get_shot_type_from_frame(self, frame: "FrameData") -> str:
        """Get the shot type from frame closeup detections.

        Args:
            frame: FrameData with closeup detections

        Returns:
            Shot type formatted for filename or "unknown"
        """
        if not frame.closeup_detections:
            return "unknown"

        # Get closeup detection with highest confidence
        best_closeup = max(frame.closeup_detections, key=lambda c: c.confidence)
        shot_type = best_closeup.shot_type

        # Format for filename (lowercase with dashes)
        formatted_shot_type = (
            shot_type.replace(" ", "-").lower() if shot_type else "unknown"
        )

        self.logger.debug(
            f"Frame {frame.frame_id} shot type: {shot_type} -> {formatted_shot_type} (confidence: {best_closeup.confidence:.3f})"
        )
        return formatted_shot_type

    def _save_frame_outputs_legacy(self, frame: FrameData) -> List[str]:
        """Legacy save method for backward compatibility with old selection data.

        This method uses the old selection fields (selected_for_poses, selected_for_head_angles)
        to maintain compatibility with frames that haven't been processed by the enhanced
        frame selection system.

        Args:
            frame: Frame data with legacy selection metadata

        Returns:
            List of output file paths that were created
        """
        pose_categories = frame.selections.selected_for_poses
        head_angle_categories = frame.selections.selected_for_head_angles
        output_files = []

        # Determine the correct file extension
        file_format = self.config.format.lower()
        extension = "jpg" if file_format == "jpeg" else file_format

        try:
            # Load the source image
            source_image = self._load_frame_image(frame)

            # Save full frame images for pose categories (legacy) - if pose cropping disabled OR full_frames enabled
            if pose_categories and (not self.config.enable_pose_cropping or self.config.full_frames):
                for category in pose_categories:
                    rank = frame.selections.selection_rank or 1
                    filename = self.naming.get_full_frame_filename(
                        frame, category, rank, extension
                    )
                    output_path = self.naming.get_full_output_path(filename)

                    # Apply resize if configured
                    resized_image = self._apply_resize(source_image)
                    self._save_image(resized_image, output_path)
                    output_files.append(str(output_path))
                    self.logger.debug(f"Saved full frame (legacy): {filename}")

            # Save face crop images for head angle categories (legacy)
            if (
                self.config.face_crop_enabled
                and head_angle_categories
                and frame.face_detections
            ):
                best_face = frame.get_best_face()
                if best_face:
                    face_crop = self._crop_face(source_image, best_face)

                    # Generate face crop for the primary head angle category only (simplified)
                    primary_category = head_angle_categories[0]  # Use first/primary category
                    rank = frame.selections.selection_rank or 1
                    filename = self.naming.get_face_crop_filename(
                        frame, primary_category, rank, extension, shot_type=None
                    )
                    output_path = self.naming.get_full_output_path(filename)

                    # Face crops already handle resize logic in _crop_face, so save directly
                    self._save_image(face_crop, output_path)
                    output_files.append(str(output_path))
                    self.logger.debug(f"Saved face crop (legacy): {filename}")

            # Generate pose crop if enabled (replaces full frames unless full_frames is enabled) - legacy
            if (
                self.config.enable_pose_cropping
                and pose_categories
                and frame.pose_detections
            ):
                best_pose = frame.get_best_pose()
                if best_pose:
                    pose_crop = self._crop_region(
                        source_image, best_pose.bbox, self.config.pose_crop_padding
                    )

                    # Generate crop for the first pose category (legacy behavior)
                    category = pose_categories[0]
                    rank = frame.selections.selection_rank or 1
                    base_filename = self.naming.get_full_frame_filename(
                        frame, category, rank, extension
                    )
                    crop_filename = self.naming.get_crop_suffixed_filename(
                        base_filename
                    )
                    crop_output_path = self.naming.get_full_output_path(crop_filename)

                    self._save_image(pose_crop, crop_output_path)
                    output_files.append(str(crop_output_path))
                    self.logger.debug(f"Saved pose crop (legacy): {crop_filename}")

            # Update frame's output files list
            frame.selections.output_files.extend(output_files)

            return output_files

        except Exception as e:
            error_msg = (
                f"Failed to save outputs for frame {frame.frame_id} (legacy): {e}"
            )
            self.logger.error(error_msg)
            raise ImageWriteError(error_msg) from e

    def _load_frame_image(self, frame: FrameData) -> np.ndarray:
        """Load frame image from cache or file.

        Args:
            frame: Frame data containing the file path and cached image.

        Returns:
            Image as a numpy array in RGB format.
        """
        if frame.image is None:
            raise ImageWriteError(
                f"Frame file not found or failed to load: {frame.file_path}"
            )

        try:
            # The cached frame.image is in BGR format, convert to RGB for PIL
            image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            return image_rgb

        except Exception as e:
            raise ImageWriteError(
                f"Error converting frame image {frame.frame_id} to RGB: {e}"
            ) from e

    def _crop_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float
    ) -> np.ndarray:
        """Crop region from image with padding and upscale to minimum size.

        Args:
            image: Source image as numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            padding: Padding factor as proportion of bbox size (0.0 to 1.0)

        Returns:
            Cropped and potentially upscaled image as numpy array
        """
        x1, y1, x2, y2 = bbox

        # Calculate padding
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        padding_x = int(bbox_width * padding)
        padding_y = int(bbox_height * padding)

        # Apply padding with bounds checking
        img_height, img_width = image.shape[:2]
        crop_x1 = max(0, x1 - padding_x)
        crop_y1 = max(0, y1 - padding_y)
        crop_x2 = min(img_width, x2 + padding_x)
        crop_y2 = min(img_height, y2 + padding_y)

        # Crop the image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Determine minimum dimension: use resize value if configured, otherwise default to 512
        min_dimension = self.config.resize if self.config.resize is not None else 512

        # Upscale if both dimensions are smaller than the minimum dimension
        crop_height, crop_width = cropped.shape[:2]

        if crop_height < min_dimension and crop_width < min_dimension:
            # Calculate scale factor to ensure at least one dimension equals min_dimension
            scale_factor = min_dimension / max(crop_width, crop_height)
            new_width = int(crop_width * scale_factor)
            new_height = int(crop_height * scale_factor)

            # Convert to PIL for high-quality Lanczos upscaling
            pil_image = Image.fromarray(cropped)
            upscaled_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            cropped = np.array(upscaled_pil)

            self.logger.debug(
                f"Upscaled crop from {crop_width}x{crop_height} to {new_width}x{new_height} "
                f"(scale: {scale_factor:.2f}, target: {min_dimension}px)"
            )

        return cropped

    def _crop_face(
        self, image: np.ndarray, face_detection: FaceDetection
    ) -> np.ndarray:
        """Crop face from image with padding and upscale to minimum size.

        Args:
            image: Source image as numpy array
            face_detection: Face detection with bounding box

        Returns:
            Cropped and potentially upscaled face image as numpy array
        """
        return self._crop_region(
            image, face_detection.bbox, self.config.face_crop_padding
        )

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save image to file with appropriate format and quality settings.

        Args:
            image: Image as numpy array (RGB format)
            output_path: Path to save the image
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)

            # Ensure RGB mode
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Save with format-specific settings
            if self.config.format.lower() == "png":
                pil_image.save(
                    output_path, format="PNG", optimize=self.config.png.optimize
                )
            elif self.config.format.lower() in ["jpg", "jpeg"]:
                pil_image.save(
                    output_path,
                    format="JPEG",
                    quality=self.config.jpeg.quality,
                    optimize=True,
                )
            else:
                raise ImageWriteError(f"Unsupported image format: {self.config.format}")

            self.logger.debug(
                f"Saved image: {output_path} ({self.config.format.upper()})"
            )

        except Exception as e:
            raise ImageWriteError(f"Failed to save image {output_path}: {e}") from e

    def _validate_config(self) -> None:
        """Validate the output configuration."""
        if self.config.format.lower() not in ["png", "jpg", "jpeg"]:
            raise ValueError(f"Unsupported output format: {self.config.format}")

        if not (0.0 <= self.config.face_crop_padding <= 1.0):
            raise ValueError(
                f"Face crop padding must be between 0.0 and 1.0, got: {self.config.face_crop_padding}"
            )

        if not (0.0 <= self.config.pose_crop_padding <= 1.0):
            raise ValueError(
                f"Pose crop padding must be between 0.0 and 1.0, got: {self.config.pose_crop_padding}"
            )

        if self.config.format.lower() in ["jpg", "jpeg"]:
            if not (70 <= self.config.jpeg.quality <= 100):
                raise ValueError(
                    f"JPEG quality must be between 70 and 100, got: {self.config.jpeg.quality}"
                )

        if self.config.resize is not None:
            if not (256 <= self.config.resize <= 4096):
                raise ValueError(
                    f"Resize dimension must be between 256 and 4096, got: {self.config.resize}"
                )

    def get_output_statistics(self) -> dict:
        """Get statistics about output generation.

        Returns:
            Dictionary with output statistics
        """
        return {
            "output_directory": str(self.output_directory),
            "video_base_name": self.video_base_name,
            "format": self.config.format,
            "face_crop_enabled": self.config.face_crop_enabled,
            "face_crop_padding": self.config.face_crop_padding,
            "enable_pose_cropping": self.config.enable_pose_cropping,
            "pose_crop_padding": self.config.pose_crop_padding,
            "resize": self.config.resize,
        }

    def _apply_resize(self, image: np.ndarray) -> np.ndarray:
        """Apply resize to image based on configuration.

        Args:
            image: Image as numpy array

        Returns:
            Resized image as numpy array
        """
        if self.config.resize is None:
            return image

        height, width = image.shape[:2]
        max_dimension = self.config.resize

        # Only resize if image is larger than the target size
        if max(width, height) > max_dimension:
            # Calculate scale factor to ensure the larger dimension equals max_dimension
            scale_factor = max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Convert to PIL for high-quality Lanczos downscaling
            pil_image = Image.fromarray(image)
            resized_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            resized_image = np.array(resized_pil)

            self.logger.debug(
                f"Resized image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.2f})"
            )
            return resized_image
        else:
            return image
