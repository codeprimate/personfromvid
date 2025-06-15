"""Pose estimation inference using YOLOv8-Pose and similar models.

This module provides human pose estimation capabilities using state-of-the-art models
like YOLOv8-Pose. It supports both CPU and GPU inference with automatic model
downloading and caching.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import cv2
import time

from ..data.detection_results import PoseDetection
from ..utils.exceptions import ModelInferenceError, PoseEstimationError
from .model_configs import ModelConfigs, ModelFormat
from .model_manager import get_model_manager

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_DEVICE = "cpu"
MINIMUM_POSE_SIZE = 50  # Minimum pose bounding box size in pixels

# COCO pose keypoint names (17 keypoints for YOLOv8-Pose)
COCO_KEYPOINT_NAMES = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
]


class PoseEstimator:
    """Pose estimation inference using YOLOv8-Pose and similar models.

    This class provides high-performance pose estimation with support for:
    - YOLOv8-Pose models (PyTorch format)
    - 17-point COCO pose keypoints
    - Batch processing for improved efficiency
    - Confidence thresholding and validation
    - CPU and GPU acceleration
    - Keypoint normalization and standardization

    Examples:
        Basic usage:
        >>> estimator = PoseEstimator("yolov8n-pose")
        >>> poses = estimator.estimate_pose(image)

        Batch processing:
        >>> poses_batch = estimator.estimate_batch([img1, img2, img3])

        Custom confidence threshold:
        >>> estimator.set_confidence_threshold(0.8)
        >>> poses = estimator.estimate_pose(image)

        Get keypoint names:
        >>> keypoint_names = estimator.get_keypoint_names()
    """

    def __init__(
        self,
        model_name: str,
        device: str = DEFAULT_DEVICE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        config: Optional["Config"] = None,
    ):
        """Initialize pose estimator with specified model.

        Args:
            model_name: Name of the pose estimation model to use
            device: Computation device ("cpu", "cuda", or "auto")
            confidence_threshold: Minimum confidence threshold for detections
            config: Application configuration object (optional)

        Raises:
            PoseEstimationError: If model loading fails or device is unsupported
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.confidence_threshold = confidence_threshold
        
        # Store config or get default
        if config is None:
            from ..data.config import get_default_config
            self.config = get_default_config()
        else:
            self.config = config

        # Get model configuration
        self.model_config = ModelConfigs.get_model(model_name)
        if not self.model_config:
            raise PoseEstimationError(f"Unknown pose estimation model: {model_name}")

        # Validate device support
        from ..data.config import DeviceType

        device_type = DeviceType.CPU if self.device == "cpu" else DeviceType.GPU
        if not self.model_config.is_device_supported(device_type):
            raise PoseEstimationError(
                f"Model {model_name} does not support device {device}"
            )

        # Download and cache model
        self.model_manager = get_model_manager()
        self.model_path = self.model_manager.ensure_model_available(model_name)

        # Initialize model inference engine
        self._model = None
        self._input_size = self.model_config.input_size
        self._model_format = self.model_config.files[0].format
        self._keypoint_names = COCO_KEYPOINT_NAMES.copy()

        # Initialize pose classifier
        from ..analysis.pose_classifier import PoseClassifier

        self._pose_classifier = PoseClassifier()

        logger.info(f"Initialized PoseEstimator with model {model_name} on {device}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            # Check for CUDA availability
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the pose estimation model for inference.

        Raises:
            PoseEstimationError: If model loading fails
        """
        if self._model is not None:
            return

        try:
            if self._model_format == ModelFormat.PYTORCH:
                self._load_pytorch_model()
            elif self._model_format == ModelFormat.ONNX:
                self._load_onnx_model()
            else:
                raise PoseEstimationError(
                    f"Unsupported model format: {self._model_format}"
                )

            logger.debug(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            raise PoseEstimationError(
                f"Failed to load model {self.model_name}: {str(e)}"
            ) from e

    def _load_pytorch_model(self) -> None:
        """Load PyTorch model using ultralytics or torch."""
        try:
            # Check if it's a YOLO model
            if "yolo" in self.model_name.lower():
                from ultralytics import YOLO

                self._model = YOLO(str(self.model_path))
                if self.device == "cuda":
                    self._model.to("cuda")
            else:
                import torch

                self._model = torch.load(str(self.model_path), map_location=self.device)
                if hasattr(self._model, "eval"):
                    self._model.eval()

        except ImportError as e:
            raise PoseEstimationError(
                "Required dependencies not installed. For YOLO: pip install ultralytics"
            ) from e

    def _load_onnx_model(self) -> None:
        """Load ONNX model using ONNXRuntime."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise PoseEstimationError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            ) from e

        # Set up providers based on device
        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Create inference session
        self._model = ort.InferenceSession(str(self.model_path), providers=providers)

        # Get input/output info
        self._input_name = self._model.get_inputs()[0].name
        self._output_names = [output.name for output in self._model.get_outputs()]

        logger.debug(f"ONNX model loaded with providers: {self._model.get_providers()}")

    def estimate_pose(self, image: np.ndarray) -> List[PoseDetection]:
        """Estimate pose in a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of pose detections with keypoints and confidence scores

        Raises:
            PoseEstimationError: If pose estimation fails
        """
        if image is None or image.size == 0:
            raise PoseEstimationError("Input image is empty or None")

        # Ensure model is loaded
        self._load_model()

        try:
            # Get raw pose detections
            if self._model_format == ModelFormat.PYTORCH:
                detections = self._estimate_pytorch(image)
            elif self._model_format == ModelFormat.ONNX:
                detections = self._estimate_onnx(image)
            else:
                raise PoseEstimationError(
                    f"Unsupported model format: {self._model_format}"
                )

            # Add pose classification
            image_shape = (image.shape[0], image.shape[1])  # (height, width)
            classified_detections = self.classify_pose_detections(
                detections, image_shape
            )

            return classified_detections

        except Exception as e:
            raise PoseEstimationError(f"Pose estimation failed: {str(e)}") from e

    def estimate_batch(self, images: List[np.ndarray]) -> List[List[PoseDetection]]:
        """Estimate poses in a batch of images for improved efficiency.

        Args:
            images: List of input images as numpy arrays (H, W, C) in BGR format

        Returns:
            List of pose detection lists, one per input image

        Raises:
            PoseEstimationError: If batch pose estimation fails
        """
        if not images:
            return []

        # Validate all images
        for i, image in enumerate(images):
            if image is None or image.size == 0:
                raise PoseEstimationError(f"Input image at index {i} is empty or None")

        # Ensure model is loaded
        self._load_model()

        try:
            # Get raw pose detections
            if self._model_format == ModelFormat.PYTORCH:
                batch_detections = self._estimate_batch_pytorch(images)
            elif self._model_format == ModelFormat.ONNX:
                batch_detections = self._estimate_batch_onnx(images)
            else:
                raise PoseEstimationError(
                    f"Unsupported model format: {self._model_format}"
                )

            # Add pose classification for each image
            classified_batch = []
            for i, (image, detections) in enumerate(zip(images, batch_detections)):
                image_shape = (image.shape[0], image.shape[1])  # (height, width)
                classified_detections = self.classify_pose_detections(
                    detections, image_shape
                )
                classified_batch.append(classified_detections)

            return classified_batch

        except Exception as e:
            raise PoseEstimationError(f"Batch pose estimation failed: {str(e)}") from e

    def _estimate_pytorch(self, image: np.ndarray) -> List[PoseDetection]:
        """Estimate poses using PyTorch model (typically YOLO)."""
        if "yolo" in self.model_name.lower():
            # Use ultralytics YOLO interface
            results = self._model(image, verbose=False)
            return self._postprocess_yolo_results(results[0], image.shape[:2])
        else:
            # Generic PyTorch model - would need specific implementation
            raise PoseEstimationError(
                f"Generic PyTorch models not yet implemented for {self.model_name}"
            )

    def _estimate_onnx(self, image: np.ndarray) -> List[PoseDetection]:
        """Estimate poses using ONNX model."""
        # Preprocess image
        input_image = self._preprocess_image(image)

        # Run inference
        outputs = self._model.run(self._output_names, {self._input_name: input_image})

        # Post-process results
        return self._postprocess_onnx_results(outputs, image.shape[:2])

    def _estimate_batch_pytorch(
        self, images: List[np.ndarray]
    ) -> List[List[PoseDetection]]:
        """Batch estimation for PyTorch models."""
        if "yolo" in self.model_name.lower():
            # YOLO models support native batch processing
            results = self._model(images, verbose=False)

            batch_poses = []
            for i, result in enumerate(results):
                poses = self._postprocess_yolo_results(result, images[i].shape[:2])
                batch_poses.append(poses)

            return batch_poses
        else:
            # Generic PyTorch batch processing
            results = []
            for image in images:
                poses = self._estimate_pytorch(image)
                results.append(poses)
            return results

    def _estimate_batch_onnx(
        self, images: List[np.ndarray]
    ) -> List[List[PoseDetection]]:
        """Batch estimation for ONNX models."""
        results = []

        # ONNX models typically process single images, so iterate
        # In future, could implement true batching for models that support it
        for image in images:
            poses = self._estimate_onnx(image)
            results.append(poses)

        return results

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model inference.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image tensor ready for inference
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb_image, self._input_size)

        # Normalize to [0, 1] and convert to float32
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

        return input_tensor

    def _postprocess_yolo_results(
        self, result, image_shape: Tuple[int, int]
    ) -> List[PoseDetection]:
        """Post-process YOLO model results to extract pose detections.

        Args:
            result: YOLO detection result object
            image_shape: Original image shape (height, width)

        Returns:
            List of validated pose detections
        """
        detections = []

        if (
            hasattr(result, "boxes")
            and result.boxes is not None
            and hasattr(result, "keypoints")
            and result.keypoints is not None
        ):
            boxes = result.boxes
            keypoints = result.keypoints

            # Extract bounding boxes and scores
            if (
                hasattr(boxes, "xyxy")
                and hasattr(boxes, "conf")
                and hasattr(keypoints, "xy")
                and hasattr(keypoints, "conf")
                and keypoints.xy is not None
                and keypoints.conf is not None
            ):
                bboxes = boxes.xyxy.cpu().numpy()  # (N, 4) in xyxy format
                scores = boxes.conf.cpu().numpy()  # (N,)
                kpts = keypoints.xy.cpu().numpy()  # (N, 17, 2) keypoint coordinates
                kpts_conf = keypoints.conf.cpu().numpy()  # (N, 17) keypoint confidences

                # Filter by confidence
                valid_indices = scores >= self.confidence_threshold

                if np.any(valid_indices):
                    valid_bboxes = bboxes[valid_indices]
                    valid_scores = scores[valid_indices]
                    valid_kpts = kpts[valid_indices]
                    valid_kpts_conf = kpts_conf[valid_indices]

                    for i in range(len(valid_bboxes)):
                        bbox = valid_bboxes[i]
                        score = float(valid_scores[i])
                        person_kpts = valid_kpts[i]
                        person_kpts_conf = valid_kpts_conf[i]

                        # Convert to integers and validate bbox
                        x1 = max(0, int(bbox[0]))
                        y1 = max(0, int(bbox[1]))
                        x2 = min(image_shape[1], int(bbox[2]))
                        y2 = min(image_shape[0], int(bbox[3]))

                        # Validate bbox dimensions
                        if x2 > x1 and y2 > y1:
                            # Process keypoints
                            keypoints_dict = {}
                            for j, keypoint_name in enumerate(self._keypoint_names):
                                if j < len(person_kpts):
                                    x = float(person_kpts[j][0])
                                    y = float(person_kpts[j][1])
                                    conf = float(person_kpts_conf[j])
                                    keypoints_dict[keypoint_name] = (x, y, conf)

                            # Validate keypoints (ensure they're reasonable)
                            keypoints_dict = self._validate_and_normalize_keypoints(
                                keypoints_dict, image_shape
                            )

                            detection = PoseDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=score,
                                keypoints=keypoints_dict,
                                pose_classifications=[],  # Classification added later
                            )
                            detections.append(detection)

        return detections

    def _postprocess_onnx_results(
        self, outputs: List[np.ndarray], image_shape: Tuple[int, int]
    ) -> List[PoseDetection]:
        """Post-process ONNX model outputs to extract pose detections.

        Args:
            outputs: Raw model outputs
            image_shape: Original image shape (height, width)

        Returns:
            List of validated pose detections
        """
        detections = []

        # Standard YOLO pose outputs format:
        # Output 0: Detections (N, 56) - [x1, y1, x2, y2, conf, 51 keypoint values (17*3)]
        if len(outputs) >= 1:
            predictions = outputs[0]  # Shape: (1, N, 56) or (N, 56)

            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension

            # Filter by confidence (5th column)
            valid_indices = predictions[:, 4] >= self.confidence_threshold

            if np.any(valid_indices):
                valid_predictions = predictions[valid_indices]

                # Scale factors back to original image size
                height, width = image_shape
                scale_x = width / self._input_size[0]
                scale_y = height / self._input_size[1]

                for pred in valid_predictions:
                    # Extract bbox and confidence
                    x1 = max(0, int(pred[0] * scale_x))
                    y1 = max(0, int(pred[1] * scale_y))
                    x2 = min(width, int(pred[2] * scale_x))
                    y2 = min(height, int(pred[3] * scale_y))
                    conf = float(pred[4])

                    # Validate bbox dimensions
                    if x2 > x1 and y2 > y1:
                        # Extract keypoints (17 keypoints * 3 values each = 51 values)
                        keypoints_dict = {}
                        for i, keypoint_name in enumerate(self._keypoint_names):
                            if 5 + i * 3 + 2 < len(pred):  # Ensure we have enough data
                                x = float(pred[5 + i * 3] * scale_x)
                                y = float(pred[5 + i * 3 + 1] * scale_y)
                                kpt_conf = float(pred[5 + i * 3 + 2])
                                keypoints_dict[keypoint_name] = (x, y, kpt_conf)

                        # Validate keypoints
                        keypoints_dict = self._validate_and_normalize_keypoints(
                            keypoints_dict, image_shape
                        )

                        detection = PoseDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            keypoints=keypoints_dict,
                            pose_classifications=[],
                        )
                        detections.append(detection)

        return detections

    def _validate_and_normalize_keypoints(
        self,
        keypoints: Dict[str, Tuple[float, float, float]],
        image_shape: Tuple[int, int],
    ) -> Dict[str, Tuple[float, float, float]]:
        """Validate and normalize keypoints.

        Args:
            keypoints: Dictionary of keypoint coordinates and confidence
            image_shape: Original image shape (height, width)

        Returns:
            Validated and normalized keypoints dictionary
        """
        height, width = image_shape
        validated_keypoints = {}

        for name, (x, y, conf) in keypoints.items():
            # Validate confidence
            if conf < 0.0 or conf > 1.0:
                conf = max(0.0, min(1.0, conf))

            # Validate coordinates are within image bounds
            if 0 <= x <= width and 0 <= y <= height:
                validated_keypoints[name] = (x, y, conf)
            else:
                # Clamp to image bounds but reduce confidence
                x_clamped = max(0, min(width, x))
                y_clamped = max(0, min(height, y))
                validated_keypoints[name] = (x_clamped, y_clamped, conf * 0.5)

        return validated_keypoints

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for pose detections.

        Args:
            threshold: New confidence threshold (0.0 to 1.0)

        Raises:
            ValueError: If threshold is outside valid range
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {threshold}"
            )

        self.confidence_threshold = threshold
        logger.debug(f"Updated confidence threshold to {threshold}")

    def get_keypoint_names(self) -> List[str]:
        """Get the list of keypoint names supported by this model.

        Returns:
            List of keypoint names in order
        """
        return self._keypoint_names.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata and configuration
        """
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self._input_size,
            "model_format": self._model_format.value if self._model_format else None,
            "model_loaded": self._model is not None,
            "keypoint_names": self._keypoint_names,
            "num_keypoints": len(self._keypoint_names),
        }

    def validate_pose_detection(
        self, detection: PoseDetection, image_shape: Tuple[int, int]
    ) -> bool:
        """Validate a pose detection result.

        Args:
            detection: Pose detection to validate
            image_shape: Original image shape (height, width)

        Returns:
            True if detection is valid, False otherwise
        """
        height, width = image_shape
        x1, y1, x2, y2 = detection.bbox

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False

        # Check dimensions
        if x2 <= x1 or y2 <= y1:
            return False

        # Check minimum size (person should be at least 50x50 pixels)
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return False

        # Check confidence
        if detection.confidence < self.confidence_threshold:
            return False

        # Check if we have any high-confidence keypoints
        high_conf_keypoints = [kp for kp in detection.keypoints.values() if kp[2] > 0.5]
        if len(high_conf_keypoints) < 3:  # Need at least 3 good keypoints
            return False

        return True

    def calculate_pose_confidence(self, detection: PoseDetection) -> float:
        """Calculate overall pose confidence based on keypoint confidences.

        Args:
            detection: Pose detection with keypoints

        Returns:
            Overall pose confidence score (0.0 to 1.0)
        """
        if not detection.keypoints:
            return 0.0

        # Get all keypoint confidences
        keypoint_confidences = [kp[2] for kp in detection.keypoints.values()]

        # Filter out very low confidence keypoints
        valid_confidences = [conf for conf in keypoint_confidences if conf > 0.1]

        if not valid_confidences:
            return 0.0

        # Calculate weighted average (emphasize higher confidence keypoints)
        sorted_confidences = sorted(valid_confidences, reverse=True)

        # Use top 70% of keypoints for confidence calculation
        top_count = max(1, int(len(sorted_confidences) * 0.7))
        top_confidences = sorted_confidences[:top_count]

        # Weighted average with higher weight for better keypoints
        weights = [1.0 / (i + 1) for i in range(len(top_confidences))]
        weighted_sum = sum(
            conf * weight for conf, weight in zip(top_confidences, weights)
        )
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def classify_pose_detections(
        self, detections: List[PoseDetection], image_shape: Tuple[int, int]
    ) -> List[PoseDetection]:
        """Classify pose detections using the pose classifier.

        Args:
            detections: List of pose detections to classify
            image_shape: Image dimensions (height, width)

        Returns:
            List of pose detections with classifications added
        """
        classified_detections = []

        for detection in detections:
            try:
                # Use the pose classifier to determine pose type
                classifications = self._pose_classifier._classify_single_pose(
                    detection, image_shape
                )

                # Create new detection with classification
                classified_detection = PoseDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    keypoints=detection.keypoints,
                    pose_classifications=classifications,
                )
                classified_detections.append(classified_detection)

            except Exception as e:
                logger.warning(f"Failed to classify pose: {e}")
                # Keep original detection without classification
                classified_detections.append(detection)

        return classified_detections

    def classify_poses_in_frame_data(self, frame_data: "FrameData") -> None:
        """Classify all pose detections within a FrameData object.

        Args:
            frame_data: FrameData object containing pose detections to classify
        """
        if frame_data.pose_detections:
            self._pose_classifier.classify_poses_in_frame(frame_data)

    def __del__(self):
        """Cleanup resources when estimator is destroyed."""
        self._model = None

    def process_frame_batch(
        self,
        frames_with_faces: List["FrameData"],
        progress_callback: Optional[callable] = None,
        interruption_check: Optional[callable] = None,
    ) -> Tuple[Dict[str, int], int]:
        """Process a batch of frames with pose estimation at a high level.

        This method handles:
        - Loading images from frame data
        - Running pose estimation in batches
        - Updating frame data with pose detections
        - Error handling for individual frames
        - Progress tracking with rate calculation
        - Statistics collection
        - Interruption checking

        Args:
            frames_with_faces: List of FrameData objects (typically from face detection step)
            progress_callback: Optional callback for progress updates (called with processed_count)
            interruption_check: Optional callback to check for interruption

        Returns:
            Tuple of (poses_by_category, total_poses_found)

        Raises:
            PoseEstimationError: If processing fails completely
        """
        if not frames_with_faces:
            return {}, 0

        from pathlib import Path

        total_poses_found = 0
        poses_by_category = {}
        processed_count = 0
        start_time = time.time()

        # Process in batches for memory efficiency
        batch_size = self.config.models.batch_size
        total_frames = len(frames_with_faces)
        total_batches = (total_frames + batch_size - 1) // batch_size

        logger.info(
            f"Starting body pose estimation on {total_frames} frames ({total_batches} batches)"
        )

        for i in range(0, total_frames, batch_size):
            # Check for interruption at the start of each batch
            if interruption_check:
                interruption_check()
                
            batch_num = i // batch_size + 1
            batch_frames = frames_with_faces[i : i + batch_size]
            batch_images = []
            batch_frame_data = []

            logger.debug(
                f"Processing pose estimation batch {batch_num}/{total_batches} ({len(batch_frames)} frames)"
            )

            # Load images for this batch
            import cv2

            for frame_data in batch_frames:
                # Check for interruption periodically during frame loading
                if interruption_check and len(batch_images) % 10 == 0:
                    interruption_check()
                    
                # Handle FrameData object
                image = cv2.imread(str(frame_data.file_path))
                if image is not None:
                    batch_images.append(image)
                    batch_frame_data.append(frame_data)
                else:
                    logger.warning(f"Could not load image: {frame_data.file_path}")

            # Skip batch if no valid images
            if not batch_images:
                processed_count += len(batch_frames)
                if progress_callback:
                    # Calculate current processing rate
                    elapsed = time.time() - start_time
                    current_rate = processed_count / elapsed if elapsed > 0 else 0
                    # Check if progress_callback accepts rate parameter
                    try:
                        progress_callback(processed_count, rate=current_rate)
                    except TypeError:
                        # Fallback to single argument for backwards compatibility
                        progress_callback(processed_count)
                continue

            # Check for interruption before running pose estimation
            if interruption_check:
                interruption_check()

            # Run pose estimation on batch
            try:
                logger.debug(f"Running pose inference on {len(batch_images)} images...")
                batch_pose_results = self.estimate_batch(batch_images)

                batch_poses_found = 0
                batch_classifications = {}

                # Process results for each frame in batch
                for j, (frame_data, poses) in enumerate(
                    zip(batch_frame_data, batch_pose_results)
                ):
                    # Check for interruption during result processing
                    if interruption_check and j % 5 == 0:
                        interruption_check()
                        
                    if poses:
                        batch_poses_found += len(poses)

                        # Add pose detections to frame data
                        frame_data.pose_detections.extend(poses)

                        total_poses_found += len(poses)

                        # Update statistics
                        for pose in poses:
                            if pose.pose_classifications:
                                for classification, _ in pose.pose_classifications:
                                    poses_by_category[classification] = (
                                        poses_by_category.get(classification, 0) + 1
                                    )
                                    batch_classifications[classification] = (
                                        batch_classifications.get(classification, 0) + 1
                                    )

                # Log batch results
                if batch_poses_found > 0:
                    logger.debug(
                        f"Batch {batch_num}/{total_batches}: {batch_poses_found} poses found, classifications: {batch_classifications}"
                    )
                else:
                    logger.debug(
                        f"Batch {batch_num}/{total_batches}: No poses detected"
                    )

            except Exception as e:
                logger.error(
                    f"Pose estimation failed for batch {batch_num}/{total_batches}: {e}"
                )
                # Continue with next batch rather than failing completely

            # Update progress with rate calculation
            processed_count += len(batch_frames)
            if progress_callback:
                # Calculate current processing rate
                elapsed = time.time() - start_time
                current_rate = processed_count / elapsed if elapsed > 0 else 0
                # Check if progress_callback accepts rate parameter
                try:
                    progress_callback(processed_count, rate=current_rate)
                except TypeError:
                    # Fallback to single argument for backwards compatibility
                    progress_callback(processed_count)

        logger.debug(
            f"Body pose estimation completed: {total_poses_found} poses found in {total_frames} frames"
        )
        logger.debug(f"Final pose categories: {poses_by_category}")

        return poses_by_category, total_poses_found


def create_pose_estimator(
    model_name: Optional[str] = None,
    device: str = "auto",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    config: Optional["Config"] = None,
) -> PoseEstimator:
    """Factory function to create a PoseEstimator instance.

    Args:
        model_name: Name of pose estimation model (default: use config default)
        device: Computation device preference
        confidence_threshold: Minimum confidence threshold
        config: Application configuration object (optional)

    Returns:
        Configured PoseEstimator instance

    Raises:
        PoseEstimationError: If model creation fails
    """
    if model_name is None:
        defaults = ModelConfigs.get_default_models()
        model_name = defaults["pose_estimation"]

    return PoseEstimator(model_name, device, confidence_threshold, config)
