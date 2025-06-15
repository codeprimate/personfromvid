"""Face detection inference using SCRFD or YOLO models.

This module provides face detection capabilities using state-of-the-art models
like SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
and YOLO variants. It supports both CPU and GPU inference with automatic model
downloading and caching.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import cv2
import time

from ..data.detection_results import FaceDetection
from ..utils.exceptions import ModelInferenceError, FaceDetectionError
from .model_configs import ModelConfigs, ModelFormat
from .model_manager import get_model_manager

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_DEVICE = "cpu"
MINIMUM_FACE_SIZE = 30  # Minimum face size in pixels


class FaceDetector:
    """Face detection inference using SCRFD or YOLO models.

    This class provides high-performance face detection with support for:
    - SCRFD models (ONNX format)
    - YOLO face detection models (PyTorch format)
    - Batch processing for improved efficiency
    - Confidence thresholding and validation
    - CPU and GPU acceleration

    Examples:
        Basic usage:
        >>> detector = FaceDetector("scrfd_10g")
        >>> faces = detector.detect_faces(image)

        Batch processing:
        >>> faces_batch = detector.detect_batch([img1, img2, img3])

        Custom confidence threshold:
        >>> detector.set_confidence_threshold(0.8)
        >>> faces = detector.detect_faces(image)
    """

    def __init__(
        self,
        model_name: str,
        device: str = DEFAULT_DEVICE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        config: Optional["Config"] = None,
    ):
        """Initialize face detector with specified model.

        Args:
            model_name: Name of the face detection model to use
            device: Computation device ("cpu", "cuda", or "auto")
            confidence_threshold: Minimum confidence threshold for detections
            config: Application configuration object (optional)

        Raises:
            FaceDetectionError: If model loading fails or device is unsupported
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
            raise FaceDetectionError(f"Unknown face detection model: {model_name}")

        # Validate device support
        from ..data.config import DeviceType

        device_type = DeviceType.CPU if self.device == "cpu" else DeviceType.GPU
        if not self.model_config.is_device_supported(device_type):
            raise FaceDetectionError(
                f"Model {model_name} does not support device {device}"
            )

        # Download and cache model
        self.model_manager = get_model_manager()
        self.model_path = self.model_manager.ensure_model_available(model_name)

        # Initialize model inference engine
        self._model = None
        self._input_size = self.model_config.input_size
        self._model_format = self.model_config.files[0].format

        logger.info(f"Initialized FaceDetector with model {model_name} on {device}")

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
        """Load the face detection model for inference.

        Raises:
            FaceDetectionError: If model loading fails
        """
        if self._model is not None:
            return

        try:
            if self._model_format == ModelFormat.ONNX:
                self._load_onnx_model()
            elif self._model_format == ModelFormat.PYTORCH:
                self._load_pytorch_model()
            else:
                raise FaceDetectionError(
                    f"Unsupported model format: {self._model_format}"
                )

            logger.debug(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            raise FaceDetectionError(
                f"Failed to load model {self.model_name}: {str(e)}"
            ) from e

    def _load_onnx_model(self) -> None:
        """Load ONNX model using ONNXRuntime."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise FaceDetectionError(
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
            raise FaceDetectionError(
                "Required dependencies not installed. For YOLO: pip install ultralytics"
            ) from e

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of face detections with bounding boxes and confidence scores

        Raises:
            FaceDetectionError: If detection fails
        """
        if image is None or image.size == 0:
            raise FaceDetectionError("Input image is empty or None")

        # Ensure model is loaded
        self._load_model()

        try:
            if self._model_format == ModelFormat.ONNX:
                return self._detect_onnx(image)
            elif self._model_format == ModelFormat.PYTORCH:
                return self._detect_pytorch(image)
            else:
                raise FaceDetectionError(
                    f"Unsupported model format: {self._model_format}"
                )

        except Exception as e:
            raise FaceDetectionError(f"Face detection failed: {str(e)}") from e

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces in a batch of images for improved efficiency.

        Args:
            images: List of input images as numpy arrays (H, W, C) in BGR format

        Returns:
            List of face detection lists, one per input image

        Raises:
            FaceDetectionError: If batch detection fails
        """
        if not images:
            return []

        # Validate all images
        for i, image in enumerate(images):
            if image is None or image.size == 0:
                raise FaceDetectionError(f"Input image at index {i} is empty or None")

        # Ensure model is loaded
        self._load_model()

        try:
            if self._model_format == ModelFormat.ONNX:
                return self._detect_batch_onnx(images)
            elif self._model_format == ModelFormat.PYTORCH:
                return self._detect_batch_pytorch(images)
            else:
                raise FaceDetectionError(
                    f"Unsupported model format: {self._model_format}"
                )

        except Exception as e:
            raise FaceDetectionError(f"Batch face detection failed: {str(e)}") from e

    def _detect_onnx(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using ONNX model (typically SCRFD)."""
        # Preprocess image
        input_image = self._preprocess_image(image)

        # Run inference
        outputs = self._model.run(self._output_names, {self._input_name: input_image})

        # Post-process results
        return self._postprocess_onnx_results(outputs, image.shape[:2])

    def _detect_pytorch(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using PyTorch model (typically YOLO)."""
        if "yolo" in self.model_name.lower():
            # Use ultralytics YOLO interface
            results = self._model(image, verbose=False)
            return self._postprocess_yolo_results(results[0], image.shape[:2])
        else:
            # Generic PyTorch model - would need specific implementation
            raise FaceDetectionError(
                f"Generic PyTorch models not yet implemented for {self.model_name}"
            )

    def _detect_batch_onnx(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Batch detection for ONNX models."""
        results = []

        # ONNX models typically process single images, so iterate
        # In future, could implement true batching for models that support it
        for image in images:
            faces = self._detect_onnx(image)
            results.append(faces)

        return results

    def _detect_batch_pytorch(
        self, images: List[np.ndarray]
    ) -> List[List[FaceDetection]]:
        """Batch detection for PyTorch models."""
        if "yolo" in self.model_name.lower():
            # YOLO models support native batch processing
            results = self._model(images, verbose=False)

            batch_faces = []
            for i, result in enumerate(results):
                faces = self._postprocess_yolo_results(result, images[i].shape[:2])
                batch_faces.append(faces)

            return batch_faces
        else:
            # Generic PyTorch batch processing
            results = []
            for image in images:
                faces = self._detect_pytorch(image)
                results.append(faces)
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

    def _postprocess_onnx_results(
        self, outputs: List[np.ndarray], image_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """Post-process ONNX model outputs to extract face detections.

        Args:
            outputs: Raw model outputs
            image_shape: Original image shape (height, width)

        Returns:
            List of validated face detections
        """
        detections = []

        # SCRFD multi-scale outputs: 9 outputs in groups of 3
        # Outputs 0,1,2: Scores for each scale (12800, 3200, 800)
        # Outputs 3,4,5: Bounding boxes for each scale (4 coordinates each)
        # Outputs 6,7,8: Landmarks for each scale (10 coordinates = 5 keypoints * 2)
        if len(outputs) == 9:
            # Process each scale separately
            for scale_idx in range(3):
                scores = outputs[scale_idx].flatten()  # Shape: (N,)
                bboxes = outputs[scale_idx + 3]  # Shape: (N, 4)
                landmarks = outputs[scale_idx + 6]  # Shape: (N, 10)

                # Filter by confidence
                valid_indices = scores >= self.confidence_threshold

                if np.any(valid_indices):
                    valid_bboxes = bboxes[valid_indices]
                    valid_scores = scores[valid_indices]
                    valid_landmarks = landmarks[valid_indices]

                    # Generate anchor points for this scale
                    input_height, input_width = self._input_size
                    anchors = self._generate_anchors(
                        scale_idx, input_height, input_width
                    )

                    # Only process anchors that match our valid detections
                    if len(anchors) >= len(valid_bboxes):
                        anchors = anchors[valid_indices]

                        # Scale factors back to original image size
                        height, width = image_shape
                        scale_x = width / input_width
                        scale_y = height / input_height

                        for i in range(len(valid_bboxes)):
                            bbox = valid_bboxes[i]
                            score = float(valid_scores[i])
                            lm = valid_landmarks[i]
                            anchor = anchors[i] if i < len(anchors) else [0, 0]

                            # Convert relative bbox to absolute coordinates
                            # SCRFD outputs are typically in (cx, cy, w, h) format relative to anchors
                            cx = (bbox[0] + anchor[0]) * scale_x
                            cy = (bbox[1] + anchor[1]) * scale_y
                            w = bbox[2] * scale_x
                            h = bbox[3] * scale_y

                            # Convert to (x1, y1, x2, y2) format
                            x1 = max(0, int(cx - w / 2))
                            y1 = max(0, int(cy - h / 2))
                            x2 = min(width, int(cx + w / 2))
                            y2 = min(height, int(cy + h / 2))

                            # Validate bbox dimensions
                            if x2 > x1 and y2 > y1:
                                # Process landmarks
                                face_landmarks = []
                                for j in range(0, len(lm), 2):
                                    lm_x = (lm[j] + anchor[0]) * scale_x
                                    lm_y = (lm[j + 1] + anchor[1]) * scale_y
                                    face_landmarks.append((float(lm_x), float(lm_y)))

                                detection = FaceDetection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=score,
                                    landmarks=face_landmarks,
                                )

                                # Apply comprehensive validation
                                if self.validate_face_detection(detection, image_shape):
                                    detections.append(detection)

        # Fallback for simpler model formats
        elif len(outputs) >= 2:
            bboxes = outputs[0]  # Shape: (N, 4)
            scores = outputs[1]  # Shape: (N,)
            landmarks = (
                outputs[2] if len(outputs) > 2 else None
            )  # Shape: (N, 10) for 5 points

            # Filter by confidence
            valid_indices = scores >= self.confidence_threshold

            if np.any(valid_indices):
                valid_bboxes = bboxes[valid_indices]
                valid_scores = scores[valid_indices]
                valid_landmarks = (
                    landmarks[valid_indices] if landmarks is not None else None
                )

                # Scale bboxes back to original image size
                height, width = image_shape
                scale_x = width / self._input_size[0]
                scale_y = height / self._input_size[1]

                for i in range(len(valid_bboxes)):
                    bbox = valid_bboxes[i]
                    score = float(valid_scores[i])

                    # Scale and validate bbox
                    x1 = max(0, int(bbox[0] * scale_x))
                    y1 = max(0, int(bbox[1] * scale_y))
                    x2 = min(width, int(bbox[2] * scale_x))
                    y2 = min(height, int(bbox[3] * scale_y))

                    # Validate bbox dimensions
                    if x2 > x1 and y2 > y1:
                        # Process landmarks if available
                        face_landmarks = None
                        if valid_landmarks is not None:
                            lm = valid_landmarks[i]
                            face_landmarks = []
                            for j in range(0, len(lm), 2):
                                lm_x = lm[j] * scale_x
                                lm_y = lm[j + 1] * scale_y
                                face_landmarks.append((float(lm_x), float(lm_y)))

                        detection = FaceDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=score,
                            landmarks=face_landmarks,
                        )

                        # Apply comprehensive validation
                        if self.validate_face_detection(detection, image_shape):
                            detections.append(detection)

        return detections

    def _generate_anchors(
        self, scale_idx: int, input_height: int, input_width: int
    ) -> np.ndarray:
        """Generate anchor points for SCRFD model at given scale.

        Args:
            scale_idx: Scale index (0, 1, or 2)
            input_height: Input image height
            input_width: Input image width

        Returns:
            Array of anchor points (N, 2) with (x, y) coordinates
        """
        # SCRFD typically uses strides of 8, 16, 32 for the three scales
        strides = [8, 16, 32]
        stride = strides[scale_idx]

        # Calculate feature map size
        feat_h = input_height // stride
        feat_w = input_width // stride

        # SCRFD uses 2 anchors per spatial location (different aspect ratios)
        num_anchors_per_location = 2

        # Generate anchor grid with multiple anchors per location
        anchor_centers = []
        for y in range(feat_h):
            for x in range(feat_w):
                for anchor_idx in range(num_anchors_per_location):
                    anchor_x = (x + 0.5) * stride
                    anchor_y = (y + 0.5) * stride
                    anchor_centers.append([anchor_x, anchor_y])

        return np.array(anchor_centers, dtype=np.float32)

    def _postprocess_yolo_results(
        self, result, image_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """Post-process YOLO model results to extract face detections.

        Args:
            result: YOLO detection result object
            image_shape: Original image shape (height, width)

        Returns:
            List of validated face detections
        """
        detections = []

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes

            # Extract bounding boxes and scores
            if hasattr(boxes, "xyxy") and hasattr(boxes, "conf"):
                bboxes = boxes.xyxy.cpu().numpy()  # (N, 4) in xyxy format
                scores = boxes.conf.cpu().numpy()  # (N,)

                # Filter by confidence
                valid_indices = scores >= self.confidence_threshold

                if np.any(valid_indices):
                    valid_bboxes = bboxes[valid_indices]
                    valid_scores = scores[valid_indices]

                    for i in range(len(valid_bboxes)):
                        bbox = valid_bboxes[i]
                        score = float(valid_scores[i])

                        # Convert to integers and validate
                        x1 = max(0, int(bbox[0]))
                        y1 = max(0, int(bbox[1]))
                        x2 = min(image_shape[1], int(bbox[2]))
                        y2 = min(image_shape[0], int(bbox[3]))

                        # Validate bbox dimensions
                        if x2 > x1 and y2 > y1:
                            detection = FaceDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=score,
                                landmarks=None,  # YOLO face models typically don't include landmarks
                            )

                            # Apply comprehensive validation
                            if self.validate_face_detection(detection, image_shape):
                                detections.append(detection)

        return detections

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for face detections.

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
        }

    def validate_face_detection(
        self, detection: FaceDetection, image_shape: Tuple[int, int]
    ) -> bool:
        """Validate a face detection result.

        Args:
            detection: Face detection to validate
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

        # Check minimum size
        if (x2 - x1) < MINIMUM_FACE_SIZE or (y2 - y1) < MINIMUM_FACE_SIZE:
            return False

        # Check confidence
        if detection.confidence < self.confidence_threshold:
            return False

        return True

    def __del__(self):
        """Cleanup resources when detector is destroyed."""
        self._model = None

    def process_frame_batch(
        self,
        frames: List["FrameData"],
        video_metadata: "VideoMetadata",
        progress_callback: Optional[callable] = None,
        interruption_check: Optional[callable] = None,
    ) -> None:
        """Process a batch of frames with face detection at a high level.

        This method handles:
        - Loading images from file paths
        - Running face detection in batches
        - Adding face detections to FrameData objects in-place
        - Error handling for individual frames
        - Progress tracking with rate calculation
        - Interruption checking

        Args:
            frames: List of FrameData objects to process
            video_metadata: Video metadata for processing context
            progress_callback: Optional callback for progress updates (called with processed_count)
            interruption_check: Optional callback to check for interruption

        Raises:
            FaceDetectionError: If processing fails completely
        """
        if not frames:
            return

        from ..data.detection_results import FaceDetection
        from pathlib import Path

        total_faces_found = 0
        processed_count = 0
        start_time = time.time()

        # Process in batches for memory efficiency
        batch_size = self.config.models.batch_size

        for i in range(0, len(frames), batch_size):
            # Check for interruption at the start of each batch
            if interruption_check:
                interruption_check()
                
            batch_frames = frames[i : i + batch_size]
            batch_images = []
            valid_frames = []

            # Load images for this batch
            for frame in batch_frames:
                # Check for interruption periodically during frame loading
                if interruption_check and len(batch_images) % 10 == 0:
                    interruption_check()
                    
                if frame.file_path.exists():
                    try:
                        image = cv2.imread(str(frame.file_path))
                        if image is not None:
                            batch_images.append(image)
                            valid_frames.append(frame)
                        else:
                            logger.warning(f"Failed to load frame: {frame.file_path}")
                    except Exception as e:
                        logger.warning(f"Error loading frame {frame.file_path}: {e}")
                else:
                    logger.warning(f"Frame file not found: {frame.file_path}")

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

            # Check for interruption before running detection
            if interruption_check:
                interruption_check()

            # Run face detection on batch
            try:
                batch_face_results = self.detect_batch(batch_images)

                # Process results and add face detections to FrameData objects
                for j, (faces, frame) in enumerate(
                    zip(batch_face_results, valid_frames)
                ):
                    # Check for interruption during result processing
                    if interruption_check and j % 5 == 0:
                        interruption_check()
                        
                    if faces:  # Add face detections to the frame
                        # Convert faces to FaceDetection objects if needed
                        face_detections = []
                        for face in faces:
                            if isinstance(face, FaceDetection):
                                face_detections.append(face)
                            else:
                                # Convert from detection result object
                                face_detections.append(
                                    FaceDetection(
                                        bbox=face.bbox,
                                        confidence=face.confidence,
                                        landmarks=face.landmarks,
                                    )
                                )

                        # Add face detections to the FrameData object
                        frame.face_detections.extend(face_detections)
                        total_faces_found += len(faces)

            except Exception as e:
                logger.error(
                    f"Face detection failed for batch {i//batch_size + 1}: {e}"
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

        logger.info(
            f"Face detection completed: {total_faces_found} faces found in "
            f"{processed_count} frames"
        )

    def _create_frame_data_from_info(
        self,
        frame_info: Dict,
        faces: List["FaceDetection"],
        image: Optional[np.ndarray],
        video_metadata: "VideoMetadata",
    ) -> "FrameData":
        """Create FrameData object from frame_info dict and face detections.

        Args:
            frame_info: Frame information dict from frame extraction
            faces: List of face detection results
            image: Optional loaded image array to avoid reloading
            video_metadata: Video metadata for creating FrameData

        Returns:
            FrameData object with face detections populated
        """
        from ..data.frame_data import FrameData, SourceInfo, ImageProperties
        from ..data.detection_results import FaceDetection
        from pathlib import Path

        # Convert faces to FaceDetection objects if needed
        face_detections = []
        for face in faces:
            if isinstance(face, FaceDetection):
                face_detections.append(face)
            else:
                # Convert from detection result object
                face_detections.append(
                    FaceDetection(
                        bbox=face.bbox,
                        confidence=face.confidence,
                        landmarks=face.landmarks,
                    )
                )

        # Get image properties
        frame_path = Path(frame_info["file_path"])
        file_size = frame_path.stat().st_size if frame_path.exists() else 0

        # Get dimensions from provided image or load if needed
        if image is not None:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
        else:
            # Fallback: try to load image to get dimensions
            loaded_image = cv2.imread(str(frame_path))
            if loaded_image is not None:
                height, width = loaded_image.shape[:2]
                channels = loaded_image.shape[2] if len(loaded_image.shape) > 2 else 1
            else:
                # Last resort: use video metadata or reasonable defaults
                width = getattr(video_metadata, "width", 1920)
                height = getattr(video_metadata, "height", 1080)
                channels = 3

        # Create SourceInfo object
        source_info = SourceInfo(
            video_timestamp=frame_info["timestamp"],
            extraction_method=frame_info["extraction_method"],
            original_frame_number=int(frame_info["timestamp"] * video_metadata.fps),
            video_fps=video_metadata.fps,
        )

        # Create ImageProperties object
        image_properties = ImageProperties(
            width=width,
            height=height,
            channels=channels,
            file_size_bytes=file_size,
            format="JPEG",
        )

        # Create and return FrameData object
        return FrameData(
            frame_id=frame_info["frame_id"],
            file_path=frame_path,
            source_info=source_info,
            image_properties=image_properties,
            face_detections=face_detections,
            # Other fields have default factories and will be populated later
        )


def create_face_detector(
    model_name: Optional[str] = None,
    device: str = "auto",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    config: Optional["Config"] = None,
) -> FaceDetector:
    """Factory function to create a FaceDetector instance.

    Args:
        model_name: Name of face detection model (default: use config default)
        device: Computation device preference
        confidence_threshold: Minimum confidence threshold
        config: Application configuration object (optional)

    Returns:
        Configured FaceDetector instance

    Raises:
        FaceDetectionError: If model creation fails
    """
    if model_name is None:
        defaults = ModelConfigs.get_default_models()
        model_name = defaults["face_detection"]

    return FaceDetector(model_name, device, confidence_threshold, config)
