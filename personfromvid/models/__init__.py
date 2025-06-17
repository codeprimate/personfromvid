"""Models package for Person From Vid.

This package contains AI model management, configuration, and inference components.
"""

from .face_detector import FaceDetector, create_face_detector
from .head_pose_estimator import HeadPoseEstimator, create_head_pose_estimator
from .model_configs import (
    ModelConfigs,
    ModelFile,
    ModelFormat,
    ModelMetadata,
    ModelProvider,
    get_model_for_config_key,
    validate_config_models,
)
from .model_manager import ModelManager, get_model_manager
from .pose_estimator import COCO_KEYPOINT_NAMES, PoseEstimator, create_pose_estimator

__all__ = [
    # Model configuration
    "ModelConfigs",
    "ModelMetadata",
    "ModelFile",
    "ModelFormat",
    "ModelProvider",
    "get_model_for_config_key",
    "validate_config_models",
    # Model management
    "ModelManager",
    "get_model_manager",
    # Face detection
    "FaceDetector",
    "create_face_detector",
    # Pose estimation
    "PoseEstimator",
    "create_pose_estimator",
    "COCO_KEYPOINT_NAMES",
    # Head pose estimation
    "HeadPoseEstimator",
    "create_head_pose_estimator",
]
