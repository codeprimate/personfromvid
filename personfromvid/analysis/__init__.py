"""Analysis package for pose and head angle classification.

This package contains classifiers for analyzing pose data and head angles
from AI model outputs, plus quality assessment for frame selection.
"""

from .pose_classifier import PoseClassifier
from .head_angle_classifier import HeadAngleClassifier
from .quality_assessor import QualityAssessor, create_quality_assessor
from .closeup_detector import CloseupDetector
from .frame_selector import FrameSelector, create_frame_selector, SelectionCriteria

__all__ = [
    "PoseClassifier",
    "HeadAngleClassifier",
    "QualityAssessor",
    "create_quality_assessor",
    "CloseupDetector",
    "FrameSelector",
    "create_frame_selector",
    "SelectionCriteria",
]
