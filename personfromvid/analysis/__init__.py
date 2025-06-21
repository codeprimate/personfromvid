"""Analysis package for pose and head angle classification.

This package contains classifiers for analyzing pose data and head angles
from AI model outputs, plus quality assessment for frame selection.
"""

from .closeup_detector import CloseupDetector
from .frame_selector import FrameSelector, SelectionCriteria, create_frame_selector
from .head_angle_classifier import HeadAngleClassifier
from .person_builder import PersonBuilder
from .person_selector import PersonSelector
from .pose_classifier import PoseClassifier
from .quality_assessor import QualityAssessor, create_quality_assessor

__all__ = [
    "PoseClassifier",
    "HeadAngleClassifier",
    "PersonBuilder",
    "PersonSelector",
    "QualityAssessor",
    "create_quality_assessor",
    "CloseupDetector",
    "FrameSelector",
    "create_frame_selector",
    "SelectionCriteria",
]
