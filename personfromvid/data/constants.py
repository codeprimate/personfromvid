"""Constants and definitions for the pipeline.

This module centralizes important constants used throughout the pipeline,
including the definition of processing steps and their order.
"""

from typing import List, Tuple


# Pipeline step definitions
PIPELINE_STEPS = {
    "initialization": "Initialize processing environment",
    "frame_extraction": "Extract frames from video",
    "face_detection": "Detect faces in frames",
    "pose_analysis": "Analyze body poses and head angles",
    "closeup_detection": "Detect closeup shots and shot types",
    "quality_assessment": "Assess frame quality",
    "frame_selection": "Select best frames",
    "output_generation": "Generate output files (images, JSON, etc.)",
}

PIPELINE_STEP_NAMES = list(PIPELINE_STEPS.keys())
TOTAL_PIPELINE_STEPS = len(PIPELINE_STEPS)

# State data keys
ALL_SELECTED_FRAMES_KEY = "all_selected_frames"


def get_pipeline_steps() -> List[Tuple[str, str]]:
    """Get a list of all pipeline steps and their descriptions."""
    return [(step, description) for step, description in PIPELINE_STEPS.items()]


def get_pipeline_step_names() -> List[str]:
    """Get the list of pipeline step names.

    Returns:
        List of step names in order
    """
    return PIPELINE_STEP_NAMES.copy()


def get_total_pipeline_steps() -> int:
    """Get the total number of pipeline steps.

    Returns:
        Number of processing steps in the pipeline
    """
    return TOTAL_PIPELINE_STEPS


def get_step_index(step_name: str) -> int:
    """Get the index of a pipeline step.

    Args:
        step_name: Name of the step

    Returns:
        Zero-based index of the step

    Raises:
        ValueError: If step name is not found
    """
    try:
        return PIPELINE_STEP_NAMES.index(step_name)
    except ValueError:
        raise ValueError(f"Unknown pipeline step: {step_name}")


def get_step_description(step_name: str) -> str:
    """Get the description for a pipeline step.

    Args:
        step_name: Name of the step

    Returns:
        Description of the step

    Raises:
        ValueError: If step name is not found
    """
    try:
        return PIPELINE_STEPS[step_name]
    except KeyError:
        raise ValueError(f"Unknown pipeline step: {step_name}")


def is_valid_step(step_name: str) -> bool:
    """Check if a step name is valid.

    Args:
        step_name: Name of the step to check

    Returns:
        True if step name is valid, False otherwise
    """
    return step_name in PIPELINE_STEP_NAMES
