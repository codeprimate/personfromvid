# Refactoring Plan: Naive Multi-Person Support

## 1. Overview

This document outlines a pragmatic, iterative strategy to handle multiple people in a video. The goal is to increase the number of selected output frames based on the number of people detected, without implementing a complex person-tracking system.

Instead of tracking individuals across frames, we will treat each person detection within a single frame as a separate, independent candidate for selection. This allows us to reuse the existing `FrameSelector` logic with minimal changes.

## 2. Core Strategy: "Detection-as-Candidate"

The core of this approach is to shift the selection process from "choosing the best frames" to "choosing the best *detections* within frames".

To achieve this without rewriting `FrameSelector`, we will introduce a lightweight wrapper or "view" around our existing `FrameData`. This view will represent a single detection (e.g., one person's pose) and will conform to the interface that `FrameSelector` already expects.

### Key Concepts

*   **Detection as a Candidate**: If a frame contains three people, we will generate three independent candidates for selection.
*   **`FrameDataView` Wrapper**: A new class that wraps a `FrameData` object and "focuses" on a single detection within it. It will proxy method calls, returning information only for its specific detection.

## 3. Proposed Changes by Module

### 3.1. Data Structures (`personfromvid/data/`)

The primary change is the introduction of a new view class. The existing `FrameData` and detection dataclasses can remain largely unchanged.

*   **New `FrameDataView` class**:
    *   This will be a new dataclass designed to mimic `FrameData`.
    *   **Attributes**:
        *   `original_frame: FrameData`: A reference to the complete `FrameData` object.
        *   `detection_type: str`: The type of detection being focused on (e.g., "pose" or "face").
        *   `detection_index: int`: The index of the specific detection within the corresponding list (e.g., `original_frame.pose_detections[detection_index]`).
    *   **Method Overrides**: The view must override key methods that `FrameSelector` relies on. When called, these methods will return data only for the *focused detection*.
        *   `get_pose_classifications()`: Returns classifications only for the single pose detection at `detection_index`.
        *   `get_head_directions()`: Returns directions only for the single face detection at `detection_index`.
        *   `is_closeup_shot()`: Returns closeup info relevant to the focused detection.
        *   It will also need proxy properties for `quality_metrics`, `source_info`, etc., that point back to the original frame.

### 3.2. Pipeline Orchestration

A new pre-processing step is required before frame selection.

1.  **"Explode" Frames into Views**:
    *   Before calling `FrameSelector`, the main pipeline will iterate through the list of `FrameData` objects.
    *   For each `FrameData` object, it will inspect its detection lists (`pose_detections`, `face_detections`).
    *   For each detection found, it will create a `FrameDataView` instance.
    *   **Example**: If `frame_10` has 2 pose detections, it will generate two `FrameDataView` objects: one focused on `pose_detections[0]` and another on `pose_detections[1]`.
    *   The result is a new, longer list of candidates (`List[FrameDataView]`) that will be passed to the `FrameSelector`.

### 3.3. Frame Selection (`personfromvid/analysis/frame_selector.py`)

This approach requires minimal changes to `FrameSelector`, which is its primary advantage.

*   **Input Type**: The `select_best_frames` method will now expect a list of `FrameDataView` objects instead of `FrameData` objects.
*   **Internal Logic**: The grouping, ranking, and diversity logic will function as-is because it operates on the `FrameDataView`s, which provide the expected interface.
*   **Scoring Functions**: The scoring functions (`_calculate_pose_frame_score`, `_calculate_head_angle_frame_score`) will now naturally score a single detection's attributes (e.g., a specific pose's confidence) rather than an ambiguous "best" pose in the frame. This is a significant improvement in clarity.
*   **Diversity**: The diversity check (`_is_diverse_enough`) will compare the *source timestamps* of the views. This means two different people in the *same frame* will be correctly identified as non-diverse (timestamp diff is 0), and only the higher-scoring one will be selected for a given category. This is desirable behavior to avoid redundant frame output.

### 3.4. Output Generation

The output stage will receive a list of selected `FrameDataView` objects and must be adapted to handle them.

*   When saving a file, the logic must access the original frame's image data via `selected_view.original_frame`.
*   The filename can be constructed as before, perhaps with a suffix indicating the detection index (e.g., `frame_10_person_1.jpg`).
*   **Optional Enhancement**: For clarity, the code can use the `detection_index` to retrieve the bounding box of the selected person and draw it onto the saved output image.

## 4. High-Level Implementation Steps

1.  **Define `FrameDataView`**: Create the new `FrameDataView` dataclass in a relevant module (e.g., `personfromvid/data/frame_data.py`). Implement its attributes and method overrides.
2.  **Implement "Explode Frames" Logic**: Create a new function that takes `List[FrameData]` and returns `List[FrameDataView]`.
3.  **Update Pipeline**: In the main processing script, insert the "Explode Frames" step before the call to `FrameSelector.select_best_frames`.
4.  **Adapt `FrameSelector` Input**: Change the type hint in `select_best_frames` to accept the new view objects. Review scoring functions to ensure they interact correctly with the view's properties.
5.  **Update Output Logic**: Modify the code that saves image files to correctly handle `FrameDataView` objects, accessing the original frame data and potentially drawing bounding boxes.