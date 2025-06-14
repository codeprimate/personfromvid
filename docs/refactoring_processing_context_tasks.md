Reference: @docs/refactoring_processing_context.md

# Refactoring Processing Context Development Plan

This document outlines the step-by-step plan to refactor the application based on the proposal in `docs/refactoring_processing_context.md`. The refactoring is divided into phases to ensure a systematic and manageable process.

## Phase 1: Introduce the Unified `ProcessingContext`

This phase focuses on creating a centralized `ProcessingContext` to simplify class initializations and data flow. This will reduce parameter passing and decouple components from the main application orchestrator.

### Task 1.1: Define the `ProcessingContext` Data Class
1.  **Create a new file** at `personfromvid/data/context.py`.
2.  **Define the `ProcessingContext` class** in this file. It should be a `dataclass` with `frozen=True`.
3.  **Add attributes** to the class. Based on the proposal, it should include:
4. 
    ```python
    from dataclasses import dataclass
    from pathlib import Path
    from .config import Config
    from ..core.temp_manager import TempManager

    @dataclass(frozen=True)
    class ProcessingContext:
        video_path: Path
        video_base_name: str
        config: Config
        temp_manager: TempManager
        output_directory: Path
    ```
4.  **Add the new module** to the `__init__.py` in `personfromvid/data/` to make it easily importable.

### Task 1.2: Instantiate `ProcessingContext` in the Main Application Entry Point
1.  **Locate the entry point:** The main entry point is the `main` function in `personfromvid/cli.py`.
2.  **Modify the logic** within the `main` function to create the `ProcessingContext` instance. This should happen after the `Config` object is loaded and command-line arguments are parsed.
3.  The `ProcessingContext` will now handle the lifecycle of `TempManager` internally.
4.  The `ProcessingContext` object will then be passed down to the main processing pipeline/orchestrator.

### Task 1.3: Refactor Core Components to Use `ProcessingContext`
This is an iterative task. For each of the components listed below:

1.  Change the `__init__` method signature to accept `context: ProcessingContext` instead of multiple individual arguments.
2.  Update the internal logic of the class to access contextual data from the `context` object (e.g., `context.output_directory`).
3.  Update the place where the component is instantiated to pass the single `context` object.
4.  Run unit and integration tests related to the component to ensure functionality is preserved.

**Initial list of target classes for refactoring:**

-   `personfromvid.core.state_manager.StateManager`
-   `personfromvid.output.image_writer.ImageWriter` (as mentioned in the proposal)
-   `personfromvid.output.naming_convention.NamingConvention` (as mentioned in the proposal)
-   Other classes that are initialized with `video_path`, `output_directory`, `config` etc. A codebase search (`grep`) for these parameters in `__init__` methods is recommended to find all candidates.

## Phase 2: Standardize on `FrameData` as the Unit of Work

This phase aims to make `FrameData` the standard data carrier for all frame-level analysis, improving encapsulation and API consistency.

### Task 2.1: Refactor Pose Classification Logic
1.  **Target file:** `personfromvid/analysis/pose_classifier.py`.
2.  **Refactor `classify_pose`:** Change the signature to accept a single `FrameData` object, as suggested in the proposal. The new signature could be `classify_poses_in_frame(self, frame: FrameData)`.
3.  **Update implementation:** The method should now internally access `frame.pose_detections` and other required attributes from the `frame` object.
4.  **Update call sites:** Find all usages of `classify_pose` and update them to pass the `FrameData` instance.

### Task 2.2: Refactor Closeup Detection Logic
1.  **Target file:** `personfromvid/analysis/closeup_detector.py`.
2.  **Identify and refactor `is_closeup`:** Locate the `is_closeup` method (or equivalent).
3.  **Change signature:** Refactor its signature to operate on data models as suggested, for example: `is_closeup(self, pose: PoseDetection, image_properties: ImageProperties)`.
4.  **Update implementation and call sites** accordingly.

### Task 2.3: General Review of the `analysis` Module
1.  **Systematically review** all public functions/methods in the `personfromvid/analysis/` directory.
2.  **Identify** any other methods that process frame-related data using primitive types instead of `FrameData` or other established data models.
3.  **Create sub-tasks** to refactor these methods to align with the new standardized approach.

## Phase 3: Verification and Cleanup
1.  **Run full test suite:** Execute all unit and integration tests to ensure no regressions were introduced.
2.  **Perform end-to-end testing:** Run the application with a sample video to validate the entire pipeline.
3.  **Code cleanup:** Remove any unused imports, variables, or helper functions that have become redundant after the refactoring.
4.  **Update documentation:** Update `docs/architecture.md` and any other relevant developer documentation to reflect the new, simplified data flow and use of `ProcessingContext`.

## Progress

### Phase 1: Introduce the Unified `ProcessingContext`
- [x] **Task 1.1: Define the `ProcessingContext` Data Class** ✅ **COMPLETED**
  - [x] Create `personfromvid/data/context.py`.
  - [x] Define `ProcessingContext` `dataclass`.
  - [x] Add attributes to `ProcessingContext`.
  - [x] Add `context` module to `personfromvid/data/__init__.py`.
- [x] **Task 1.2: Instantiate `ProcessingContext` in the Main Application Entry Point** ✅ **COMPLETED**
  - [x] Locate entry point in `personfromvid/cli.py`.
  - [x] Create `ProcessingContext` instance in `main`.
  - [x] `ProcessingContext` now handles `TempManager` internally.
  - [x] Pass `ProcessingContext` to the processing pipeline.
- [x] **Task 1.3: Refactor Core Components to Use `ProcessingContext`** ✅ **COMPLETED**
  - [x] Refactor `personfromvid.core.state_manager.StateManager`.
  - [x] Refactor `personfromvid.output.image_writer.ImageWriter`.
  - [x] Refactor `personfromvid.output.naming_convention.NamingConvention`.
  - [x] Updated `personfromvid.core.pipeline.ProcessingPipeline` to use context.
  - [x] Updated `personfromvid.core.steps.output_generation.OutputGenerationStep` to use context.
- [x] **Task 1.4: Remove Legacy Compatibility** ✅ **COMPLETED**
  - [x] Remove all legacy parameter support from component constructors.
  - [x] Update all components to require `ProcessingContext` only.
  - [x] Clean up imports and remove unused Optional types.
  - [x] Update tests to use new ProcessingContext-only API.
  - [x] Verify legacy compatibility is completely removed.

### Phase 2: Standardize on `FrameData` as the Unit of Work
- [x] **Task 2.1: Refactor Pose Classification Logic** ✅ **COMPLETED**
  - [x] Target `personfromvid/analysis/pose_classifier.py`.
  - [x] Refactor `classify_pose` to `_classify_single_pose` (private helper method).
  - [x] Add new `classify_poses_in_frame` method to accept a `FrameData` object.
  - [x] Update implementation to use `FrameData` and modify pose detections in place.
  - [x] Update call sites in `personfromvid/models/pose_estimator.py`.
  - [x] Update unit tests to use new `classify_poses_in_frame` API.
  - [x] Update docstring examples to reflect new API.
- [x] **Task 2.2: Refactor Closeup Detection Logic** ✅ **COMPLETED**
  - [x] Target `personfromvid/analysis/closeup_detector.py`.
  - [x] Refactor API to operate on `FrameData` objects instead of primitive types.
  - [x] Add new `detect_closeups_in_frame` method to accept a `FrameData` object.
  - [x] Update implementation to modify `FrameData.closeup_detections` in place.
  - [x] Remove legacy methods that accepted primitive types (no backward compatibility).
  - [x] Update unit tests to use new `FrameData`-based API.
  - [x] Update docstring examples to reflect new API.
  - [x] Verify integration with existing pipeline steps.
- [x] **Task 2.3: General Review of the `analysis` Module** ✅ **COMPLETED**
  - [x] Systematically review all public functions/methods in `personfromvid/analysis/`.
  - [x] Identify methods that process frame-related data using primitive types.
  - [x] **Task 2.3.1: Refactor Head Angle Classifier Logic** ✅ **COMPLETED**
    - [x] Target `personfromvid/analysis/head_angle_classifier.py`.
    - [x] Update `HeadPoseResult` data model to make `direction` optional and add `direction_confidence`.
    - [x] Refactor `HeadAngleClassifier` to have single public method `classify_head_poses_in_frame`.
    - [x] Update call sites in `personfromvid/models/head_pose_estimator.py`.
    - [x] Refactor unit tests to use new `FrameData`-based API.
  - [x] **Task 2.3.2: Refactor Quality Assessor Logic** ✅ **COMPLETED**
    - [x] Target `personfromvid/analysis/quality_assessor.py`.
    - [x] Replace `assess_quality`, `assess_frame_batch`, and `process_frame_batch` with single `assess_quality_in_frame` method.
    - [x] Update call sites in `personfromvid/core/steps/quality_assessment.py`.
    - [x] Refactor unit tests to use new `FrameData`-based API.
    - [x] Remove batch processing logic from `QualityAssessor` class.
  - [x] **Task 2.3.3: Review Frame Selector** ✅ **COMPLETED**
    - [x] Confirmed `personfromvid/analysis/frame_selector.py` already uses `FrameData` correctly.
    - [x] No changes needed - already aligned with refactoring goals.

### Phase 3: Verification and Cleanup
- [ ] Run full test suite.
- [ ] Perform end-to-end testing.
- [ ] Code cleanup.
- [ ] Update documentation (`docs/architecture.md`).