# **Enhanced Frame Selection Scoring: A Plan for Transparency and Debuggability**

**Author:** Gemini AI
**Date:** 2025-06-15
**Status:** Approved

---

## 1. Overview & Motivation

### 1.1. The Problem: An Opaque Selection Process

An initial analysis of the frame selection process revealed a significant limitation: **the selection algorithm operates as a "black box."** While the system calculates sophisticated composite scores to rank frames, this crucial information is discarded immediately after use. The final output lacks the necessary data to answer critical questions, such as "Why was this high-quality frame not selected?" or "What was the final selection score that differentiated two competing frames?" This opacity makes debugging a matter of guesswork.

### 1.2. The Goal: Full Transparency and Logical Separation of Concerns

This document outlines a plan to refactor the pipeline to achieve two primary goals:
1.  **Persist all relevant scoring and ranking data** to make every selection decision transparent and auditable.
2.  **Enforce a clean separation of concerns** between the `QualityAssessmentStep` and `FrameSelectionStep`, as per the clarified system intent.

**Clarified System Intent:**
-   **Quality Assessment**: Must quantify quality, score each frame, and **rank all frames by that score.**
-   **Frame Selection**: Must **use the pre-calculated quality scores and ranks** as inputs to its own holistic selection logic.

By implementing these changes, we will create a more logical, maintainable, and debuggable system.

---

## 2. Critical Analysis of the Current System

A deep-dive into the existing codebase identified three primary architectural shortcomings:

-   **The Lost Information Problem**: The system calculates composite selection scores but immediately discards them, losing the "why" behind each decision.
-   **Debugging and Transparency Issues**: The lack of stored scores makes answering fundamental debugging questions impossible.
-   **Inconsistent Data Model**: Fields like `quality_rank` are defined but never populated, creating technical debt.

---

## 3. Proposed Enhancements

### 3.1. Core Idea: Store All Computed Data and Respect Step Boundaries

The central principle is to **store all computed scores and ranks** and to ensure they are computed in the correct pipeline step. To avoid ambiguity, we must distinguish between two types of ranking:

-   **Global Quality Ranking**: A single, global `quality_rank` is assigned to every frame based on its `overall_quality` score. This is a pure assessment of pixel quality and is the responsibility of the **`QualityAssessmentStep`**.

-   **Categorical Selection Ranking**: The `selection_rank` is a local rank (e.g., 1st, 2nd, 3rd) assigned *only to the few frames that are ultimately chosen* for a specific output category (like "closeup" or "front-facing"). This rank is a result of a complex selection process involving quality, diversity, and other factors. It is therefore the responsibility of the **`FrameSelectionStep`**.

The `selection_score` (the composite score used for this selection process) is also calculated and stored in the `FrameSelectionStep`.

### 3.2. Enhanced Data Model

The `SelectionInfo` dataclass in `personfromvid/data/frame_data.py` will be extended to capture this new, rich data.

**Proposed `SelectionInfo` Data Model:**

```python
@dataclass
class SelectionInfo:
    """Information about frame selection for output."""
    
    # --- Existing fields ---
    selected_for_poses: List[str] = field(default_factory=list)
    selected_for_head_angles: List[str] = field(default_factory=list)
    final_output: bool = False
    output_files: List[str] = field(default_factory=list)
    crop_regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    
    # --- Enhanced ranking fields ---
    selection_rank: Optional[int] = None      # Local rank (1-3) for frames *chosen* for a category
    quality_rank: Optional[int] = None        # Global rank among ALL usable candidates (Set by QualityAssessmentStep)
    
    # --- NEW: Selection scoring fields (Set by FrameSelectionStep) ---
    pose_selection_score: Optional[float] = None      # Composite score used for pose categories
    head_angle_selection_score: Optional[float] = None # Composite score used for head angle categories
    final_selection_score: Optional[float] = None     # The score that determined the frame's final selection
    
    # --- NEW: Score breakdown for full transparency ---
    selection_score_breakdown: Dict[str, float] = field(default_factory=dict)  # e.g., {"quality": 0.7, "pose_confidence": 0.2, "face_size": 0.1}
    
    # --- NEW: Rejection reason for non-selected frames ---
    rejection_reason: Optional[str] = None  # e.g., "below_quality_threshold", "insufficient_diversity", "not_top_ranked"
    
    # --- NEW: Category-specific ranks (before diversity filtering) ---
    category_ranks: Dict[str, int] = field(default_factory=dict) # e.g., {"pose_closeup": 1, "head_angle_front": 3}
```

---

## 4. Detailed Implementation Plan

### **Phase 1: Data Model & Serialization**
-   **Task:** Update the `SelectionInfo` dataclass in `personfromvid/data/frame_data.py`.
-   **Task:** Update the `to_dict` and `from_dict` methods in `FrameData` to handle all new `SelectionInfo` fields, ensuring backward compatibility.

### **Phase 2: Enhance `QualityAssessmentStep` (NEW)**
-   **Location:** `personfromvid/core/steps/quality_assessment.py`
-   **Task:** At the end of the `execute` method, after all frames have been assessed, add a call to a new private method: `_rank_frames_by_quality(self, frames)`.
-   **Task:** Implement `_rank_frames_by_quality`. This method will sort the list of assessed frames by `quality_metrics.overall_quality` and iterate through the sorted list to populate the `frame.selections.quality_rank` for each frame.

### **Phase 3: Enhance `FrameSelectionStep`**
-   **Location:** `personfromvid/analysis/frame_selector.py`
-   **Task:** The `FrameSelector` will now *consume* the global `quality_rank`, not create it. Its responsibility is to generate the local `selection_rank` for the items it chooses.
-   **Task:** Modify the scoring functions (`_calculate_pose_frame_score`, etc.) to optionally return a breakdown dictionary.
-   **Task:** In `_select_for_category`, store the composite `selection_score` and its `selection_score_breakdown` on the `frame.selections` object for every candidate.
-   **Task:** Store the pre-diversity-filter rank in `frame.selections.category_ranks`.
-   **Task:** Modify `_select_diverse_frames` to return rejection reasons for frames filtered out due to a lack of diversity. Store this in `rejection_reason`.
-   **Task:** Add a final step in `select_best_frames` to assign rejection reasons to any other unselected frames (e.g., "not_top_contender").

### **Phase 4: Testing Strategy**
-   **Unit Tests for `QualityAssessmentStep`**:
    -   `test_quality_ranking_is_correct`: Ensure `quality_rank` is correctly populated on a list of frames with varying quality scores.
-   **Unit Tests for `FrameSelectionStep`**:
    -   `test_selection_scores_are_stored`: Verify `final_selection_score` and `selection_score_breakdown` are populated.
    -   `test_rejection_reasons_are_assigned`: Check reasons for diversity and other rejections.
-   **Integration Tests**:
    -   `test_end_to_end_selection_transparency`: A full run verifying all new fields from both steps are populated correctly.

### **Phase 5: Validation & Success Metrics**
The success of this refactor is defined by the ability to perform deep analysis on the process. The following `jq` queries are exemplars of the new analytical capabilities:

-   **Quality vs. Selection Correlation**:
    ```bash
    jq '[.frames[] | select(.selections.quality_rank != null)] | group_by(.selections.final_output) | map({selected: .[0].selections.final_output, avg_quality_rank: (map(.selections.quality_rank) | add / length)})' tmp/test2_info.json
    ```
-   **Rejection Reason Distribution**:
    ```bash
    jq '[.frames[] | .selections.rejection_reason] | group_by(.) | map({reason: .[0], count: length})' tmp/test2_info.json
    ```
-   **Inspect a single frame's selection logic**:
    ```bash
    jq '.frames.frame_000360.selections' tmp/test2_info.json
    ```

This improved plan creates a more robust, logical, and maintainable pipeline, paying immediate dividends in debuggability and analytical insight. 