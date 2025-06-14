After a critical review of the class initializations and data flow described in `docs/architecture.md`, here is an analysis of opportunities to simplify inter-class communication and data representation.

The architecture already employs strong data modeling patterns, particularly with the `PipelineState` and `FrameData` classes. These act as central aggregators of state and data, which is excellent. However, this principle can be applied more broadly to simplify initialization and data flow in several key areas.

### 1. Introduce a Unified `ProcessingContext` Object

**Observation:**
Many classes across different modules are initialized with individual, yet related, pieces of contextual information like paths and configuration objects. This data often originates from the same source (initial CLI arguments and video file) but is passed down through multiple layers as separate arguments.

For example:
- `TempManager` is initialized with `video_path`.
- `StateManager` is initialized with `video_path` and a `TempManager` instance.
- `ImageWriter` and `NamingConvention` are initialized with `video_base_name` and `output_directory`.

This approach creates verbose initializers and couples classes to the specific way their parent orchestrator manages these pieces of data.

**Suggestion:**
Introduce a single, immutable `ProcessingContext` data class. This object would be instantiated once when the pipeline begins and passed to any class that requires this shared context.

**Key characteristics:**
- It is a `dataclass` with `frozen=True` to prevent mutation.
- It contains all shared data: `video_path`, `config`, `output_directory`.
- It handles the lifecycle of shared services like `TempManager` internally.

```python
@dataclass(frozen=True)
class ProcessingContext:
    video_path: Path
    config: Config
    output_directory: Path
    
    # TempManager is created internally
    temp_manager: TempManager = field(init=False)

    def __post_init__(self):
        # ... validation logic ...
        object.__setattr__(self, 'temp_manager', TempManager(self.video_path))
```

The consuming classes would then have a much simpler and more consistent interface:
- `StateManager(context: ProcessingContext)`
- `ImageWriter(context: ProcessingContext, image_config: OutputImageConfig)`
- `NamingConvention(context: ProcessingContext)`

This approach provides several benefits:
- **Simplicity:** Constructors are cleaner and more predictable.
- **Decoupling:** Components no longer need to know where contextual data comes from; they just need the `context` object.
- **Maintainability:** If a new piece of shared context is needed (e.g., a global logger instance), it can be added to `ProcessingContext` without changing the signature of every class that might need it.
- **Centralized Initialization:** The logic for creating shared services like `TempManager` is centralized within `ProcessingContext`, ensuring it's done consistently.

### 2. Standardize on `FrameData` as the Unit of Work for Analysis

**Observation:**
The `FrameData` class is an excellent aggregator for all information related to a single frame. However, some analysis methods still accept primitives or partial data structures, forcing the calling code to deconstruct the `FrameData` object.

A key example is in `analysis/pose_classifier.py`:
```python
def classify_pose(pose_detection: PoseDetection, image_shape: Tuple[int, int]) -> List[Tuple[str, float]]
```
Here, the caller must extract a `PoseDetection` object and the `image_shape` from a `FrameData` instance to pass them to the classifier.

**Suggestion:**
Refactor analysis methods to accept the `FrameData` object directly. The method can then access the data it needs internally.

```python
# In analysis/pose_classifier.py
def classify_poses_in_frame(self, frame: FrameData) -> List[PoseClassificationResult]:
    # Internally, it would iterate over frame.pose_detections
    # and use frame.image_properties to get the shape.
    # It could then update the PoseDetection objects within the FrameData instance directly
    # or return new classification result objects.
```

The method `is_closeup` is another example. Instead of:
```python
def is_closeup(keypoints: Dict, bbox: Tuple, image_shape: Tuple) -> bool
```
It could be simplified to operate on the existing data models:
```python
def is_closeup(self, pose: PoseDetection, image_properties: ImageProperties) -> bool
```

**Benefits:**
- **Encapsulation:** The internal logic of how `FrameData` is structured is hidden from the analysis methods' signatures.
- **Consistency:** Promotes a uniform pattern where any operation on a frame's data is passed the entire `FrameData` object.
- **Future-Proofing:** If a classification algorithm is improved to consider another metric (e.g., `quality_metrics` from the `FrameData` object), the method signature does not need to change.

### 3. Consolidate Redundant Configuration in Output Classes

**Observation:**
The `ImageWriter` and `NamingConvention` classes are both initialized with `output_directory` and `video_base_name`. This is a specific instance of the issue described in the first point but is particularly noticeable in the `output` module, where both classes are closely related.

```python
# output/image_writer.py
def __init__(self, config: OutputImageConfig, output_directory: Path, video_base_name: str)

# output/naming_convention.py
def __init__(self, video_base_name: str, output_directory: Path)
```

**Suggestion:**
This can be solved by the proposed `ProcessingContext`. Alternatively, a more localized `OutputContext` could be created by the `OutputGenerationStep` to group all data needed for generating files.

```python
# In output/naming_convention.py or a shared location
@dataclass(frozen=True)
class OutputContext:
    video_base_name: str
    output_directory: Path
    # Could also include output format, quality, etc.
```
Both `ImageWriter` and `NamingConvention` would then be initialized with this single object.

**Benefits:**
- **Reduces Redundancy:** Eliminates duplicate parameters.
- **Improves Cohesion:** Groups all output-related contextual data into a single, dedicated structure, making the concerns of the output stage clearer.

### Summary

The existing architecture is robust. By extending the current data aggregation patterns through the introduction of a `ProcessingContext` object and ensuring `FrameData` is used consistently as the primary data carrier, the design can be made even cleaner and more maintainable. These changes would lead to simpler class initializations, reduced coupling, and a more consistent and intuitive data flow throughout the pipeline.