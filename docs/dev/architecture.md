# Person From Vid - Architecture Documentation

## Overview

This document provides a detailed technical overview of the Person From Vid architecture, including the processing pipeline, core domain objects, and project structure.

## Project Structure

```
personfromvid/
├── docs/                             # Documentation
│   └── dev/                          # Development documentation
│       ├── architecture.md           # This file - system architecture
│       ├── code_quality.md           # Code quality guidelines
│       └── publishing.md             # PyPI publishing guide
├── personfromvid/                    # Main package directory
│   ├── __init__.py                   # Package initialization
│   ├── __main__.py                   # Main executable entry point
│   ├── cli.py                        # Command-line interface entry point
│   ├── analysis/                     # Image analysis modules
│   │   ├── __init__.py
│   │   ├── closeup_detector.py       # Close-up shot identification
│   │   ├── frame_selector.py         # Best frame selection logic
│   │   ├── head_angle_classifier.py  # Head angle classification
│   │   ├── pose_classifier.py        # Body pose classification logic
│   │   └── quality_assessor.py       # Image quality evaluation
│   ├── core/                         # Core processing modules
│   │   ├── steps/                    # Modular pipeline processing steps
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Base class for a pipeline step
│   │   │   ├── closeup_detection.py  # Close-up detection step
│   │   │   ├── face_detection.py     # Face detection step
│   │   │   ├── frame_extraction.py   # Frame extraction step
│   │   │   ├── frame_selection.py    # Frame selection step
│   │   │   ├── initialization.py     # Pipeline initialization step
│   │   │   ├── output_generation.py  # Output generation step
│   │   │   ├── pose_analysis.py      # Pose analysis step
│   │   │   └── quality_assessment.py # Quality assessment step
│   │   ├── __init__.py
│   │   ├── frame_extractor.py        # Keyframe extraction logic
│   │   ├── pipeline.py               # Main processing pipeline orchestrator
│   │   ├── state_manager.py          # Pipeline state persistence and resumption
│   │   ├── temp_manager.py           # Temporary directory management
│   │   └── video_processor.py        # Video analysis and frame extraction
│   ├── data/                         # Data models and structures
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration management
│   │   ├── constants.py              # Project-wide constants
│   │   ├── context.py                # Processing context data models
│   │   ├── detection_results.py      # AI model output structures
│   │   ├── frame_data.py             # Frame metadata structures
│   │   └── pipeline_state.py         # Pipeline state data models
│   ├── models/                       # AI model management
│   │   ├── __init__.py
│   │   ├── face_detector.py          # Face detection inference
│   │   ├── head_pose_estimator.py    # Head orientation analysis
│   │   ├── model_configs.py          # Model configuration and metadata
│   │   ├── model_manager.py          # Model downloading and caching
│   │   ├── model_utils.py            # Utilities for model operations
│   │   └── pose_estimator.py         # Body pose estimation
│   ├── output/                       # Output generation
│   │   ├── __init__.py
│   │   ├── image_writer.py           # JPEG encoding and file generation
│   │   └── naming_convention.py      # File naming logic
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── exceptions.py             # Custom exception classes
│       ├── formatting.py             # Text and data formatting utilities
│       ├── logging.py                # Logging configuration
│       ├── output_formatter.py       # Console output formatting
│       ├── progress.py               # Rich progress display management
│       └── validation.py             # Input validation utilities
├── scripts/                          # Development and utility scripts
│   └── clean.py                      # Cleanup utility script
├── tests/                            # Test suite
│   ├── fixtures/                     # Test data and mock files
│   │   └── test_video.mp4            # Sample video for testing
│   ├── integration/                  # Integration tests
│   │   ├── __init__.py
│   │   └── test_model_integration.py # Model integration tests
│   ├── unit/                         # Unit tests
│   │   ├── __init__.py
│   │   ├── test_closeup_detector.py  # Closeup detection tests
│   │   ├── test_config.py            # Configuration tests
│   │   ├── test_data_models.py       # Data model tests
│   │   ├── test_face_detector.py     # Face detection tests
│   │   ├── test_frame_extractor.py   # Frame extraction tests
│   │   ├── test_frame_selector.py    # Frame selection tests
│   │   ├── test_head_angle_classifier.py  # Head angle classification tests
│   │   ├── test_head_pose_estimator.py    # Head pose estimation tests
│   │   ├── test_model_management.py  # Model management tests
│   │   ├── test_model_utils.py       # Model utility tests
│   │   ├── test_output.py            # Output generation tests
│   │   ├── test_pipeline.py          # Pipeline tests
│   │   ├── test_pose_classifier.py   # Pose classification tests
│   │   ├── test_pose_estimator.py    # Pose estimation tests
│   │   ├── test_processing_context.py # Processing context tests
│   │   ├── test_progress.py          # Progress display tests
│   │   ├── test_quality_assessment_step.py  # Quality assessment step tests
│   │   ├── test_quality_assessor.py  # Quality assessment tests
│   │   ├── test_state_manager.py     # State management tests
│   │   ├── test_temp_manager.py      # Temporary directory management tests
│   │   ├── test_validation.py        # Validation tests
│   │   └── test_video_processor.py   # Video processing tests
│   ├── __init__.py
│   └── models_tests_README.md        # Testing documentation for models
├── LICENSE                           # Project license
├── Makefile                          # Build and development commands
├── pyproject.toml                    # Package configuration and dependencies
├── README.md                         # Project overview
└── requirements.txt                  # Python dependencies
```

## Processing Pipeline Workflow

The processing pipeline is orchestrated by the `ProcessingPipeline` class in `core/pipeline.py`. It is a modular, stateful, and resumable system built from a series of discrete processing steps. Each step is responsible for a specific phase of the analysis, and its state is saved upon completion, allowing the pipeline to be interrupted and resumed.

### Pipeline Orchestration and Setup

Before the first step (`InitializationStep`) is executed, the `ProcessingPipeline` performs several critical setup tasks:

**Components Involved:**
- `cli.py`: Parses command-line arguments and validates input.
- `core/pipeline.py`: Orchestrates the entire processing workflow.
- `core/state_manager.py`: Checks for existing processing state (`..._info.json`).
- `core/temp_manager.py`: Creates and manages the temporary working directory.
- `data/context.py`: Creates a unified `ProcessingContext` that holds all shared data.

**Workflow Steps:**
1. **Context Creation**: The `ProcessingContext` is built, consolidating paths, configuration, and managers.
2. **Input Validation**: The pipeline's constructor verifies that the video file exists and is not empty.
3. **State Discovery**: The `StateManager` looks for an existing state file. If `force_restart` is enabled, any existing state is deleted.
4. **Resume Decision**: Based on the presence and validity of a state file, the pipeline decides whether to resume processing or start fresh.
5. **Initial State Creation**: If starting a new session, an initial `PipelineState` object is created, which includes extracting core video metadata (e.g., resolution, duration, FPS) and calculating a video file hash for integrity checks.

This setup ensures that the processing environment is ready and the pipeline knows its starting point before individual steps are run.

### 1. Pipeline Initialization (`InitializationStep`)

This is the first formal step in the pipeline. It performs final validation of the video content.

**Components Involved:**
- `core/steps/initialization.py`: The step class that executes this phase.
- `core/video_processor.py`: Provides tools to validate video format and content.

**Workflow Steps:**
1. **Video Format Validation**: The `VideoProcessor` is used to validate the video container and codecs to ensure they are processable by FFmpeg.
2. **Metadata Confirmation**: It confirms that basic video metadata has been read and stores a summary in the step's progress data.
3. **Environment Check**: Confirms that the temporary workspace and pipeline are ready for the upcoming processing-intensive steps.

### 2. Frame Extraction (`FrameExtractionStep`)

This step extracts frames from the video for further analysis.

**Primary Component**: `core/steps/frame_extraction.py` (using `core/frame_extractor.py`)

**Workflow Steps:**
1. **Keyframe Detection**: Use FFmpeg to identify I-frames (keyframes).
2. **Temporal Sampling**: Extract additional frames at regular intervals.
3. **Frame Deduplication**: Remove duplicate frames to create a unique set.
4. **Frame Storage**: Save extracted frames to a temporary directory.
5. **Metadata Creation**: Initialize `FrameData` objects for each extracted frame.

### 3. Face Detection (`FaceDetectionStep`)

This step analyzes the extracted frames to find human faces and filters them to retain only forward-facing individuals.

**Primary Components**:
- `core/steps/face_detection.py`
- `models/face_detector.py`
- `models/head_pose_estimator.py`

**Workflow Steps:**
1. **Model Loading**: The face detection model (e.g., YOLOv8-Face) is loaded via the `ModelManager`.
2. **Batch Processing**: Frames are processed in batches for efficient inference.
3. **Face Inference**: The model is run on each frame to find face bounding boxes.
4. **Confidence Filtering**: Detections below a confidence threshold are discarded.
5. **Head Pose Filtering**: A head pose estimation model is used to analyze each detected face. Detections that are not sufficiently forward-facing (based on yaw, pitch, and roll) are discarded.
6. **State Update**: The final, filtered face detections are recorded in each frame's `FrameData` object.

### 4. Pose and Head Angle Analysis (`PoseAnalysisStep`)

This step analyzes faces and bodies to determine pose and orientation.

**Primary Components**:
- `core/steps/pose_analysis.py`
- `models/pose_estimator.py`
- `models/head_pose_estimator.py`
- `analysis/pose_classifier.py`
- `analysis/head_angle_classifier.py`

**Workflow Steps:**
1. **Model Loading**: Load pose and head pose estimation models.
2. **Body Pose Estimation**: Extract 2D body landmarks for each detected person.
3. **Head Pose Estimation**: Predict yaw, pitch, and roll for each detected face.
4. **Classification**: Classify body poses (e.g., `standing`, `sitting`) and head angles (e.g., `front`, `left_profile`).
5. **Quality Filtering**: Remove poor quality results, such as excessively tilted heads.

### 5. Close-up Detection (`CloseupDetectionStep`)

This step analyzes the framing of each person to classify the shot type (e.g., close-up, medium shot).

**Primary Component**: `core/steps/closeup_detection.py` (using `analysis/closeup_detector.py`)

**Workflow Steps:**
1. **Multi-Factor Analysis**: For each detected person, it analyzes the face bounding box size relative to the frame area. If available, it also considers body pose landmarks (like shoulder width) for a more accurate classification.
2. **Shot Type Classification**: Based on configurable thresholds, it classifies the shot into categories such as `extreme_closeup`, `closeup`, `medium_closeup`, `medium_shot`, or `wide_shot`.
3. **State Update**: The detailed `CloseupDetection` result, including the shot type, is stored in the frame's metadata.

### 6. Quality Assessment (`QualityAssessmentStep`)

This step evaluates the technical quality of each frame that contains a person.

**Primary Component**: `core/steps/quality_assessment.py` (using `analysis/quality_assessor.py`)

**Workflow Steps:**
1. **Candidate Filtering**: The step first filters the list of all frames to only those that contain detected faces and poses, as only these are relevant for final output.
2. **Quality Metrics**: For each candidate frame, it calculates metrics for sharpness (blur), edge definition, brightness, and contrast.
3. **Scoring Algorithm**: These metrics are combined into a single, weighted `overall_quality` score.
4. **Global Ranking**: After all frames are assessed, they are ranked globally based on their quality score. This rank is stored in the frame's metadata and is critical for the final selection step.
5. **State Update**: The detailed `QualityMetrics` and the global `quality_rank` are stored in each frame's `FrameData` object.

### 7. Frame Selection (`FrameSelectionStep`)

This step selects the best and most diverse frames for the final output, based on all previously gathered data.

**Primary Component**: `core/steps/frame_selection.py` (using `analysis/frame_selector.py`)

**Workflow Steps:**
1. **Dynamic Quotas**: The number of frames to select per category is dynamically adjusted based on the video's duration to ensure sufficient coverage for longer videos.
2. **Category Grouping**: Usable frames (those with sufficient quality) are grouped by criteria like body pose (e.g., `standing`, `closeup`) and head angle (e.g., `front`, `profile_left`).
3. **Advanced Scoring**: Within each category, frames are ranked using a weighted score that considers not only the global quality assessment but also other factors like the relative size of the face in the frame.
4. **Diverse Selection**: The final selection algorithm iterates through the ranked frames in each category, picking the top-ranked ones while ensuring they are visually distinct from each other to maximize diversity.
5. **State Update**: A list of all unique selected frame IDs is stored in the pipeline state, and the `FrameData` for each selected frame is updated to indicate which categories it was chosen for.

### 8. Output Generation (`OutputGenerationStep`)

The final step generates the output images based on the selections from the previous step.

**Primary Component**: `core/steps/output_generation.py` (using `output/image_writer.py` and `output/naming_convention.py`)

**Workflow Steps:**
1. **File Naming**: A structured filename is generated for each output image, based on the source frame, the category it was selected for (e.g., `pose_standing`, `head_angle_front`), and its quality rank.
2. **Image Generation**: For each selected frame, the step generates one or more output files:
    - **Full Frames**: Saved for categories where the full context is important (e.g., `standing`).
    - **Pose Crops**: If enabled, crops focusing on the person's body are generated.
    - **Face Crops**: If enabled, tight crops of the face are generated for head angle categories.
3. **Image Encoding**: All output images are saved as high-quality JPEGs (or other configured formats) in the final output directory.
4. **State Update**: A list of all generated output file paths is added to the pipeline's processing statistics.

### State Persistence and Cleanup

Two important aspects of the pipeline are handled by the main `ProcessingPipeline` orchestrator rather than a specific step:

1.  **Metadata Persistence**: The complete state of the pipeline is continuously saved to a `{video_base}_info.json` file inside the temporary working directory. The `StateManager` handles this process, and it is updated after every successfully completed step. This file serves as the single source of truth for the entire process, enabling resumability. There is no separate "final" metadata file; this state file contains all the collected information.

2.  **Temporary Directory Cleanup**: The temporary directory, which contains all extracted frames and the state file, is managed by the `TempManager`. The `ProcessingPipeline` decides whether to delete this directory upon completion (or failure) based on the user's configuration (`--keep-temp` flag and other settings). This cleanup is not part of any single step but is a final action performed by the pipeline orchestrator.

## Major Domain Objects and Classes

### Core Pipeline Classes

#### `ProcessingPipeline`
**File**: `core/pipeline.py`

**Purpose**: Orchestrates the entire video processing workflow

**Key Public Methods:**
```python
def __init__(self, video_path: str, config: Config, formatter: Optional[Any] = None)
def process() -> ProcessingResult
def resume() -> ProcessingResult
def get_status() -> PipelineStatus
def interrupt_gracefully() -> None
def is_interrupted() -> bool
```

**Key Responsibilities:**
- Coordinate all processing steps
- Handle interruption and resumption logic
- Manage `PipelineState` via the `StateManager`
- Manage progress display and user feedback
- Centralized error handling and recovery

#### `StateManager`
**File**: `core/state_manager.py`

**Purpose**: Manages pipeline state persistence and resumption

**Key Public Methods:**
```python
def __init__(self, video_path: str, temp_manager: 'TempManager')
def load_state() -> Optional[PipelineState]
def save_state(state: PipelineState) -> None
def update_step_progress(step: str, progress: Dict) -> None
def mark_step_complete(step: str) -> None
def can_resume() -> bool
def get_resume_point() -> Optional[str]
def delete_state() -> None
def get_state_info() -> Optional[Dict[str, Any]]
```

**Key Responsibilities:**
- Read/write state file (`..._info.json`) in the temp directory
- Validate video file integrity using a hash
- Track step completion status
- Enable graceful resumption

#### `VideoProcessor`
**File**: `core/video_processor.py`

**Purpose**: High-level video analysis and metadata extraction

**Key Public Methods:**
```python
def __init__(self, video_path: str)
def extract_metadata() -> VideoMetadata
def calculate_hash() -> str
def validate_format() -> bool
def get_duration() -> float
def get_frame_count() -> int
def get_video_info_summary() -> Dict[str, Any]
```

**VideoMetadata Structure:**
- Basic properties: `duration`, `fps`, `width`, `height`, `codec`
- Extended properties: `total_frames`, `file_size_bytes`, `format`
- Computed properties: `aspect_ratio`, `resolution_string`

#### `TempManager`
**File**: `core/temp_manager.py`

**Purpose**: Manages temporary directory lifecycle

**Key Public Methods:**
```python
def __init__(self, video_path: str)
def create_temp_structure() -> Path
def cleanup_temp_files() -> None
def get_temp_path() -> Path
def get_frames_dir() -> Path
def get_temp_file_path(filename: str, subdir: Optional[str] = None) -> Path
def get_temp_usage_info() -> dict
```

### AI Model Classes

#### `ModelManager`
**File**: `models/model_manager.py`

**Purpose**: Handles model downloading, caching, and version management

**Key Public Methods:**
```python
def __init__(self, cache_dir: Optional[str] = None)
def ensure_model_available(model_name: str) -> Path
def download_model(model_name: str) -> Path
def get_model_path(model_name: str) -> Path
def is_model_cached(model_name: str) -> bool
def list_cached_models() -> List[str]
def get_cache_size() -> int
def clear_cache() -> None
```

#### `FaceDetector`
**File**: `models/face_detector.py`

**Purpose**: Face detection inference using SCRFD or YOLO models

**Key Public Methods:**
```python
def __init__(self, model_name: str, device: str = "cpu", confidence_threshold: float = 0.3)
def detect_faces(image: np.ndarray) -> List[FaceDetection]
def detect_batch(images: List[np.ndarray]) -> List[List[FaceDetection]]
def set_confidence_threshold(threshold: float) -> None
```

#### `PoseEstimator`
**File**: `models/pose_estimator.py`

**Purpose**: Body pose estimation using YOLOv8-Pose or similar models

**Key Public Methods:**
```python
def __init__(self, model_name: str, device: str = "cpu", confidence_threshold: float = 0.3)
def estimate_pose(image: np.ndarray) -> List[PoseDetection]
def estimate_batch(images: List[np.ndarray]) -> List[List[PoseDetection]]
def get_keypoint_names() -> List[str]
```

#### `HeadPoseEstimator`
**File**: `models/head_pose_estimator.py`

**Purpose**: Head orientation estimation using HopeNet or similar models

**Key Public Methods:**
```python
def __init__(self, model_name: str, device: str = "cpu", confidence_threshold: float = 0.3)
def estimate_head_pose(face_image: np.ndarray) -> HeadPoseResult
def estimate_batch(face_images: List[np.ndarray]) -> List[HeadPoseResult]
def angles_to_direction(yaw: float, pitch: float, roll: float) -> str
```

### Analysis Classes

#### `PoseClassifier`
**File**: `analysis/pose_classifier.py`

**Purpose**: Classifies body poses from keypoint data

**Key Public Methods:**
```python
def __init__(self)
def classify_pose(pose_detection: PoseDetection, image_shape: Tuple[int, int]) -> List[Tuple[str, float]]
def classify_poses(pose_detections: List[PoseDetection], image_shape: Tuple[int, int]) -> List[List[Tuple[str, float]]]
def is_standing(keypoints: Dict) -> bool
def is_sitting(keypoints: Dict) -> bool
def is_squatting(keypoints: Dict) -> bool
def is_closeup(keypoints: Dict, bbox: Tuple, image_shape: Tuple) -> bool
```

#### `HeadAngleClassifier`
**File**: `analysis/head_angle_classifier.py`

**Purpose**: Classifies head orientation into cardinal directions

**Key Public Methods:**
```python
def __init__(self)
def classify_head_angle(yaw: float, pitch: float, roll: float) -> str
def classify_head_pose(head_pose_result: HeadPoseResult) -> Tuple[str, float]
def classify_head_poses(head_pose_results: List[HeadPoseResult]) -> List[Tuple[str, float]]
def is_valid_orientation(roll: float) -> bool
def get_angle_ranges() -> Dict[str, Dict[str, Tuple[float, float]]]
```

#### `QualityAssessor`
**File**: `analysis/quality_assessor.py`

**Purpose**: Evaluates image quality using multiple metrics

**Key Public Methods:**
```python
def __init__(self)
def assess_quality(image: np.ndarray) -> QualityMetrics
def calculate_laplacian_variance(gray_image: np.ndarray) -> float
def calculate_sobel_variance(gray_image: np.ndarray) -> float
def assess_brightness(gray_image: np.ndarray) -> float
def assess_contrast(gray_image: np.ndarray) -> float
def calculate_overall_score(laplacian_variance: float, sobel_variance: float, brightness_score: float, contrast_score: float) -> float
def identify_quality_issues(laplacian_variance: float, brightness_score: float, contrast_score: float) -> List[str]
```

#### `FrameSelector`
**File**: `analysis/frame_selector.py`

**Purpose**: Selects best frames based on quality and diversity

**Key Public Methods:**
```python
def __init__(self, criteria: SelectionCriteria)
def select_best_frames(candidate_frames: List[FrameData], progress_callback: Optional[Callable[[str], None]] = None) -> SelectionSummary
def rank_by_quality(frames: List[FrameData]) -> List[FrameData]
def group_by_pose(frames: List[FrameData]) -> Dict[str, List[FrameData]]
def group_by_head_angle(frames: List[FrameData]) -> Dict[str, List[FrameData]]
```

### Data Model Classes

#### `FrameData`
**File**: `data/frame_data.py`

**Purpose**: Represents all data associated with a single frame

**Key Attributes:**
```python
@dataclass
class FrameData:
    frame_id: str
    file_path: Path
    source_info: SourceInfo
    image_properties: ImageProperties
    face_detections: List[FaceDetection]
    pose_detections: List[PoseDetection]
    head_poses: List[HeadPoseResult]
    quality_metrics: Optional[QualityMetrics]
    closeup_detections: List[CloseupDetection]
    selections: SelectionInfo
    processing_steps: Dict[str, ProcessingStepInfo]
    processing_timings: ProcessingTimings
```

#### `PipelineState`
**File**: `data/pipeline_state.py`

**Purpose**: Represents the complete state of pipeline processing

**Key Attributes:**
```python
@dataclass
class PipelineState:
    video_file: str
    video_hash: str
    video_metadata: VideoMetadata
    model_versions: Dict[str, str]
    created_at: datetime
    last_updated: datetime
    current_step: str
    completed_steps: List[str]
    failed_steps: List[str]
    step_progress: Dict[str, StepProgress]
    processing_stats: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    frames: List[FrameData]
```

#### `StepProgress`
**File**: `data/pipeline_state.py`

**Purpose**: Tracks detailed progress for individual pipeline steps

**Key Attributes:**
```python
@dataclass
class StepProgress:
    total_items: int
    processed_count: int
    completed: bool
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    step_data: Dict[str, Any]  # Step-specific metadata
```

#### `DetectionResults`
**File**: `data/detection_results.py`

**Purpose**: Data structures for AI model outputs

**Key Classes:**
```python
@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]]

@dataclass
class PoseDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    keypoints: Dict[str, Tuple[float, float, float]]
    pose_classifications: List[Tuple[str, float]]

@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float
    confidence: float
    direction: str

@dataclass
class CloseupDetection:
    is_closeup: bool
    shot_type: str
    confidence: float
    face_area_ratio: float

@dataclass
class QualityMetrics:
    laplacian_variance: float
    sobel_variance: float
    brightness_score: float
    contrast_score: float
    overall_quality: float
    usable: bool

@dataclass
class ProcessingTimings:
    face_detection_ms: Optional[float]
    pose_estimation_ms: Optional[float]
    head_pose_estimation_ms: Optional[float]
    quality_assessment_ms: Optional[float]
```

### Output Classes

#### `ImageWriter`
**File**: `output/image_writer.py`

**Purpose**: Handles high-quality JPEG encoding and file output

**Key Public Methods:**
```python
def __init__(self, config: OutputImageConfig, output_directory: Path, video_base_name: str)
def save_frame_outputs(frame: FrameData, pose_categories: List[str], head_angle_categories: List[str]) -> List[str]
```

#### `NamingConvention`
**File**: `output/naming_convention.py`

**Purpose**: Generates consistent output filenames

**Key Public Methods:**
```python
def __init__(self, video_base_name: str, output_directory: Path)
def get_full_frame_filename(self, frame: FrameData, category: str, rank: int, extension: str = "png") -> str
def get_face_crop_filename(self, frame: FrameData, head_angle: str, rank: int, extension: str = "png") -> str
def validate_filename(filename: str) -> bool
```

### Utility Classes

#### `ProgressManager`
**File**: `utils/progress.py`

**Purpose**: Manages rich console progress displays

**Key Public Methods:**
```python
def __init__(self, console: Optional[Console] = None)
def start_pipeline_progress(self, pipeline_state: PipelineState) -> None
def update_pipeline_state(self, pipeline_state: PipelineState) -> None
def start_step_progress(self, step_name: str, total_items: int, description: str = "") -> None
def update_step_progress(self, step_name: str, processed_count: int, extra_info: Optional[Dict[str, Any]] = None) -> None
def complete_step_progress(self, step_name: str) -> None
def stop_progress(self) -> None
def display_final_summary(results: ProcessingResult) -> None
```

## Error Handling and State Recovery

### Exception Hierarchy

```python
class PersonFromVidError(Exception):
    """Base exception for all Person From Vid errors"""

class ConfigurationError(PersonFromVidError):
    """Configuration-related errors"""

class VideoProcessingError(PersonFromVidError):
    """Base for video processing errors"""

class FrameExtractionError(VideoProcessingError):
    """Frame extraction failures"""

class ModelError(PersonFromVidError):
    """Base for AI model errors"""

class ModelNotFoundError(ModelError):
    """Model not found"""

class ModelDownloadError(ModelError):
    """Model download failures"""

class ModelInferenceError(ModelError):
    """Model inference failures"""

class AnalysisError(PersonFromVidError):
    """Base for analysis errors"""

class QualityAssessmentError(AnalysisError):
    """Quality assessment failures"""

class StateManagementError(PersonFromVidError):
    """Base for state management errors"""

class StateLoadError(StateManagementError):
    """State loading failures"""

class StateSaveError(StateManagementError):
    """State saving failures"""

class OutputError(PersonFromVidError):
    """Base for output generation errors"""

class ImageWriteError(OutputError):
    """Image writing failures"""
```

### Graceful Interruption

The pipeline supports interruption via SIGINT (Ctrl+C) with state preservation:

1. **Signal Handler**: Registers SIGINT handler during pipeline initialization
2. **State Checkpoints**: Regular state saves at each processing step
3. **Graceful Shutdown**: Completes current operation before stopping
4. **Resource Cleanup**: Ensures temporary files are properly managed

### Resume Logic

When restarting processing on a previously interrupted video:

1. **State Validation**: Verify video file hasn't changed (hash comparison)
2. **Step Recovery**: Resume from the last incomplete step
3. **Partial Progress**: Continue from within partially completed steps
4. **Model Consistency**: Ensure same model versions are used

This architecture provides a robust, modular, and maintainable foundation for the Person From Vid package, with clear separation of concerns and comprehensive error handling.