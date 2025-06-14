# Person From Vid - Architecture Documentation

## Overview

This document provides a detailed technical overview of the Person From Vid architecture, including the processing pipeline, core domain objects, and project structure.

## Project Structure

```
personfromvid/
├── personfromvid/                    # Main package directory
│   ├── __init__.py                   # Package initialization
│   ├── cli.py                        # Command-line interface entry point
│   ├── core/                         # Core processing modules
│   │   ├── __init__.py
│   │   ├── pipeline.py               # Main processing pipeline orchestrator
│   │   ├── state_manager.py          # Pipeline state persistence and resumption
│   │   ├── video_processor.py        # Video analysis and frame extraction
│   │   ├── frame_extractor.py        # Keyframe extraction logic
│   │   └── temp_manager.py           # Temporary directory management
│   ├── models/                       # AI model management
│   │   ├── __init__.py
│   │   ├── model_manager.py          # Model downloading and caching
│   │   ├── face_detector.py          # Face detection inference
│   │   ├── pose_estimator.py         # Body pose estimation
│   │   ├── head_pose_estimator.py    # Head orientation analysis
│   │   └── model_configs.py          # Model configuration and metadata
│   ├── analysis/                     # Image analysis modules
│   │   ├── __init__.py
│   │   ├── pose_classifier.py        # Body pose classification logic
│   │   ├── head_angle_classifier.py  # Head angle classification
│   │   ├── quality_assessor.py       # Image quality evaluation
│   │   ├── closeup_detector.py       # Close-up shot identification
│   │   └── frame_selector.py         # Best frame selection logic
│   ├── output/                       # Output generation
│   │   ├── __init__.py
│   │   ├── image_writer.py           # JPEG encoding and file generation
│   │   ├── naming_convention.py      # File naming logic
│   │   └── metadata_writer.py        # Metadata and info file generation
│   ├── utils/                        # Utility modules
│   │   ├── __init__.py
│   │   ├── progress.py               # Rich progress display management
│   │   ├── logging.py                # Logging configuration
│   │   ├── validation.py             # Input validation utilities
│   │   └── exceptions.py             # Custom exception classes
│   └── data/                         # Data models and structures
│       ├── __init__.py
│       ├── frame_data.py             # Frame metadata structures
│       ├── detection_results.py      # AI model output structures
│       ├── pipeline_state.py         # Pipeline state data models
│       └── config.py                 # Configuration management
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── fixtures/                     # Test data and mock files
├── docs/                             # Documentation
│   ├── specification.md              # Technical specification
│   ├── architecture.md               # This file
│   └── api_reference.md              # API documentation
├── scripts/                          # Development and utility scripts
├── pyproject.toml                    # Package configuration
├── requirements.txt                  # Dependencies
└── README.md                         # Project overview
```

## Processing Pipeline Workflow

### 1. Pipeline Initialization

The pipeline begins when a user runs the CLI command:

```bash
personfromvid video.mp4
```

**Components Involved:**
- `cli.py`: Parses command-line arguments and validates input
- `pipeline.py`: Orchestrates the entire processing workflow
- `state_manager.py`: Checks for existing processing state
- `temp_manager.py`: Creates or manages temporary directories

**Workflow Steps:**
1. **Input Validation**: Verify video file exists and is readable
2. **State Discovery**: Look for existing `{video_base}_info.json` file
3. **Resume Decision**: Determine if processing should resume or start fresh
4. **Temporary Setup**: Create `.personfromvid_temp` directory structure
5. **Progress Initialization**: Set up rich console progress displays

### 2. Frame Extraction Phase

**Primary Component**: `frame_extractor.py`

**Workflow Steps:**
1. **Video Analysis**: Extract metadata (duration, FPS, resolution, codec)
2. **Keyframe Detection**: Use FFmpeg to identify I-frames from compression metadata
3. **Temporal Sampling**: Extract additional frames at 0.25-second intervals
4. **Frame Deduplication**: Remove duplicate frames between methods
5. **Frame Storage**: Save extracted frames to temporary directory
6. **Metadata Creation**: Initialize frame metadata with source information

**State Updates:**
- `frame_extraction.total_frames`: Total frames extracted
- `frame_extraction.completed`: Mark phase as complete

### 3. Face Detection Phase

**Primary Component**: `face_detector.py`

**Workflow Steps:**
1. **Model Loading**: Download and cache face detection model (SCRFD/YOLO)
2. **Batch Processing**: Process frames in efficient batches
3. **Face Detection**: Run inference to identify face bounding boxes
4. **Confidence Filtering**: Keep only high-confidence detections
5. **Metadata Update**: Record face locations and confidence scores
6. **Frame Filtering**: Continue only with frames containing faces

**State Updates:**
- `face_detection.processed_count`: Frames processed so far
- `face_detection.faces_found`: Total faces detected
- `face_detection.completed`: Mark phase as complete

### 4. Pose and Head Angle Analysis Phase

**Primary Components**: `pose_estimator.py`, `head_pose_estimator.py`, `pose_classifier.py`, `head_angle_classifier.py`

**Workflow Steps:**

**Body Pose Analysis:**
1. **Model Loading**: Load pose estimation model (YOLOv8-Pose/ViTPose)
2. **Keypoint Detection**: Extract 2D body landmarks
3. **Pose Classification**: Classify into `standing`, `sitting`, `squatting` based on joint angles
4. **Close-up Detection**: Identify close-up shots based on face bounding box size

**Head Pose Analysis:**
1. **Model Loading**: Load head pose estimation model (HopeNet)
2. **Angle Estimation**: Predict yaw, pitch, roll angles
3. **Direction Classification**: Classify into 9 cardinal directions
4. **Quality Filtering**: Remove excessively tilted heads (roll > 30°)

**State Updates:**
- `pose_analysis.processed_count`: Frames analyzed
- `pose_analysis.poses_found`: Counts by pose category
- `pose_analysis.head_angles_found`: Counts by head direction
- `pose_analysis.completed`: Mark phase as complete

### 5. Quality Assessment and Selection Phase

**Primary Components**: `quality_assessor.py`, `frame_selector.py`

**Workflow Steps:**
1. **Quality Metrics**: Calculate multiple blur and quality metrics per frame
2. **Scoring Algorithm**: Combine metrics into unified quality score
3. **Category Grouping**: Group frames by pose category and head angle
4. **Selection Logic**: Select top 3 frames per category based on quality
5. **Crop Planning**: Define face crop regions for head angle outputs

**State Updates:**
- `quality_assessment.completed`: Mark phase as complete

### 6. Output Generation Phase

**Primary Components**: `image_writer.py`, `naming_convention.py`, `metadata_writer.py`

**Workflow Steps:**
1. **File Naming**: Generate consistent filenames using naming convention
2. **Image Encoding**: Save full frames and face crops as high-quality JPEG
3. **Metadata Finalization**: Create comprehensive info JSON file
4. **File Organization**: Save outputs to video's source directory
5. **Cleanup**: Remove temporary directory while preserving info file

**State Updates:**
- `output_generation.completed`: Mark processing as complete

## Major Domain Objects and Classes

### Core Pipeline Classes

#### `ProcessingPipeline`
**File**: `core/pipeline.py`

**Purpose**: Orchestrates the entire video processing workflow

**Key Public Methods:**
```python
def __init__(self, video_path: str, config: Config)
def process() -> ProcessingResult
def resume() -> ProcessingResult
def get_status() -> PipelineStatus
def interrupt_gracefully() -> None
```

**Key Responsibilities:**
- Coordinate all processing phases
- Handle interruption and resumption logic
- Manage progress display and user feedback
- Error handling and recovery

#### `StateManager`
**File**: `core/state_manager.py`

**Purpose**: Manages pipeline state persistence and resumption

**Key Public Methods:**
```python
def __init__(self, video_path: str)
def load_state() -> Optional[PipelineState]
def save_state(state: PipelineState) -> None
def update_step_progress(step: str, progress: Dict) -> None
def mark_step_complete(step: str) -> None
def can_resume() -> bool
def get_resume_point() -> Optional[str]
```

**Key Responsibilities:**
- Read/write info JSON files
- Validate video file integrity
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
```

### AI Model Classes

#### `ModelManager`
**File**: `models/model_manager.py`

**Purpose**: Handles model downloading, caching, and version management

**Key Public Methods:**
```python
def __init__(self, cache_dir: Optional[str] = None)
def download_model(model_name: str) -> Path
def get_model_path(model_name: str) -> Path
def is_model_cached(model_name: str) -> bool
def get_model_info(model_name: str) -> ModelInfo
def cleanup_old_versions() -> None
```

#### `FaceDetector`
**File**: `models/face_detector.py`

**Purpose**: Face detection inference using SCRFD or YOLO models

**Key Public Methods:**
```python
def __init__(self, model_path: str, device: str = "cpu")
def detect_faces(image: np.ndarray) -> List[FaceDetection]
def detect_batch(images: List[np.ndarray]) -> List[List[FaceDetection]]
def set_confidence_threshold(threshold: float) -> None
```

#### `PoseEstimator`
**File**: `models/pose_estimator.py`

**Purpose**: Body pose estimation using YOLOv8-Pose or similar models

**Key Public Methods:**
```python
def __init__(self, model_path: str, device: str = "cpu")
def estimate_pose(image: np.ndarray) -> List[PoseDetection]
def estimate_batch(images: List[np.ndarray]) -> List[List[PoseDetection]]
def get_keypoint_names() -> List[str]
```

#### `HeadPoseEstimator`
**File**: `models/head_pose_estimator.py`

**Purpose**: Head orientation estimation using HopeNet or similar models

**Key Public Methods:**
```python
def __init__(self, model_path: str, device: str = "cpu")
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
def classify_pose(keypoints: Dict[str, Tuple[float, float, float]]) -> PoseClassification
def calculate_joint_angles(keypoints: Dict) -> Dict[str, float]
def is_standing(keypoints: Dict) -> bool
def is_sitting(keypoints: Dict) -> bool
def is_squatting(keypoints: Dict) -> bool
```

#### `HeadAngleClassifier`
**File**: `analysis/head_angle_classifier.py`

**Purpose**: Classifies head orientation into cardinal directions

**Key Public Methods:**
```python
def __init__(self)
def classify_head_angle(yaw: float, pitch: float, roll: float) -> str
def is_valid_orientation(roll: float) -> bool
def get_angle_ranges() -> Dict[str, Tuple[float, float]]
```

#### `QualityAssessor`
**File**: `analysis/quality_assessor.py`

**Purpose**: Evaluates image quality using multiple metrics

**Key Public Methods:**
```python
def __init__(self)
def assess_quality(image: np.ndarray) -> QualityMetrics
def calculate_laplacian_variance(image: np.ndarray) -> float
def calculate_sobel_variance(image: np.ndarray) -> float
def assess_brightness(image: np.ndarray) -> float
def calculate_overall_score(metrics: QualityMetrics) -> float
```

#### `FrameSelector`
**File**: `analysis/frame_selector.py`

**Purpose**: Selects best frames based on quality and diversity

**Key Public Methods:**
```python
def __init__(self)
def select_best_frames(candidates: List[FrameData], count: int = 3) -> List[FrameData]
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
    timestamp: float
    source_info: SourceInfo
    image_properties: ImageProperties
    face_detections: List[FaceDetection]
    pose_detections: List[PoseDetection]
    head_poses: List[HeadPoseResult]
    quality_metrics: Optional[QualityMetrics]
    selections: SelectionInfo
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
    pose_classification: Optional[str]

@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float
    confidence: float
    direction: str
```

### Output Classes

#### `ImageWriter`
**File**: `output/image_writer.py`

**Purpose**: Handles high-quality JPEG encoding and file output

**Key Public Methods:**
```python
def __init__(self, quality: int = 95)
def save_frame(image: np.ndarray, output_path: Path) -> None
def save_face_crop(image: np.ndarray, face_bbox: Tuple, output_path: Path) -> None
def save_batch(frames_and_paths: List[Tuple[np.ndarray, Path]]) -> None
```

#### `NamingConvention`
**File**: `output/naming_convention.py`

**Purpose**: Generates consistent output filenames

**Key Public Methods:**
```python
def __init__(self, video_base_name: str)
def get_pose_filename(pose: str, sequence: int) -> str
def get_face_filename(head_angle: str, sequence: int) -> str
def get_info_filename() -> str
def validate_filename(filename: str) -> bool
```

### Utility Classes

#### `ProgressManager`
**File**: `utils/progress.py`

**Purpose**: Manages rich console progress displays

**Key Public Methods:**
```python
def __init__(self)
def create_main_progress() -> Progress
def create_step_progress(step_name: str, total: int) -> TaskID
def update_step_progress(task_id: TaskID, advance: int) -> None
def add_statistics_panel(stats: Dict[str, Any]) -> None
def display_final_summary(results: ProcessingResult) -> None
```

## Error Handling and State Recovery

### Exception Hierarchy

```python
class PersonFromVidError(Exception):
    """Base exception for all Person From Vid errors"""

class VideoProcessingError(PersonFromVidError):
    """Errors during video analysis and frame extraction"""

class ModelError(PersonFromVidError):
    """Errors related to AI model loading or inference"""

class StateError(PersonFromVidError):
    """Errors in state management and persistence"""

class QualityError(PersonFromVidError):
    """Errors in image quality assessment"""

class OutputError(PersonFromVidError):
    """Errors during output file generation"""
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