# Multi-Person Video Analysis: Complete System Specification

## 1. Executive Summary

This specification defines a comprehensive multi-person video analysis system that combines person tracking, detection association, and intelligent frame selection. The system maintains data consistency through a centralized `FrameData` architecture while providing robust person identification and tracking capabilities.

### 1.1. Core Objectives

- **Person Identification**: Assign stable identifiers ("person1", "person2", etc.) across video frames
- **Detection Association**: Link face, pose, and head pose detections to specific individuals
- **Intelligent Selection**: Select optimal frames for each person based on quality and diversity
- **Data Consistency**: Maintain centralized data architecture with full serialization support
- **Backward Compatibility**: Preserve existing functionality when tracking is disabled

### 1.2. System Architecture Overview

```
Video Input → Frame Extraction → Multi-Detection → Person Tracking → 
Association → Quality Assessment → Person-Aware Selection → 
Multi-Person Output Generation
```

## 2. Data Architecture Specification

### 2.1. Enhanced Detection Structures

#### 2.1.1. Face Detection Enhancement
```python
@dataclass
class FaceDetection:
    # Existing core fields
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None
    
    # Person tracking fields
    person_id: Optional[str] = None          # "person1", "person2", etc.
    track_id: Optional[int] = None           # Internal tracker ID
    track_confidence: Optional[float] = None  # Association confidence (0.0-1.0)
    track_age: Optional[int] = None          # Frames since track started
    is_new_track: bool = False               # First appearance flag
```

#### 2.1.2. Pose Detection Enhancement
```python
@dataclass
class PoseDetection:
    # Existing core fields
    bbox: Tuple[int, int, int, int]
    confidence: float
    keypoints: Dict[str, Tuple[float, float, float]]
    pose_classifications: List[Tuple[str, float]] = field(default_factory=list)
    
    # Person tracking fields
    person_id: Optional[str] = None          # "person1", "person2", etc.
    track_id: Optional[int] = None           # Internal tracker ID
    track_confidence: Optional[float] = None  # Association confidence (0.0-1.0)
    track_age: Optional[int] = None          # Frames since track started
    is_new_track: bool = False               # First appearance flag
```

#### 2.1.3. Head Pose Enhancement
```python
@dataclass
class HeadPoseResult:
    # Existing core fields
    yaw: float
    pitch: float
    roll: float
    confidence: float
    direction: Optional[str] = None
    
    # Enhanced association fields
    face_id: int = 0                         # Index in face_detections list
    person_id: Optional[str] = None          # Associated person identifier
```

### 2.2. Frame-Level Tracking Metadata

#### 2.2.1. Tracking Summary Structure
```python
@dataclass
class TrackingSummary:
    """Per-frame tracking metadata."""
    
    total_people_detected: int = 0
    active_track_ids: List[int] = field(default_factory=list)
    person_ids_present: List[str] = field(default_factory=list)
    new_tracks_started: List[str] = field(default_factory=list)
    tracks_lost: List[str] = field(default_factory=list)
    tracking_quality_score: Optional[float] = None
    detection_associations: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Format: {"person1": {"face_idx": 0, "pose_idx": 1, "head_pose_idx": 0}}
```

#### 2.2.2. Enhanced FrameData Structure
```python
@dataclass
class FrameData:
    # Existing fields preserved...
    
    # New tracking integration
    tracking_summary: TrackingSummary = field(default_factory=TrackingSummary)
    
    # Enhanced query methods
    def get_detections_for_person(self, person_id: str) -> PersonDetectionGroup
    def get_people_present(self) -> List[str]
    def has_person(self, person_id: str) -> bool
    def get_person_count(self) -> int
```

### 2.3. Pipeline-Level State Management

#### 2.3.1. Global Tracking State
```python
@dataclass
class GlobalTrackingState:
    """Video-wide tracking state."""
    
    person_registry: Dict[int, str] = field(default_factory=dict)  # track_id -> person_id
    next_person_number: int = 1
    total_unique_people: int = 0
    tracking_enabled: bool = True
    tracker_config: Dict[str, Any] = field(default_factory=dict)
    person_appearance_history: Dict[str, List[float]] = field(default_factory=dict)
    # Format: {"person1": [timestamp1, timestamp2, ...]}
```

#### 2.3.2. Enhanced Pipeline State
```python
@dataclass
class PipelineState:
    # Existing fields preserved...
    
    # New tracking state
    tracking_state: GlobalTrackingState = field(default_factory=GlobalTrackingState)
    
    # Enhanced query methods
    def get_frames_with_person(self, person_id: str) -> List[FrameData]
    def get_person_statistics(self) -> Dict[str, Dict[str, Any]]
    def get_tracking_summary(self) -> Dict[str, Any]
```

## 3. Person Tracking Service Specification

### 3.1. Core Tracking Service

#### 3.1.1. PersonTracker Class
```python
class PersonTracker:
    """DeepSORT-based person tracking service."""
    
    def __init__(self, config: TrackingConfig)
    def track_frame(self, frame_data: FrameData) -> FrameData
    def reset_tracking(self) -> None
    def get_tracking_statistics(self) -> Dict[str, Any]
    
    # Private methods
    def _initialize_deepsort(self) -> DeepSort
    def _extract_detections_for_tracking(self, frame_data: FrameData) -> List[Detection]
    def _update_detections_with_tracking(self, frame_data: FrameData, tracks: List[Track]) -> None
    def _assign_person_ids(self, tracks: List[Track]) -> Dict[int, str]
    def _update_tracking_summary(self, frame_data: FrameData, tracks: List[Track]) -> None
```

#### 3.1.2. Detection Association Service
```python
class DetectionAssociator:
    """Associates detections belonging to the same person."""
    
    def associate_detections(self, frame_data: FrameData) -> List[PersonDetectionGroup]
    def _calculate_spatial_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float
    def _assign_tracking_ids_to_groups(self, groups: List[PersonDetectionGroup], tracks: List[Track]) -> None
    def _validate_associations(self, groups: List[PersonDetectionGroup]) -> List[PersonDetectionGroup]
```

#### 3.1.3. Person Detection Group
```python
@dataclass
class PersonDetectionGroup:
    """Grouped detections for a single person."""
    
    person_id: Optional[str] = None
    track_id: Optional[int] = None
    face_detection: Optional[FaceDetection] = None
    pose_detection: Optional[PoseDetection] = None
    head_pose: Optional[HeadPoseResult] = None
    primary_bbox: Tuple[int, int, int, int] = None
    combined_confidence: float = 0.0
    
    def get_best_bbox(self) -> Tuple[int, int, int, int]
    def get_detection_types(self) -> List[str]
    def is_complete_detection(self) -> bool  # Has face AND pose
```

### 3.2. Configuration Specification

#### 3.2.1. Tracking Configuration
```python
@dataclass
class TrackingConfig:
    """Comprehensive tracking configuration."""
    
    # DeepSORT parameters
    max_disappeared: int = 30
    max_distance: float = 0.5
    min_confidence: float = 0.3
    enable_reid: bool = True
    reid_threshold: float = 0.7
    
    # Association parameters
    face_pose_iou_threshold: float = 0.3
    max_association_distance: float = 100.0
    association_confidence_threshold: float = 0.5
    
    # Performance parameters
    batch_size: int = 1
    max_tracks: int = 10
    cleanup_interval: int = 100
    memory_limit_mb: int = 512
    
    # Debug parameters
    save_tracking_visualization: bool = False
    log_track_lifecycle: bool = True
    export_tracking_data: bool = False
```

## 4. Person-Aware Frame Selection Specification

### 4.1. Frame View Architecture

#### 4.1.1. PersonFrameView Class
```python
@dataclass
class PersonFrameView:
    """Frame view focused on a specific person."""
    
    original_frame: FrameData
    person_id: str
    detection_group: PersonDetectionGroup
    
    # Proxy methods that delegate to person-specific data
    def get_pose_classifications(self) -> List[str]
    def get_head_directions(self) -> List[str]
    def is_closeup_shot(self) -> bool
    def get_quality_metrics(self) -> Optional[QualityMetrics]
    def get_person_bbox(self) -> Optional[Tuple[int, int, int, int]]
    def get_person_confidence(self) -> float
    
    # Frame metadata proxies
    @property
    def frame_id(self) -> str
    @property
    def source_info(self) -> SourceInfo
    @property
    def image_properties(self) -> ImageProperties
```

#### 4.1.2. Frame Explosion Service
```python
class FrameExploder:
    """Converts multi-person frames into person-specific views."""
    
    def explode_frames(self, frames: List[FrameData]) -> List[PersonFrameView]
    def _create_person_views(self, frame: FrameData) -> List[PersonFrameView]
    def _validate_person_view(self, view: PersonFrameView) -> bool
    def _enrich_person_view(self, view: PersonFrameView) -> PersonFrameView
```

### 4.2. Enhanced Frame Selection

#### 4.2.1. PersonAwareFrameSelector Class
```python
class PersonAwareFrameSelector(FrameSelector):
    """Frame selector with person identity awareness."""
    
    def select_best_frames(self, frames: List[FrameData]) -> Dict[str, List[PersonFrameView]]
    def _group_frames_by_person(self, frames: List[FrameData]) -> Dict[str, List[PersonFrameView]]
    def _select_frames_for_person(self, views: List[PersonFrameView], person_id: str) -> List[PersonFrameView]
    def _calculate_person_frame_score(self, view: PersonFrameView) -> float
    def _ensure_person_diversity(self, selected_views: List[PersonFrameView]) -> List[PersonFrameView]
```

#### 4.2.2. Selection Criteria Enhancement
```python
@dataclass
class PersonSelectionCriteria(SelectionCriteria):
    """Enhanced selection criteria for multi-person scenarios."""
    
    # Existing criteria preserved...
    
    # Person-specific criteria
    min_frames_per_person: int = 3
    max_frames_per_person: int = 10
    person_diversity_threshold: float = 2.0  # seconds
    prefer_complete_detections: bool = True  # Face + pose preferred
    person_quality_weight: float = 0.4
    cross_person_diversity_weight: float = 0.3
```

## 5. Pipeline Integration Specification

### 5.1. Processing Pipeline Order

```
1. Frame Extraction (existing)
2. Face Detection (existing)
3. Pose Detection (existing)
4. Person Tracking (NEW)
   ├── Detection Association
   ├── Track Assignment
   └── Person ID Assignment
5. Head Pose Estimation (enhanced with person context)
6. Quality Assessment (existing)
7. Closeup Detection (enhanced with person context)
8. Person-Aware Frame Selection (NEW)
9. Multi-Person Output Generation (enhanced)
```

### 5.2. Pipeline Step Implementation

#### 5.2.1. PersonTrackingStep Class
```python
class PersonTrackingStep(PipelineStep):
    """Pipeline step for person tracking and association."""
    
    def __init__(self, config: TrackingConfig)
    def process_frames(self, frames: List[FrameData], pipeline_state: PipelineState) -> List[FrameData]
    def _initialize_tracking_state(self, pipeline_state: PipelineState) -> None
    def _update_global_state(self, frame: FrameData, global_state: GlobalTrackingState) -> None
    def _cleanup_stale_tracks(self, global_state: GlobalTrackingState) -> None
```

#### 5.2.2. PersonAwareSelectionStep Class
```python
class PersonAwareSelectionStep(PipelineStep):
    """Pipeline step for person-aware frame selection."""
    
    def __init__(self, selection_criteria: PersonSelectionCriteria)
    def process_frames(self, frames: List[FrameData], pipeline_state: PipelineState) -> Dict[str, List[PersonFrameView]]
    def _validate_person_consistency(self, frames: List[FrameData]) -> bool
    def _generate_selection_report(self, selections: Dict[str, List[PersonFrameView]]) -> Dict[str, Any]
```

## 6. Output Generation Specification

### 6.1. Multi-Person Output Generator

#### 6.1.1. PersonAwareOutputGenerator Class
```python
class PersonAwareOutputGenerator:
    """Output generator with person identification and labeling."""
    
    def __init__(self, config: OutputConfig)
    def generate_output_files(self, selections: Dict[str, List[PersonFrameView]]) -> List[str]
    def _generate_person_filename(self, view: PersonFrameView, category: str) -> str
    def _save_person_crop(self, view: PersonFrameView, filename: str) -> None
    def _save_annotated_frame(self, view: PersonFrameView, filename: str) -> None
    def _generate_person_summary(self, person_id: str, views: List[PersonFrameView]) -> Dict[str, Any]
```

#### 6.1.2. Output Configuration
```python
@dataclass
class OutputConfig:
    """Configuration for multi-person output generation."""
    
    include_person_id_in_filename: bool = True
    crop_to_person_bbox: bool = False
    annotate_bounding_boxes: bool = True
    generate_person_summaries: bool = True
    create_person_directories: bool = False
    
    # Filename templates
    person_filename_template: str = "{video_name}_{person_id}_{category}_{frame_id}.jpg"
    summary_filename_template: str = "{video_name}_{person_id}_summary.json"
```

### 6.2. Output File Organization

```
output/
├── video_name_person1_standing_frame_0045.jpg
├── video_name_person1_closeup_frame_0123.jpg
├── video_name_person2_sitting_frame_0067.jpg
├── video_name_person2_profile_frame_0234.jpg
├── summaries/
│   ├── video_name_person1_summary.json
│   └── video_name_person2_summary.json
└── tracking_visualization/
    ├── tracking_paths.json
    └── frame_annotations/
```

## 7. Quality Assurance and Testing Specification

### 7.1. Testing Strategy

#### 7.1.1. Unit Testing Requirements
```python
class TestPersonTracking:
    def test_single_person_tracking_consistency()
    def test_multiple_person_tracking_isolation()
    def test_person_reidentification_after_occlusion()
    def test_tracking_data_serialization_integrity()
    def test_detection_association_accuracy()

class TestPersonAwareSelection:
    def test_person_specific_frame_selection()
    def test_cross_person_diversity_enforcement()
    def test_person_view_proxy_methods()
    def test_selection_criteria_application()

class TestMultiPersonOutput:
    def test_person_labeled_filename_generation()
    def test_person_crop_accuracy()
    def test_output_file_organization()
    def test_person_summary_generation()
```

#### 7.1.2. Integration Testing Requirements
```python
class TestPipelineIntegration:
    def test_full_pipeline_with_tracking_enabled()
    def test_backward_compatibility_tracking_disabled()
    def test_pipeline_state_persistence()
    def test_error_recovery_and_graceful_degradation()
    def test_performance_under_load()
```

### 7.2. Performance Benchmarks

#### 7.2.1. Performance Targets
- **Tracking Latency**: < 50ms per frame for up to 5 people
- **Memory Usage**: < 512MB for tracking state
- **Accuracy**: > 95% person identity consistency across frames
- **Throughput**: Process 30 FPS video in real-time on standard hardware

#### 7.2.2. Quality Metrics
- **Track Consistency**: Percentage of frames where person IDs remain stable
- **Association Accuracy**: Percentage of correct face-pose associations
- **Selection Quality**: Diversity and quality scores for selected frames
- **Output Completeness**: Percentage of people with sufficient frame selections

## 8. Configuration and Deployment Specification

### 8.1. Configuration Management

#### 8.1.1. Configuration File Structure
```yaml
# config/multi_person.yml
multi_person:
  tracking:
    enabled: true
    deepsort:
      max_disappeared: 30
      max_distance: 0.5
      min_confidence: 0.3
      enable_reid: true
      reid_threshold: 0.7
    
    association:
      face_pose_iou_threshold: 0.3
      max_association_distance: 100
      confidence_threshold: 0.5
    
    performance:
      batch_size: 1
      max_tracks: 10
      cleanup_interval: 100
      memory_limit_mb: 512
  
  selection:
    min_frames_per_person: 3
    max_frames_per_person: 10
    person_diversity_threshold: 2.0
    prefer_complete_detections: true
  
  output:
    include_person_id_in_filename: true
    crop_to_person_bbox: false
    annotate_bounding_boxes: true
    generate_person_summaries: true
  
  debug:
    save_tracking_visualization: false
    log_track_lifecycle: true
    export_tracking_data: false
```

### 8.2. Migration and Rollout Strategy

#### 8.2.1. Phase 1: Data Structure Foundation (Week 1-2)
- Extend detection dataclasses with tracking fields
- Add tracking metadata to FrameData and PipelineState
- Update serialization methods
- Ensure backward compatibility

#### 8.2.2. Phase 2: Tracking Service Implementation (Week 3-4)
- Implement PersonTracker with DeepSORT integration
- Create DetectionAssociator service
- Build PersonDetectionGroup data structure
- Add tracking pipeline step

#### 8.2.3. Phase 3: Selection Enhancement (Week 5-6)
- Implement PersonFrameView and FrameExploder
- Create PersonAwareFrameSelector
- Update selection criteria and algorithms
- Integrate person-aware selection step

#### 8.2.4. Phase 4: Output and Integration (Week 7-8)
- Implement PersonAwareOutputGenerator
- Update pipeline orchestration
- Add configuration management
- Complete integration testing

#### 8.2.5. Phase 5: Testing and Optimization (Week 9-10)
- Comprehensive testing suite implementation
- Performance optimization and tuning
- Documentation and deployment preparation
- Production readiness validation

## 9. Success Criteria and Validation

### 9.1. Functional Requirements Validation
- ✅ Stable person identification across video frames
- ✅ Accurate association of face, pose, and head pose detections
- ✅ Person-aware frame selection with quality and diversity
- ✅ Multi-person output generation with proper labeling
- ✅ Backward compatibility with existing single-person workflows

### 9.2. Technical Requirements Validation
- ✅ Data consistency maintained through FrameData architecture
- ✅ Full serialization support for all tracking data
- ✅ Configurable performance parameters
- ✅ Comprehensive error handling and recovery
- ✅ Memory-efficient processing for long videos

### 9.3. Quality Requirements Validation
- ✅ > 95% person identity consistency
- ✅ < 5% false association rate for detections
- ✅ Real-time processing capability
- ✅ Graceful degradation under challenging conditions
- ✅ Comprehensive test coverage (> 90%)

This specification provides a complete blueprint for implementing robust multi-person video analysis while maintaining the existing data-first architecture and ensuring system reliability and performance. 