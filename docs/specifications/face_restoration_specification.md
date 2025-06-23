# Feature: Real-ESRGAN Face Restoration Integration

## Problem Statement

The current face processing pipeline uses basic Lanczos upscaling for face crops, which produces adequate but not optimal results for face images extracted from videos. Users need higher-quality face restoration using cutting-edge GAN technology to significantly improve the visual appeal, detail, and overall quality of face crops. This enhancement should integrate seamlessly with the existing pipeline while maintaining the established architecture patterns and configuration flexibility.

The current implementation in `ImageWriter._crop_region()` applies Lanczos upscaling when face dimensions are smaller than the target size. While functional, this approach lacks the sophistication needed to restore fine facial details, reduce artifacts, and enhance overall image quality that modern GAN-based restoration can provide.

## Requirements

### Functional Requirements

1. **GAN-Based Face Enhancement**: Apply Real-ESRGAN 4x super-resolution to face crops when face restoration is enabled
2. **Restoration Strength Control**: Provide configurable blending ratio (0.0-1.0) between original and restored faces  
3. **Target Size Logic**: Ensure face crops are sized to at least 512px or the configured resize value, whichever is larger
4. **Three-Stage Processing**: Lanczos resize → Real-ESRGAN 4x upscale → downscale to target with strength blending
5. **Model Integration**: Integrate Real-ESRGAN into existing ModelConfigs and model management system
6. **Configuration Control**: Provide configuration toggles for feature enable/disable (default: enabled) and strength (default: 0.8)
7. **CLI Interface**: Provide `--face-restoration/--no-face-restoration` and `--face-restoration-strength` command-line controls
8. **Fallback Mechanism**: Gracefully fall back to current Lanczos-only processing if Real-ESRGAN fails
9. **Device Support**: Support both CPU and GPU inference following existing device management patterns
10. **Output Compatibility**: Maintain full compatibility with existing output formats (PNG/JPEG) and resize configurations

### Technical Constraints

1. **Performance Impact**: Face restoration will significantly increase processing time per face (acceptable trade-off for quality)
2. **Memory Usage**: Real-ESRGAN model loading and 4x upscaling must respect existing memory management and device selection
3. **Dependency Management**: Add Real-ESRGAN dependencies without breaking existing functionality or dependency versions
4. **Architecture Consistency**: Follow existing model loading, error handling, and logging patterns from other model classes
5. **Model Size**: Real-ESRGAN model (~67MB) must integrate with existing model download and caching system
6. **Device Compatibility**: Must work seamlessly with existing DeviceType.CPU, DeviceType.GPU, and DeviceType.AUTO configurations
7. **Fixed Model Configuration**: No user configurability for model settings (precision, face enhancement, tiling, scale factor)

### Edge Cases & Error Handling

1. **Model Loading Failures**: Handle Real-ESRGAN model download or loading failures with fallback to Lanczos-only processing
2. **Inference Failures**: Handle Real-ESRGAN processing errors without crashing the pipeline, log errors and fallback
3. **Memory Constraints**: Detect and handle GPU out-of-memory conditions with automatic CPU fallback
4. **Invalid Input**: Handle malformed or extremely small face crops gracefully with appropriate error logging
5. **Network Issues**: Handle model download failures with clear error messages and retry logic via existing model manager
6. **Device Switching**: Handle device unavailability (e.g., CUDA not available) with graceful degradation to CPU processing
7. **Strength Parameter Validation**: Ensure restoration strength is clamped to valid range [0.0, 1.0] with clear error messages

## Technical Approach

### Implementation Strategy

This implementation follows the established patterns in the codebase and integrates seamlessly with existing systems:

1. **Model Configuration**: Add Real-ESRGAN model metadata to `ModelConfigs` class using existing `ModelMetadata` pattern
2. **Face Restorer Class**: Create `FaceRestorer` class in `personfromvid/models/face_restorer.py` following `FaceDetector` architecture patterns
3. **Pipeline Integration**: Modify `ImageWriter._crop_region()` method to conditionally apply face restoration with strength blending
4. **Configuration Extension**: Add `face_restoration_enabled` and `face_restoration_strength` to `OutputImageConfig` class
5. **CLI Integration**: Add `--face-restoration/--no-face-restoration` and `--face-restoration-strength` flags to existing CLI argument structure
6. **Error Handling**: Implement comprehensive error handling with fallback mechanisms and detailed logging following existing patterns

### Affected Components

#### Code Areas Requiring Modification
- `personfromvid/models/model_configs.py`: Add `REALESRGAN_X4PLUS` model configuration
- `personfromvid/models/face_restorer.py`: New FaceRestorer class for Real-ESRGAN inference (create new file)
- `personfromvid/output/image_writer.py`: Modify `_crop_region()` method to integrate face restoration
- `personfromvid/data/config.py`: Add `face_restoration_enabled` field to `OutputImageConfig`
- `personfromvid/cli.py`: Add `--face-restoration/--no-face-restoration` CLI argument
- `requirements.txt`: Add `realesrgan` and related dependencies

#### Integration Points
- **Model Manager**: Leverage existing `ModelManager` for Real-ESRGAN model download and caching
- **Device Management**: Use existing device resolution and GPU/CPU switching logic
- **Error Handling**: Follow existing error handling patterns with `ImageWriteError` exceptions
- **Configuration System**: Integrate with existing Pydantic-based configuration validation
- **Logging**: Use existing logger infrastructure for face restoration operations

#### Dependencies & Integration
- **External Dependencies**: `realesrgan` package, compatible PyTorch version
- **Model File**: `RealESRGAN_x4plus.pth` (~67MB) from official Real-ESRGAN repository
- **Internal Integration**: Full integration with existing model management, device selection, and configuration systems
- **Backward Compatibility**: Feature is optional and fully backward compatible

## Acceptance Criteria

- [ ] Real-ESRGAN model configuration (`REALESRGAN_X4PLUS`) is added to `ModelConfigs` with correct URL, hash, and metadata
- [ ] `FaceRestorer` class successfully loads Real-ESRGAN model and performs 4x upscaling on both CPU and GPU
- [ ] Face restoration is applied to ALL face crops when `face_restoration_enabled` is True (default)
- [ ] Processing workflow: Lanczos resize to target → Real-ESRGAN 4x upscale → downscale to target size
- [ ] Target size is correctly calculated as max(512px, configured_resize_value)
- [ ] Configuration option `face_restoration_enabled: bool = True` is added to `OutputImageConfig`
- [ ] Configuration option `face_restoration_strength: float = 0.8` is added to `OutputImageConfig` with validation
- [ ] CLI flag `--face-restoration/--no-face-restoration` controls the feature (enabled by default)
- [ ] CLI argument `--face-restoration-strength` controls restoration intensity (0.0-1.0)
- [ ] Fallback to Lanczos-only processing works correctly when Real-ESRGAN fails
- [ ] Error handling provides clear logging without crashing the pipeline
- [ ] Device management respects existing CPU/GPU/AUTO configuration
- [ ] Memory usage remains manageable for typical video processing workflows
- [ ] All existing functionality and tests continue to work unchanged

## Implementation Tasks

- [ ] **Model Configuration**: Add Real-ESRGAN model to `ModelConfigs.REALESRGAN_X4PLUS` with proper metadata
- [ ] **FaceRestorer Class**: Create `personfromvid/models/face_restorer.py` with Real-ESRGAN inference
- [ ] **Model Loading**: Implement device-aware model loading (CPU/GPU) following existing patterns
- [ ] **Inference Pipeline**: Implement 4x upscale and intelligent downscaling logic
- [ ] **Configuration**: Add `face_restoration_enabled` and `face_restoration_strength` to `OutputImageConfig` with validation
- [ ] **Pipeline Integration**: Modify `ImageWriter._crop_region()` to use FaceRestorer with strength blending when enabled
- [ ] **CLI Integration**: Add `--face-restoration/--no-face-restoration` and `--face-restoration-strength` arguments to main CLI
- [ ] **CLI Override**: Update `_apply_output_overrides()` to handle both face restoration flags
- [ ] **Error Handling**: Implement comprehensive error handling with fallback to Lanczos
- [ ] **Logging**: Add detailed logging for face restoration operations and fallbacks
- [ ] **Dependencies**: Update `requirements.txt` with Real-ESRGAN package
- [ ] **Unit Tests**: Create tests for `FaceRestorer` class covering success and failure cases
- [ ] **Integration Tests**: Test face restoration in full pipeline workflow
- [ ] **Configuration Tests**: Test configuration validation and CLI argument processing
- [ ] **Performance Testing**: Benchmark processing time impact and memory usage
- [ ] **Documentation**: Update CLI help text and configuration documentation

## Risk Assessment

### Potential Issues

- **Significant Performance Impact**: Real-ESRGAN 4x processing will substantially increase face processing time
- **Memory Usage**: Large model and 4x upscaling could cause memory issues, especially on GPU
- **Dependency Conflicts**: Real-ESRGAN dependencies might conflict with existing PyTorch/OpenCV versions
- **Model Download**: 67MB model download could fail on slow/unreliable connections
- **Quality Variability**: Real-ESRGAN might occasionally produce artifacts or lower quality results than expected

### Mitigation Strategies

- **Performance Monitoring**: Implement detailed timing logs and performance metrics for monitoring
- **Memory Management**: Implement GPU memory monitoring and automatic CPU fallback for memory constraints
- **Dependency Management**: Carefully specify compatible dependency versions and test thoroughly
- **Download Robustness**: Leverage existing model manager's checksum validation and retry mechanisms
- **Quality Assurance**: Implement logging to track restoration success/failure rates for monitoring
- **Progressive Rollout**: Feature defaults to enabled but can be easily disabled if issues arise

### Investigation Requirements

- **Performance Benchmarking**: Test Real-ESRGAN processing time on various hardware configurations
- **Memory Profiling**: Profile GPU and CPU memory usage during 4x upscaling operations
- **Quality Assessment**: Evaluate Real-ESRGAN output quality across diverse face types and conditions
- **Dependency Testing**: Test Real-ESRGAN integration with existing PyTorch and OpenCV versions

## Technical Implementation Details

### Real-ESRGAN Model Configuration

```python
REALESRGAN_X4PLUS = ModelMetadata(
    name="realesrgan_x4plus",
    version="1.0.0",
    provider=ModelProvider.DIRECT_URL,
    files=[
        ModelFile(
            filename="RealESRGAN_x4plus.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            sha256_hash="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            size_bytes=67040989,  # ~67MB
            format=ModelFormat.PYTORCH,
            description="Real-ESRGAN 4x super-resolution model for general images (face enhancement enabled, full precision, no tiling)",
        )
    ],
    supported_devices=[DeviceType.CPU, DeviceType.GPU],
    input_size=(None, None),  # Variable input size
    description="Real-ESRGAN 4x super-resolution for image enhancement with face-specific optimizations",
    license="BSD-3-Clause",
    requirements=["realesrgan>=0.3.0"],
)
```

### Configuration Integration

```python
class OutputImageConfig(BaseModel):
    # ... existing fields ...
    
    face_restoration_enabled: bool = Field(
        default=True,
        description="Enable Real-ESRGAN face restoration for enhanced quality",
    )
    
    face_restoration_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Face restoration strength (0.0=no effect, 1.0=full restoration)",
    )
```

### Processing Workflow

When face restoration is enabled, the workflow is:

1. **Input**: Face crop of any size
2. **Target Size Calculation**: `target_size = max(512, config.resize or 512)`
3. **Initial Resize**: Lanczos resize to target_size if needed → **[Original A]**
4. **Real-ESRGAN Processing**: Apply 4x upscale using Real-ESRGAN with face enhancement enabled
5. **Final Downscale**: Downscale back to target_size using high-quality interpolation → **[Restored B]**
6. **Strength Blending**: `final = (strength × B) + ((1-strength) × A)`
7. **Output**: Enhanced face crop at target_size with controlled restoration intensity

When face restoration is disabled, the workflow follows the current logic:
- **Input** → **Lanczos resize to target_size if needed** → **Output**

### Real-ESRGAN Model Configuration Requirements

**Fixed Model Settings** (non-configurable as per requirements):
- **Model**: `RealESRGAN_x4plus.pth` (general purpose model)
- **Precision**: Full precision (FP32) - no half precision allowed
- **Face Enhancement**: Always enabled
- **Tile Size**: No tiling (process full image)
- **Scale Factor**: Fixed 4x upscale

**User Configurable Settings**:
- **Feature Toggle**: Enable/disable face restoration (default: True)
- **Restoration Strength**: 0.0-1.0 blend ratio (default: 0.8)

## Summary

This specification provides comprehensive coverage for integrating Real-ESRGAN face restoration while maintaining consistency with existing architecture patterns and ensuring robust error handling and fallback mechanisms. 