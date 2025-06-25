# Feature: Fixed Aspect Ratio Crop Generation

## Problem Statement

The current crop generation system produces variable aspect ratio crops based on the original bounding box dimensions with padding. Users need an option to generate fixed aspect ratio crops (1:1, 4:3, 16:9, etc.) with consistent dimensions for use cases like social media content, thumbnails, or profile pictures where uniform aspect ratios are required.

The current system generates crops that maintain the natural aspect ratio of detected faces and poses. While this preserves original proportions, many applications require crops with specific aspect ratios and identical dimensions for consistent presentation and processing.

This enhancement integrates with the existing crop generation pipeline (`enable_pose_cropping`) and provides a configurable option to transform variable aspect crops into uniform fixed aspect ratio outputs while respecting existing padding, boundary handling, and face restoration features.

## Requirements

### Functional Requirements

- **Fixed Aspect Ratio Generation**: Generate crops with a specified aspect ratio (e.g., 1:1, 4:3, 16:9) for both face and pose detections when a crop ratio is specified.
- **Conditional Application**: Fixed aspect ratio crops are only generated when `--crop-ratio` is specified, which automatically enables `enable_pose_cropping`.
- **Default Behavior**: When `--crop-ratio` is not specified, the original variable aspect ratio crop behavior is used.
- **Size Determination**: Crop dimensions are determined by the specified aspect ratio, with the `--resize` parameter or `OutputImageConfig.default_crop_size` serving as the maximum extent (longer dimension).
- **Centering Logic**: Fixed aspect ratio crops are centered on the original bounding box's center, respecting existing padding conventions.
- **Boundary Handling**: When a crop extends beyond image boundaries, the crop region is shifted to maintain maximum padding while staying within image bounds.
- **Scaling Integration**: After extracting crops, Lanczos up/downscaling is applied to achieve the exact target dimensions.
- **Replacement Behavior**: Fixed aspect ratio crops completely replace variable aspect ratio crops when enabled (they are not generated in addition to them).
- **Crop Data Storage**: Store calculated crop regions with aspect ratio keys in `FrameData.selections.crop_regions` for use by `ImageWriter` (e.g., "1:1", "4:3"), preserving original bbox data.
- **Universal Application**: Apply fixed aspect ratio cropping to all crop types (face crops, pose crops) when enabled.

### Technical Constraints

- **Configuration Integration**: Integrate with the existing `OutputImageConfig` structure and CLI argument patterns.
- **Aspect Ratio Validation**: Parse and validate standard WxH aspect ratio format with positive integers only (e.g., "16:9", "4:3", "1:1"). The calculated ratio (width/height) must be between 0.1 and 100.0 inclusive. Malformed strings like "16:", ":9", "16/9", or decimal ratios like "1.78:1" must be rejected.
- **Automatic Cropping Enablement**: The `--crop-ratio` parameter automatically enables `enable_pose_cropping` when specified.
- **Device Compatibility**: Must work correctly on both CPU and GPU processing environments.

### Edge Cases & Error Handling

- **Boundary Conditions**: Handle cases where a fixed aspect ratio crop would extend beyond image boundaries by shifting it while preserving the center position as much as possible.
- **Small Detection Handling**: For detections smaller than the target crop size, expand the crop region symmetrically from the center.
- **Large Detection Handling**: For detections larger than the target crop size, crop to the specified aspect ratio while maintaining center positioning.
- **Missing Resize Parameter**: Use `OutputImageConfig.default_crop_size` as the maximum extent when `--crop-ratio` is specified but `--resize` is not provided.
- **Automatic Configuration**: When `--crop-ratio` is specified, `enable_pose_cropping` is automatically set to `True`.
- **Invalid Aspect Ratio Format**: Handle malformed aspect ratio strings (e.g., "16:", ":9", "16/9", "1.5:1", "invalid") and ratios outside the valid range (0.1-100.0).
- **Zero-Padding Edge Cases**: Handle detections at image edges where padding cannot be applied uniformly.

## Technical Approach

### Implementation Strategy

To ensure a clean and maintainable architecture, the cropping logic will be separated into three distinct components: a utility module for calculations, a dedicated class for image processing, and a refactored `ImageWriter` acting as an orchestrator.

- **Configuration Enhancement**: Add `crop_ratio` and `default_crop_size` to `OutputImageConfig`.
- **New Calculation Module (`crop_utils.py`)**: Create a new module `personfromvid/output/crop_utils.py` for all geometric calculations.
    - `def parse_aspect_ratio(ratio_str: str) -> Optional[Tuple[int, int]]`: A pure function to parse and validate aspect ratio strings.
    - `def calculate_fixed_aspect_ratio_bbox(original_bbox: Tuple, image_dims: Tuple, aspect_ratio: Tuple) -> Tuple`: A pure function to calculate the final crop region based on the original detection, target ratio, and image boundaries.
- **New Image Processing Class (`image_processor.py`)**: Create a new module `personfromvid/output/image_processor.py` containing an `ImageProcessor` class to handle all pixel-level manipulations.
    - `def crop_and_resize(self, image: np.ndarray, crop_bbox: Tuple, target_size: Optional[int], ...)`: This method will encapsulate cropping, resizing, and face restoration logic, replacing the current `_crop_region` implementation.
- **ImageWriter Refactoring**: Refactor the existing `ImageWriter` to be a lean orchestrator. It will use `crop_utils` to determine the crop region and delegate image manipulation to the `ImageProcessor`.
- **CLI Integration**: Add the `--crop-ratio` parameter and automatic `enable_pose_cropping` enablement to `personfromvid/cli.py`.
- **Hardcoded Value Replacement**: Replace all hardcoded 512px references related to crop sizing with the configurable `default_crop_size`.

### Affected Components

- **Configuration Files**:
  - `personfromvid/data/config.py`: Add `crop_ratio` and `default_crop_size` fields to `OutputImageConfig` and update validation logic.
- **New Utility Module**:
  - `personfromvid/output/crop_utils.py`: Will contain the new `parse_aspect_ratio` and `calculate_fixed_aspect_ratio_bbox` functions.
- **New Image Processor Module**:
    - `personfromvid/output/image_processor.py`: Will contain the new `ImageProcessor` class responsible for all pixel-level operations.
- **Image Processing (Refactored)**:
  - `personfromvid/output/image_writer.py`: Refactor to orchestrate the output process. It will instantiate and use `ImageProcessor` and `crop_utils`. The `_crop_region` method will be removed and its logic moved to `ImageProcessor`.
- **CLI Interface**:
  - `personfromvid/cli.py`: Add the `--crop-ratio` parameter, automatic `enable_pose_cropping` enablement, and override logic.

### Dependencies & Integration

- **Existing Crop Pipeline**: Builds upon the current `enable_pose_cropping` functionality.
- **Resize Integration**: The new `ImageProcessor` will integrate with the existing `--resize` parameter.
- **Face Restoration**: The `ImageProcessor` will respect the existing face restoration configuration, invoking the `FaceRestorer` when needed.
- **Padding System**: Respects the existing `face_crop_padding` and `pose_crop_padding` settings.

## Acceptance Criteria

- [ ] Configuration option `crop_ratio: Optional[str] = None` is added to `OutputImageConfig`.
- [ ] Configuration option `default_crop_size: int = 640` is added to `OutputImageConfig` with validation (256-4096).
- [ ] CLI parameter `--crop-ratio` is added with proper help text and integration.
- [ ] CLI automatic enablement ensures `--crop-ratio` automatically sets `enable_pose_cropping=True` in the configuration.
- [ ] Aspect ratio parsing validates standard WxH integer format (e.g., "16:9") and ensures the calculated ratio is between 0.1 and 100.0.
- [ ] All hardcoded 512px references related to crop sizing are replaced with `config.output.image.default_crop_size`.
- [ ] `FrameData.selections.crop_regions` is used to store fixed aspect ratio crop data, keyed by aspect ratio string (e.g., "1:1", "4:3").
- [ ] The fixed aspect ratio crop calculation algorithm correctly centers crops on original bbox centers.
- [ ] The boundary shifting algorithm keeps crops within image bounds while preserving maximum padding.
- [ ] Fixed aspect ratio crops are generated using `--resize` or `default_crop_size` as the maximum extent (longer dimension).
- [ ] Face restoration is applied to fixed aspect ratio face crops when enabled.
- [ ] Pose crops and face crops both support fixed aspect ratio crop generation.
- [ ] Fixed aspect ratio crops completely replace variable aspect ratio crops when enabled.
- [ ] Configuration automatically enables `enable_pose_cropping` when `--crop-ratio` is specified.
- [ ] Serialization/deserialization of `crop_regions` works correctly with aspect ratio keys.
- [ ] Key naming convention for `crop_regions` uses the aspect ratio format (e.g., "1:1", "4:3", "16:9").
- [ ] All existing functionality continues to work when `--crop-ratio` is not specified.

## Implementation Tasks

- [ ] **Configuration Updates**: Add `crop_ratio` and `default_crop_size` fields to `OutputImageConfig`.
- [ ] **New Module (`crop_utils.py`)**: Create the `personfromvid/output/crop_utils.py` module.
- [ ] **Aspect Ratio Parser**: Implement `def parse_aspect_ratio(...)` in `crop_utils.py`.
- [ ] **Fixed Aspect Ratio Crop Algorithm**: Implement `def calculate_fixed_aspect_ratio_bbox(...)` in `crop_utils.py`.
- [ ] **New Module (`image_processor.py`)**: Create the `personfromvid/output/image_processor.py` module with the `ImageProcessor` class.
- [ ] **ImageProcessor Implementation**: Move logic from `ImageWriter._crop_region` into `ImageProcessor.crop_and_resize`.
- [ ] **ImageWriter Refactoring**: Refactor `ImageWriter` to use `ImageProcessor` and `crop_utils`, and remove the `_crop_region` method.
- [ ] **Hardcoded Value Replacement**: Replace all 512px crop size references with `config.output.image.default_crop_size`.
- [ ] **CLI Parameter Addition**: Add `--crop-ratio` parameter with proper type and help text.
- [ ] **CLI Automatic Enablement**: Update CLI logic to automatically enable `enable_pose_cropping` when `--crop-ratio` is specified.
- [ ] **CLI Override Logic**: Update `_apply_output_overrides` to handle the `crop_ratio` setting.
- [ ] **Configuration Validation**: Add validation rules for aspect ratio format and `default_crop_size` range.
- [ ] **Unit Tests**: Create comprehensive tests for `crop_utils`, `ImageProcessor`, and new configuration logic.
- [ ] **Integration Tests**: Test the full pipeline with `--crop-ratio` enabled for both face and pose crops.
- [ ] **Configuration Tests**: Test CLI parameter integration, validation, and configuration override behavior.
- [ ] **Edge Case Tests**: Test boundary conditions, small/large detection handling, invalid aspect ratios, and error scenarios.
- [ ] **Documentation Updates**: Update CLI help text, configuration documentation, and README examples.
- **Configuration Integration**: Ensure `crop_ratio` automatically enables `enable_pose_cropping` and integrates properly with `resize` and padding settings.

## Risk Assessment

### Potential Issues

- **Breaking Changes**: Modifying `FrameData` structure could affect state file compatibility if not handled carefully.
- **Boundary Algorithm Complexity**: Edge cases with image boundaries could produce unexpected crop positioning.
- **Configuration Simplicity**: The automatic enablement of `enable_pose_cropping` when `--crop-ratio` is specified should be clearly documented for users.
- **Default Size Conflicts**: The new `default_crop_size` of 640px differs from the previous hardcoded default of 512px, which may alter output dimensions for users who relied on the old behavior without specifying `--resize`.

### Mitigation Strategies

- **Backward Compatibility**: Ensure the `crop_regions` field has proper defaults and doesn't break existing state files.
- **Comprehensive Testing**: Implement extensive boundary condition testing with various image sizes and detection positions.
- **Clear Documentation**: Provide clear examples and error messages for configuration dependencies.
- **Robust Validation**: Implement robust parsing and validation for aspect ratios with clear error messages for invalid formats.
- **Systematic Migration**: Update hardcoded values systematically with proper testing at each step.
- **Actionable Error Messages**: Provide clear, actionable error messages for invalid configuration combinations.

### Investigation Requirements

- **State File Compatibility**: Investigate the impact of using the `crop_regions` field on existing state persistence.
- **Edge Case Enumeration**: Systematically identify and test all boundary condition scenarios.
- **Configuration Integration**: Ensure seamless integration between `crop_ratio`, automatic `enable_pose_cropping` enablement, `resize`, and padding settings. 