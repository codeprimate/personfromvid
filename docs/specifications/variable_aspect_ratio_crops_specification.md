# Feature: Variable Aspect Ratio Crop Generation with "any" Option

## Problem Statement

The current crop generation system provides two distinct modes: no cropping (when `--crop-ratio` is not specified) and fixed aspect ratio cropping (when `--crop-ratio` is specified with values like "1:1", "16:9", etc.). However, there is no option to enable cropping with padding while maintaining the natural variable aspect ratios of detected faces and poses.

Users who want to generate cropped images with consistent padding but preserve the natural proportions of their subjects currently cannot achieve this. They must either:
1. Use full frames (no cropping), which includes unwanted background content
2. Use fixed aspect ratios (e.g., "1:1", "16:9"), which may distort or add excessive black bars to their content

This feature addresses the gap by introducing a `--crop-ratio any` option that enables pose cropping with padding while preserving the natural aspect ratios of the detected bounding boxes.

## Requirements

### Functional Requirements

- **Variable Aspect Ratio Cropping**: Generate crops that maintain the natural aspect ratio of detected faces and poses when `--crop-ratio any` is specified.
- **Automatic Cropping Enablement**: The `--crop-ratio any` parameter automatically enables `enable_pose_cropping` just like other crop-ratio values.
- **Padding Application**: Apply the configured padding (`face_crop_padding` and `pose_crop_padding`) to the detected bounding boxes before cropping.
- **No Aspect Ratio Enforcement**: Skip the fixed aspect ratio calculation step (`calculate_fixed_aspect_ratio_bbox`) when "any" is specified.
- **Integration with Existing Pipeline**: Work seamlessly with existing resize, face restoration, and output generation features.
- **Consistent Behavior**: Maintain the same automatic cropping enablement behavior as other crop-ratio values.

### Technical Constraints

- **Configuration Validation**: Accept "any" as a valid value for the `crop_ratio` field alongside existing W:H format ratios.
- **Case Insensitivity**: Accept "any", "ANY", "Any" as equivalent values for user convenience.
- **Backward Compatibility**: Maintain all existing functionality for `crop_ratio = None` and specific aspect ratios.
- **CLI Integration**: Integrate with the existing `--crop-ratio` parameter without adding new CLI options.
- **Device Compatibility**: Must work correctly on both CPU and GPU processing environments.

### Edge Cases & Error Handling

- **Input Validation**: Properly validate and handle "any" as a special string value distinct from aspect ratio parsing.
- **Case Variations**: Handle different case variations of "any" consistently.
- **Empty/Null Bounding Boxes**: Handle edge cases where detected bounding boxes are invalid or empty.
- **Image Boundary Handling**: Apply existing boundary constraint logic when padding extends beyond image boundaries.
- **Configuration Dependencies**: Ensure proper validation that "any" still requires `enable_pose_cropping = True`.

## Technical Approach

### Implementation Strategy

This feature extends the existing crop ratio system by adding a special case handler for the "any" value. The implementation leverages the existing padding and cropping infrastructure while bypassing the fixed aspect ratio calculation step.

- **Configuration Enhancement**: Modify the `crop_ratio` field validation in `OutputImageConfig` to accept "any" as a valid value.
- **Image Processor Modification**: Update the `expand_bbox_to_aspect_ratio` method in `ImageProcessor` to handle the "any" case.
- **CLI Integration**: Update the CLI help text to document the "any" option.
- **Validation Logic**: Ensure "any" follows the same dependency rules as other crop-ratio values.

### Affected Components

- **Configuration File**:
  - `personfromvid/data/config.py`: Modify `validate_crop_ratio_format` to accept "any" as valid input.
- **Image Processing**:
  - `personfromvid/output/image_processor.py`: Update `expand_bbox_to_aspect_ratio` method to handle "any" case.
- **CLI Interface**:
  - `personfromvid/cli.py`: Update help text for `--crop-ratio` parameter to include "any" option.
- **Testing**:
  - `tests/unit/test_config.py`: Add tests for "any" value validation.
  - `tests/integration/test_fixed_ratio_crop_integration.py`: Add integration tests for variable aspect ratio cropping.

### Dependencies & Integration

- **Existing Crop Pipeline**: Builds upon the current `enable_pose_cropping` functionality without modifying core cropping logic.
- **Padding System**: Uses the existing `face_crop_padding` and `pose_crop_padding` configurations.
- **Image Processing**: Integrates with the existing `ImageProcessor.expand_bbox_to_aspect_ratio` method.
- **Configuration System**: Extends the existing `crop_ratio` validation without breaking existing aspect ratio parsing.

## Acceptance Criteria

- [ ] Configuration validation accepts "any" as a valid `crop_ratio` value.
- [ ] Case insensitive handling supports "any", "ANY", "Any" variations.
- [ ] `--crop-ratio any` automatically enables `enable_pose_cropping=True`.
- [ ] Variable aspect ratio crops are generated with proper padding when "any" is specified.
- [ ] Fixed aspect ratio calculation is bypassed when `crop_ratio = "any"`.
- [ ] Existing functionality for specific aspect ratios (e.g., "1:1", "16:9") remains unchanged.
- [ ] Existing functionality when `crop_ratio = None` remains unchanged.
- [ ] Face crops and pose crops both support variable aspect ratio generation.
- [ ] Integration with resize functionality works correctly with variable aspect ratios.
- [ ] Integration with face restoration works correctly with variable aspect ratios.
- [ ] CLI help text documents the "any" option clearly.
- [ ] Configuration dependency validation still applies (requires `enable_pose_cropping = True`).

## Implementation Tasks

- [ ] **Configuration Validation Update**: Modify `validate_crop_ratio_format` in `config.py` to accept "any".
- [ ] **Image Processor Logic**: Update `expand_bbox_to_aspect_ratio` method to handle "any" case.
- [ ] **CLI Documentation**: Update `--crop-ratio` help text to include "any" option.
- [ ] **Unit Tests**: Add tests for "any" value validation and case sensitivity.
- [ ] **Integration Tests**: Create tests for variable aspect ratio crop generation.
- [ ] **Edge Case Tests**: Test boundary conditions and error scenarios with "any" value.
- [ ] **Regression Tests**: Ensure existing fixed aspect ratio functionality is unaffected.
- [ ] **Documentation Updates**: Update README and configuration examples to include "any" option.

## Risk Assessment

### Potential Issues

- **Configuration Parsing**: Risk of "any" being misinterpreted or conflicting with existing aspect ratio parsing logic.
- **Behavior Confusion**: Users may be confused by the difference between `crop_ratio = None` (no cropping) and `crop_ratio = "any"` (variable aspect cropping).
- **Case Sensitivity**: Inconsistent handling of case variations could lead to user confusion.
- **Validation Complexity**: Adding special case validation may complicate the existing robust aspect ratio validation logic.

### Mitigation Strategies

- **Clear Documentation**: Provide clear examples and documentation distinguishing between None, "any", and specific aspect ratios.
- **Comprehensive Testing**: Test all case variations and ensure they behave consistently.
- **Explicit Validation**: Add explicit validation for "any" that occurs before numeric aspect ratio parsing.
- **Error Messages**: Provide clear error messages that help users understand valid crop_ratio options.
- **Regression Protection**: Ensure extensive testing that existing aspect ratio functionality is preserved.

### Investigation Requirements

- **User Experience Impact**: Investigate whether case sensitivity requirements match user expectations.
- **Validation Order**: Confirm the optimal order for validating "any" versus numeric aspect ratios.
- **Help Text Clarity**: Ensure CLI help text clearly explains the distinction between different crop_ratio options.
- **Configuration Examples**: Verify that configuration file examples properly demonstrate the "any" option usage. 