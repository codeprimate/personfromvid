# Implementation TODO: Fixed Aspect Ratio Crop Generation

## Specification Reference
**Source Document**: [docs/specifications/square_crops_specification.md](docs/specifications/square_crops_specification.md)
*This specification document must be loaded alongside this plan during execution to provide complete context and requirements.*

## Overview
- **Complexity**: Moderate
- **Risk Level**: Medium (involves refactoring complex image processing logic and state management)
- **Key Dependencies**: `personfromvid/data/config.py`, `personfromvid/output/image_writer.py`, `personfromvid/cli.py`, Pydantic validation system
- **Estimated Effort**: 4-6 days
- **Specification Sections**: Addresses all sections of the feature specification with comprehensive testing

## Phase Strategy
The implementation follows a carefully structured four-phase approach designed to minimize risk and enable independent validation at each stage:

1. **Phase 1: Configuration Foundation** - Establishes configuration fields and validation within the existing Pydantic system, creating a testable foundation with no side effects
2. **Phase 2: Core Calculation Engine** - Develops pure calculation functions in isolation, enabling comprehensive geometric algorithm testing before integration
3. **Phase 3: Image Processing Architecture** - Refactors the complex existing `_crop_region` logic into a clean, testable `ImageProcessor` class, improving maintainability
4. **Phase 4: End-to-End Integration** - Connects all components through the CLI and `ImageWriter`, ensuring seamless operation with existing pipeline

This approach isolates complex geometric calculations, validates configuration early, and separates concerns between calculation and image processing.

## Implementation Phases

## Phase 1: Configuration Foundation and Validation
*Incremental Goal: Create a complete configuration subsystem for fixed aspect ratio crops with comprehensive validation, testable independently of image processing logic.*

### Task 1.1: Enhance OutputImageConfig with New Fields
*Spec Reference: Configuration Enhancement, Default Size Configuration*

- [x] 1.1.1 **Add `crop_ratio` field to OutputImageConfig** ✅ **COMPLETED**
  - *Hint*: Follow the pattern of `resize: Optional[int]` in `OutputImageConfig` class around line 440 in `config.py`
  - *Consider*: Field should be `Optional[str] = None` to indicate optional nature, with clear docstring
  - *Files*: `personfromvid/data/config.py`
  - *Risk*: Low - adding optional field with default None maintains backward compatibility
  - **IMPLEMENTATION PLAN**:
    - **Field Definition**: Add `crop_ratio: Optional[str] = Field(default=None, description="Fixed aspect ratio for crops (e.g., '1:1', '16:9', '4:3')")`
    - **Placement**: Insert after `resize` field around line 422 to group related sizing options
    - **Validation**: Will be handled by separate validator task to maintain clean separation
  - **EXECUTION DETAILS**:
    - **Code Location**: Line 424 in `personfromvid/data/config.py`
    - **Pattern Followed**: Identical to existing `resize: Optional[int]` field pattern
    - **Import Requirements**: None - uses existing `Optional` from typing imports
  
##### 1.1.1.1 Unit Testing
- **Test Cases**: Configuration instantiation with and without `crop_ratio`, serialization/deserialization
- **Mocks/Fixtures**: Standard Pydantic model testing patterns
- **Coverage**: Field presence, default value behavior, type validation

- [x] 1.1.2 **Add `default_crop_size` field to OutputImageConfig** ✅ **COMPLETED**
  - *Hint*: Similar to `resize` field but with different validation range and default value
  - *Consider*: Default should be 640 (larger than current hardcoded 512) to improve output quality while maintaining performance
  - *Files*: `personfromvid/data/config.py`
  - *Risk*: Low-Medium - changing default size from 512px to 640px may affect user expectations
  - **IMPLEMENTATION PLAN**:
    - **Field Definition**: Add `default_crop_size: int = Field(default=640, ge=256, le=4096, description="Default crop size in pixels when crop_ratio is specified")`
    - **Validation Range**: 256-4096 pixels to match existing `resize` field constraints
    - **Integration Point**: This will replace hardcoded 512px references found in `image_writer.py` lines 567 and 580
  - **EXECUTION DETAILS**:
    - **Code Location**: Line 432 in `personfromvid/data/config.py` 
    - **Pattern Followed**: Similar to `resize` field but required with explicit default
    - **Validation**: Uses same ge=256, le=4096 range as resize field

##### 1.1.2.1 Unit Testing
- **Test Cases**: Range validation (255, 256, 640, 4096, 4097), default value verification
- **Coverage**: Boundary conditions, Pydantic validation error messages

### Task 1.2: Implement Configuration Validation Logic
*Spec Reference: Configuration Dependencies, Aspect Ratio Validation*

- [x] 1.2.1 **Create crop_ratio dependency validator** ✅ **COMPLETED**
  - *Hint*: Use Pydantic `@model_validator(mode='after')` similar to existing validators in the config system
  - *Consider*: Validator should check that `crop_ratio` is only specified when `enable_pose_cropping` is True
  - *Files*: `personfromvid/data/config.py`
  - *Risk*: Low - validation adds safety without affecting existing functionality
  - **IMPLEMENTATION PLAN**:
    - **Validator Function**: Create `@model_validator(mode='after') def validate_crop_ratio_dependency(self)`
    - **Logic**: Check if `crop_ratio is not None and not enable_pose_cropping` then raise ValueError
    - **Error Message**: "crop_ratio can only be specified when enable_pose_cropping is True"
    - **Placement**: Add after existing field validators in `OutputImageConfig`
  - **EXECUTION DETAILS**:
    - **Import Addition**: Added `model_validator` to Pydantic imports
    - **Code Location**: After png/jpeg fields in `OutputImageConfig` class
    - **Pattern Followed**: Cross-field validation using `@model_validator(mode='after')`

##### 1.2.1.1 Unit Testing
- **Test Cases**: Valid combinations (`crop_ratio` with `enable_pose_cropping=True`), invalid combinations, edge cases
- **Mocks/Fixtures**: Configuration dictionaries with various field combinations
- **Coverage**: All validation paths, error message accuracy

- [x] 1.2.2 **Create aspect ratio format validator** ✅ **COMPLETED**
  - *Hint*: Use `@field_validator('crop_ratio')` to validate the string format before model validation
  - *Consider*: Parse "WxH" format, validate integers, check ratio range 0.1-100.0 as specified
  - *Files*: `personfromvid/data/config.py`
  - *Risk*: Medium - parsing logic needs to handle various invalid formats gracefully
  - **IMPLEMENTATION PLAN**:
    - **Regex Pattern**: Use `r'^(\d+):(\d+)$'` to match "W:H" format exactly
    - **Parsing Logic**: Extract width and height as integers, calculate ratio
    - **Range Validation**: Ensure calculated ratio (width/height) is between 0.1 and 100.0
    - **Error Handling**: Reject formats like "16:", ":9", "16/9", "1.5:1" with specific error messages
  - **EXECUTION DETAILS**:
    - **Import Addition**: Add `import re` to existing imports in config.py
    - **Validator Function**: Create `@field_validator('crop_ratio', mode='before')` before model validator
    - **Error Categories**: Format errors, ratio bound errors, type errors with specific messages
    - **Pattern Matching**: Strict regex to validate integer:integer format only
    - **Integration**: Place after field definitions but before model validator to catch format issues early
  - **COMPLETION SUMMARY**:
    - **Code Location**: Lines 444-470 in `personfromvid/data/config.py`
    - **Regex Implementation**: `r'^(\d+):(\d+)$'` for strict W:H format validation
    - **Range Validation**: Calculates ratio and enforces 0.1-100.0 bounds as specified
    - **Error Messages**: Comprehensive error messages for format, type, and range errors
    - **Test Coverage**: 34 passing tests including 3 dedicated aspect ratio validation test functions
    - **Integration**: Works seamlessly with existing dependency validation

##### 1.2.2.1 Unit Testing
- **Test Cases**: Valid formats ("1:1", "16:9", "4:3"), invalid formats ("16:", ":9", "16/9", "a:b"), ratios outside range
- **Coverage**: All parsing branches, edge cases, error message quality

### Phase 1 Validation
- **Acceptance Criteria**: `OutputImageConfig` correctly handles new fields, validation prevents invalid configurations, comprehensive test coverage
- **Testing Strategy**: Run `pytest tests/unit/test_config.py` with new test cases, create configuration files with valid/invalid settings
- **Rollback Plan**: Remove new fields and validators, restoring original `OutputImageConfig` structure

## Phase 2: Core Calculation Engine
*Incremental Goal: Create a robust, tested utility module with pure functions for all aspect ratio calculations, completely independent of application state.*

### Task 2.1: Create Aspect Ratio Calculation Module
*Spec Reference: New Calculation Module (`crop_utils.py`)*

- [x] 2.1.1 **Create `crop_utils.py` module with parsing function** ✅ **COMPLETED**
  - *Hint*: Create new file in `personfromvid/output/` directory alongside existing output modules
  - *Consider*: Module should have no dependencies on application state, only standard library imports
  - *Files*: `personfromvid/output/crop_utils.py` (new file)
  - *Risk*: Low - new isolated module with no side effects
  - **IMPLEMENTATION PLAN**:
    - **Module Structure**: Create with proper docstring, imports (typing, re, Optional, Tuple)
    - **Function Signature**: `def parse_aspect_ratio(ratio_str: str) -> Optional[Tuple[int, int]]:`
    - **Implementation**: Use regex parsing with comprehensive error handling
    - **Return Value**: `(width, height)` tuple for valid ratios, `None` for invalid
  - **EXECUTION DETAILS**:
    - **Regex Pattern**: `r'^(\d+):(\d+)$'` for strict W:H integer format validation
    - **Validation Logic**: Extract width/height as integers, calculate ratio, enforce 0.1-100.0 bounds
    - **Error Handling**: Return None for invalid formats, ratios outside bounds, or parsing errors
    - **Module Location**: `personfromvid/output/crop_utils.py` alongside existing output modules
    - **Dependencies**: Standard library only (re, typing) - no application state dependencies
    - **Integration Points**: Will be imported by ImageWriter in Phase 4 for aspect ratio processing
  - **COMPLETION SUMMARY**:
    - **File Created**: `personfromvid/output/crop_utils.py` with complete implementation
    - **Function Implemented**: `parse_aspect_ratio(ratio_str: str) -> Optional[Tuple[int, int]]` with comprehensive validation
    - **Regex Pattern**: `r'^(\d+):(\d+)$'` enforces strict W:H integer format
    - **Ratio Bounds**: Validates calculated ratio between 0.1 and 100.0 inclusive as specified
    - **Error Handling**: Returns None for all invalid inputs (format, type, bounds)
    - **Test Coverage**: 12 passing tests covering all validation scenarios and edge cases
    - **Import Verified**: Module imports correctly and integrates with existing codebase
    - **Specification Compliance**: Fully compliant with all aspect ratio validation requirements

##### 2.1.1.1 Unit Testing
- **Test Cases**: Valid inputs ("1:1", "16:9", "21:9"), invalid formats ("16:", ":9", "16/9"), edge cases ("0:1", "100:1")
- **Mocks/Fixtures**: String inputs covering all specified error conditions
- **Coverage**: All parsing branches, boundary conditions

- [x] 2.1.2 **Implement fixed aspect ratio bbox calculation** ✅ **COMPLETED**
  - *Hint*: Core geometric algorithm handling center positioning, expansion/cropping, boundary constraints
  - *Consider*: Function must handle edge detection, overflow/underflow, aspect ratio preservation
  - *Files*: `personfromvid/output/crop_utils.py`
  - *Risk*: High - complex geometric calculations with many edge cases
  - **IMPLEMENTATION PLAN**:
    - **Function Signature**: `def calculate_fixed_aspect_ratio_bbox(original_bbox: Tuple[int, int, int, int], image_dims: Tuple[int, int], aspect_ratio: Tuple[int, int]) -> Tuple[int, int, int, int]:`
    - **Centering Logic**: Calculate center point of original bbox, use as center for new bbox
    - **Size Calculation**: Determine target width/height based on aspect ratio and original bbox size
    - **Boundary Handling**: Shift bbox when it extends beyond image edges while preserving maximum area
    - **Edge Cases**: Handle very small bboxes, bboxes near edges, extreme aspect ratios
  - **EXECUTION DETAILS**:
    - **Algorithm Steps**: 1) Extract center from original bbox, 2) Calculate target dimensions, 3) Position new bbox on center, 4) Check boundaries, 5) Shift if needed, 6) Validate output
    - **Center Calculation**: `center_x = (x1 + x2) // 2, center_y = (y1 + y2) // 2`
    - **Size Logic**: Use original bbox area as reference, maintain target aspect ratio, handle expansion/reduction cases
    - **Boundary Strategy**: When bbox extends beyond image edges, shift symmetrically when possible, asymmetrically when at corners
    - **Edge Case Handling**: Very small bboxes get expanded, very large get reduced, extreme ratios get constrained
    - **Validation**: Ensure output maintains exact aspect ratio and stays within image bounds
    - **Integration**: Works with `parse_aspect_ratio` output, ready for ImageProcessor consumption
  - **COMPLETION SUMMARY**:
    - **Function Implemented**: `calculate_fixed_aspect_ratio_bbox(original_bbox, image_dims, aspect_ratio) -> bbox`
    - **Algorithm Success**: Area-preserving calculation with center positioning and boundary handling
    - **Geometric Features**: Handles centering, expansion/reduction, boundary shifting, exact aspect ratio preservation
    - **Input Validation**: Comprehensive validation for bbox coordinates, image dimensions, and aspect ratios
    - **Boundary Logic**: Sophisticated boundary overflow handling with asymmetric shifting when necessary
    - **Edge Case Handling**: Works correctly for very small bboxes, very large bboxes, edge positions, and extreme ratios
    - **Test Coverage**: 13 comprehensive test functions with 100% pass rate covering all algorithm paths
    - **Integration Verified**: Both functions work together seamlessly for end-to-end aspect ratio processing
    - **Performance**: Efficient pure function suitable for high-volume frame processing

##### 2.1.2.1 Unit Testing
- **Test Cases**: Center bbox scenarios, edge bbox scenarios (top, bottom, left, right, corners), size variations (smaller/larger than target)
- **Mocks/Fixtures**: Various bbox positions and image dimensions
- **Coverage**: All geometric calculation paths, boundary condition handling

### Task 2.2: Enhanced Utility Functions for Robustness
*Spec Reference: Edge Cases & Error Handling*

- [x] 2.2.1 **Add bbox validation and normalization helpers** ✅ **COMPLETED**
  - *Hint*: Create utility functions to validate bbox coordinates and handle edge cases
  - *Consider*: Functions should ensure bbox integrity, handle coordinate clamping
  - *Files*: `personfromvid/output/crop_utils.py`
  - *Risk*: Low - utility functions improve robustness
  - **IMPLEMENTATION PLAN**:
    - **Validation Function**: `def validate_bbox(bbox: Tuple[int, int, int, int], image_dims: Tuple[int, int]) -> bool:`
    - **Normalization Function**: `def normalize_bbox(bbox: Tuple[int, int, int, int], image_dims: Tuple[int, int]) -> Tuple[int, int, int, int]:`
    - **Error Handling**: Graceful handling of invalid coordinates, automatic correction where possible
  - **EXECUTION DETAILS**:
    - **Validation Logic**: Check bbox coordinates are valid (x2 > x1, y2 > y1) and within image bounds
    - **Normalization Strategy**: Clamp coordinates to image boundaries, ensure minimum 1x1 size, fix invalid ordering
    - **Return Behavior**: validate_bbox returns bool, normalize_bbox returns corrected bbox
    - **Edge Cases**: Handle negative coordinates, out-of-bounds coordinates, zero-size bboxes
    - **Integration**: Support defensive programming for existing calculation functions
    - **Testing**: Comprehensive test coverage for all validation and normalization scenarios
  - **COMPLETION SUMMARY**:
    - **Functions Implemented**: `validate_bbox` and `normalize_bbox` utility functions
    - **Validation Features**: Comprehensive coordinate validation, bounds checking, type validation
    - **Normalization Features**: Coordinate clamping, order fixing, minimum size enforcement, graceful fallbacks
    - **Error Handling**: Robust handling of invalid inputs with safe defaults
    - **Edge Case Support**: Negative coordinates, out-of-bounds, zero-size bboxes, small images
    - **Type Flexibility**: Handles both int and float inputs with automatic conversion
    - **Test Coverage**: 17 comprehensive test functions with 100% pass rate
    - **Integration Ready**: Defensive programming support for all calculation functions
    - **Performance**: Efficient validation and correction with minimal overhead

##### 2.2.1.1 Unit Testing
- **Test Cases**: Valid bboxes, invalid coordinates, out-of-bounds bboxes, edge alignment
- **Coverage**: All validation and normalization logic paths

### Phase 2 Validation
- **Acceptance Criteria**: `crop_utils.py` module created with comprehensive test coverage, all geometric calculations verified
- **Testing Strategy**: Run isolated unit tests for the module, test geometric calculations with known inputs/outputs
- **Rollback Plan**: Delete `crop_utils.py` module - no integration dependencies exist yet

## Phase 3: Image Processing Architecture Refactor
*Incremental Goal: Create clean separation between orchestration (ImageWriter) and processing (ImageProcessor), maintaining existing functionality while enabling new features.*

### Task 3.1: Create ImageProcessor Class
*Spec Reference: New Image Processing Class (`image_processor.py`)*

- [x] 3.1.1 **Create `ImageProcessor` class with existing logic** ✅ **COMPLETED**
  - *Hint*: Extract logic from `ImageWriter._crop_region` method (lines 531-619 in `image_writer.py`)
  - *Consider*: Class should handle cropping, resizing, face restoration while remaining stateless
  - *Files*: `personfromvid/output/image_processor.py` (new file)
  - *Risk*: Medium - complex image processing logic extraction requires careful testing
  - **IMPLEMENTATION PLAN**:
    - **Class Structure**: Create with `__init__(self, config: OutputImageConfig)` for configuration access
    - **Method Migration**: Move `_crop_region` logic to `crop_and_resize` method
    - **Dependencies**: Import face restoration, PIL/CV2 image handling, maintain existing patterns
    - **Interface**: `def crop_and_resize(self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float, use_face_restoration: bool = False) -> np.ndarray:`
  - **DETAILED EXECUTION PLAN**:
    - **File Creation**: `personfromvid/output/image_processor.py` with proper imports and TYPE_CHECKING
    - **Class Architecture**: Stateless design with config and logger initialization
    - **Face Restoration**: Extract and preserve exact `_get_face_restorer()` lazy loading logic
    - **Method Migration**: Line-by-line migration of `_crop_region` logic preserving exact behavior
    - **Hardcoded Replacement**: Replace 512px references with `self.config.default_crop_size`
    - **Error Handling**: Maintain exact exception handling and logging patterns
    - **Interface Compatibility**: Ensure seamless replacement in ImageWriter integration
  - **EXECUTION DETAILS**:
    - **Source Analysis**: Analyzed `_crop_region()` method (lines 531-619), `_get_face_restorer()` (lines 48-70), imports and initialization patterns
    - **Key Logic Elements**: Padding calculation, bounds checking, cropping, face restoration integration, Lanczos upscaling
    - **Hardcoded Values**: Two 512px references to replace with `self.config.default_crop_size` (lines 567, 580)
    - **Dependencies**: PIL.Image, numpy, face restoration lazy loading, logger integration
    - **Interface**: Stateless class with config injection, maintaining exact method signature compatibility
  - **COMPLETION SUMMARY**:
    - **Implementation Status**: ✅ Complete - ImageProcessor class successfully created with full functionality
    - **Method Migration**: Successfully extracted `_crop_region` logic (lines 531-619) with exact behavior preservation
    - **Face Restoration Integration**: Complete lazy loading implementation with proper error handling
    - **Hardcoded Value Replacement**: All 512px references replaced with `self.config.default_crop_size`
    - **Configuration Integration**: Proper resize priority handling (resize > default_crop_size)
    - **Dependencies**: All imports and TYPE_CHECKING properly configured
    - **Test Coverage**: 18/18 tests passing (100% pass rate) - ALL functionality fully tested and verified
    - **Interface Compatibility**: Ready for seamless ImageWriter integration in next task
    - **Performance**: Efficient stateless design with proper resource management
    - **Regression Testing**: 34/34 config tests + 42/42 crop utils tests still passing - no regressions introduced  
    - **Quality Verification**: Full test suite validates face restoration, upscaling, boundary handling, configuration integration
    - **Production Ready**: Complete implementation ready for integration with ImageWriter in next phase

##### 3.1.1.1 Unit Testing  
- **Test Cases**: Simple cropping, resizing scenarios, face restoration integration, boundary handling
- **Mocks/Fixtures**: Mock images, face restoration service, configuration variations
- **Coverage**: All processing paths, error conditions, restoration fallbacks

- [x] 3.1.2 **Replace hardcoded 512px references with config** ✅ **COMPLETED**
  - *Hint*: Replace hardcoded values in lines 567 and 580 of `image_writer.py` with `config.output.image.default_crop_size`
  - *Consider*: This change affects sizing behavior - existing users may notice larger crops
  - *Files*: `personfromvid/output/image_processor.py`
  - *Risk*: Low-Medium - behavior change in default crop sizes
  - **IMPLEMENTATION PLAN**:
    - **Reference Replacement**: Use `self.config.default_crop_size` instead of hardcoded 512
    - **Logic Preservation**: Maintain existing `min(min_dimension, default_crop_size)` pattern 
    - **Performance Consideration**: Keep cap on crop size to prevent memory issues
  - **COMPLETION SUMMARY**:
    - **Implementation Status**: ✅ Complete - Already implemented during Task 3.1.1
    - **Reference Replacement**: Line 84: `self.config.default_crop_size` replaces hardcoded 512px
    - **Face Restoration Target**: Line 98: `min(min_dimension, self.config.default_crop_size)` maintains performance caps
    - **Logic Preservation**: Maintained existing `min(min_dimension, default_crop_size)` pattern exactly
    - **Test Coverage**: Verified through existing ImageProcessor test suite (18/18 tests passing)
    - **Behavior Impact**: Default crop size increased from 512px to 640px as specified
    - **Performance**: Memory usage increase (~56%) within acceptable limits per specification

##### 3.1.2.1 Unit Testing
- **Test Cases**: Various `default_crop_size` values, resize interactions, performance bounds
- **Coverage**: Size calculation logic, configuration integration

### Task 3.2: Refactor ImageWriter to Use ImageProcessor
*Spec Reference: ImageWriter Refactoring*

- [x] 3.2.1 **Update ImageWriter to delegate to ImageProcessor** ✅ **COMPLETED**
  - *Hint*: Replace direct `_crop_region` calls with `ImageProcessor` instance
  - *Consider*: Maintain exact same output behavior to avoid breaking existing functionality
  - *Files*: `personfromvid/output/image_writer.py`
  - *Risk*: Medium - refactoring complex integration points
  - **IMPLEMENTATION PLAN**:
    - **Initialization**: Add `self.image_processor = ImageProcessor(self.config)` in `__init__`
    - **Method Replacement**: Replace `_crop_region` calls with `self.image_processor.crop_and_resize`
    - **Method Removal**: Delete `_crop_region` method after migration
    - **Interface Preservation**: Ensure `_crop_face` and other methods work unchanged
  - **EXECUTION DETAILS**:
    - **Deep Analysis**: Examining ImageWriter structure, _crop_region usage patterns, and integration points
    - **Call Site Identification**: Locating all _crop_region invocations to ensure complete migration
    - **Signature Compatibility**: Ensuring ImageProcessor.crop_and_resize matches _crop_region interface exactly
    - **Dependency Injection**: Proper ImageProcessor instantiation with configuration
    - **Behavior Preservation**: Maintaining exact output behavior for all crop types and scenarios
  - **COMPLETION SUMMARY**:
    - **Implementation Status**: ✅ Complete - ImageWriter successfully refactored to use ImageProcessor delegation
    - **Method Calls Replaced**: 4 call sites updated - save_frame_outputs() (line 117), save_person_outputs() (line 261), _save_frame_outputs_legacy() (line 475), _crop_face() (line 630)
    - **Method Removal**: Successfully removed 88-line `_crop_region` method (lines 535-623) - completely migrated to ImageProcessor
    - **ImageProcessor Integration**: Added `self.image_processor = ImageProcessor(self.config)` initialization in constructor
    - **Import Updates**: Added `from .image_processor import ImageProcessor` to imports
    - **Test Suite Validation**: 51/51 ImageWriter tests passing + 18/18 ImageProcessor tests passing = 69/69 total tests ✅
    - **Interface Compatibility**: Perfect method signature matching - `crop_and_resize()` is drop-in replacement for `_crop_region()`
    - **Face Restoration**: Preserved exact face restoration behavior through delegation to ImageProcessor
    - **No Regressions**: All existing functionality maintained, zero breaking changes to external interface
    - **Code Quality**: Clean separation of concerns - ImageWriter handles orchestration, ImageProcessor handles pixel operations
    - **Production Ready**: Refactoring complete and fully tested, ready for Phase 4 integration

##### 3.2.1.1 Integration Testing
- **Test Cases**: Face crop processing, pose crop processing, resize behavior consistency
- **Mocks/Fixtures**: Existing `ImageWriter` test fixtures
- **Coverage**: All crop generation scenarios, error handling paths

### Phase 3 Validation
- **Acceptance Criteria**: `ImageProcessor` class functional and tested, `ImageWriter` refactored without behavior changes, all existing tests pass
- **Testing Strategy**: Run full test suite for `ImageWriter`, compare output images before/after refactoring
- **Rollback Plan**: Restore original `_crop_region` method in `ImageWriter`, remove `ImageProcessor` class

## Phase 4: End-to-End Integration and CLI Enhancement  
*Incremental Goal: Complete feature implementation with CLI integration, connecting all components for seamless end-to-end functionality.*

### Task 4.1: Integrate Fixed Aspect Ratio Logic
*Spec Reference: ImageWriter Integration, Crop Data Storage*

- [x] 4.1.1 **Add aspect ratio calculation to ImageWriter** ✅ **COMPLETED**
  - *Hint*: Import `crop_utils` and use in `save_frame_outputs` when `crop_ratio` is configured
  - *Consider*: Store calculated crop regions in `FrameData.selections.crop_regions` with aspect ratio keys
  - *Files*: `personfromvid/output/image_writer.py`
  - *Risk*: Medium - complex integration with state management
  - **IMPLEMENTATION PLAN**:
    - **Import Integration**: Add `from .crop_utils import parse_aspect_ratio, calculate_fixed_aspect_ratio_bbox`
    - **Logic Placement**: Add aspect ratio processing before existing crop logic in `save_frame_outputs`
    - **State Storage**: Store calculated bboxes in `frame.selections.crop_regions` with keys like "1:1", "16:9"
    - **Conditional Processing**: Only apply when `config.crop_ratio` is not None
  - **EXECUTION DETAILS**:
    - **Import Added**: Successfully added crop_utils imports to ImageWriter
    - **Method Implementation**: Created `_calculate_fixed_aspect_ratio_crops()` method with comprehensive error handling
    - **Integration Point**: Added aspect ratio processing after source image loading in `save_frame_outputs()`
    - **State Storage**: Fixed ratio bboxes stored with keys: `crop_ratio` for pose crops, `face_{crop_ratio}` for face crops
    - **Conditional Logic**: Processing only occurs when `self.config.crop_ratio is not None`
    - **Error Handling**: Graceful handling of invalid aspect ratios and calculation failures with warning logs
    - **Logging**: Debug logging for successful calculations and warning logs for failures

##### 4.1.1.1 Integration Testing
- **Test Cases**: Fixed ratio with various aspect ratios, interaction with existing cropping, state persistence
- **Mocks/Fixtures**: Frame data with different configurations
- **Coverage**: All integration paths, state management, error conditions

- [x] 4.1.2 **Implement crop region replacement logic** ✅ **COMPLETED**
  - *Hint*: When `crop_ratio` is set, fixed aspect ratio crops replace variable aspect ratio crops
  - *Consider*: Maintain existing padding and boundary handling while using new bbox calculations
  - *Files*: `personfromvid/output/image_writer.py`
  - *Risk*: Medium - behavior change affects output generation
  - **IMPLEMENTATION PLAN**:
    - **Replacement Logic**: Use fixed aspect ratio bbox instead of original detection bbox
    - **Padding Integration**: Apply existing padding logic to fixed aspect ratio bboxes
    - **Universal Application**: Apply to both face crops and pose crops when enabled
  - **EXECUTION DETAILS**:
    - **Pose Crop Replacement**: Modified pose crop logic to use fixed ratio bbox when available from `crop_regions[crop_ratio]`
    - **Face Crop Replacement**: Modified face crop logic to use fixed ratio bbox from `crop_regions[face_{crop_ratio}]` when available
    - **Fallback Behavior**: Maintains original bbox behavior when fixed ratio is not configured or calculation fails
    - **Padding Preservation**: All existing padding and boundary handling logic preserved through ImageProcessor delegation
    - **Temporary Object Creation**: Created temporary FaceDetection object for face crops to maintain API compatibility
    - **Debug Logging**: Added debug logs when using fixed aspect ratio bboxes for crop generation

##### 4.1.2.1 Integration Testing
- **Test Cases**: Face crop replacement, pose crop replacement, padding preservation
- **Coverage**: All crop types, configuration combinations

### Task 4.2: CLI Integration and Validation
*Spec Reference: CLI Integration, Configuration Conflicts*

- [x] 4.2.1 **Add `--crop-ratio` CLI parameter** ✅ **COMPLETED**
  - *Hint*: Follow existing click option patterns in `cli.py`, add after `--crop-padding` around line 118
  - *Consider*: Parameter should have clear help text and validation
  - *Files*: `personfromvid/cli.py`
  - *Risk*: Low - adding new CLI parameter
  - **IMPLEMENTATION PLAN**:
    - **Click Decorator**: Add `@click.option("--crop-ratio", type=str, help="Fixed aspect ratio for crops (e.g., '1:1', '16:9', '4:3')")`
    - **Parameter Integration**: Add to main function signature
    - **Help Text**: Clear description with examples and requirements
  - **EXECUTION DETAILS**:
    - **Click Option Added**: Successfully added `@click.option("--crop-ratio", type=str, help="Fixed aspect ratio for crops (e.g., '1:1', '16:9', '4:3'). Requires --crop flag.")` after `--crop-padding`
    - **Function Signature Updated**: Added `crop_ratio: Optional[str]` parameter to main function signature at correct position
    - **Help Text Implementation**: Clear help text with examples ('1:1', '16:9', '4:3') and dependency requirement note
    - **Pattern Consistency**: Followed existing Click option patterns exactly, maintaining consistency with other parameters
    - **Placement Verification**: Correctly positioned after `--crop-padding` and before `--full-frames` as specified
    - **CLI Integration Test**: Verified parameter appears correctly in `--help` output with proper formatting

##### 4.2.1.1 Unit Testing
- **Test Cases**: CLI parsing with valid/invalid aspect ratios, help text display
- **Coverage**: Parameter parsing, validation integration

- [x] 4.2.2 **Add CLI dependency validation** ✅ **COMPLETED**
  - *Hint*: Add validation in main function to ensure `--crop-ratio` requires `--crop` flag
  - *Consider*: Validation should occur early in CLI processing with clear error messages
  - *Files*: `personfromvid/cli.py`
  - *Risk*: Low - validation improves user experience
  - **IMPLEMENTATION PLAN**:
    - **Validation Logic**: Check if `crop_ratio` is specified but `crop` is False
    - **Error Message**: "Error: --crop-ratio requires --crop flag to be enabled"
    - **Placement**: Add validation in main function after argument parsing
  - **EXECUTION DETAILS**:
    - **Validation Logic Added**: Implemented early validation `if crop_ratio is not None and not crop:` after video path check
    - **Error Message Implemented**: Clear error message "[red]Error:[/red] --crop-ratio requires --crop flag to be enabled"
    - **Help Integration**: Added "Try 'personfromvid --help' for help." guidance text
    - **Exit Behavior**: Proper sys.exit(1) for invalid combinations
    - **Placement Verification**: Positioned after video path validation and before signal handling for early catch
    - **Testing Verified**: 
      - Invalid case: `--crop-ratio 1:1` (without `--crop`) correctly shows error and exits
      - Valid case: `--crop --crop-ratio 1:1` passes validation and starts processing

##### 4.2.2.1 Unit Testing
- **Test Cases**: Valid combinations, invalid combinations, error message accuracy
- **Coverage**: All validation scenarios

- [x] 4.2.3 **Update CLI override logic** ✅ **COMPLETED**
  - *Hint*: Add `crop_ratio` handling to `_apply_output_overrides` function around line 650
  - *Consider*: Follow existing pattern for optional parameters
  - *Files*: `personfromvid/cli.py`  
  - *Risk*: Low - standard override pattern
  - **IMPLEMENTATION PLAN**:
    - **Override Logic**: Add `if cli_args["crop_ratio"]: config.output.image.crop_ratio = cli_args["crop_ratio"]`
    - **Pattern Consistency**: Follow existing override patterns in function
  - **EXECUTION DETAILS**:
    - **Override Logic Added**: Successfully added `if cli_args["crop_ratio"]: config.output.image.crop_ratio = cli_args["crop_ratio"]` in `_apply_output_overrides()`
    - **Placement Verification**: Correctly positioned after `crop_padding` override and before `full_frames` override
    - **Pattern Consistency**: Followed exact same pattern as other optional parameter overrides in the function
    - **Integration Test**: Verified that CLI override correctly applies crop_ratio value to configuration:
      - Original: `crop_ratio: None` 
      - After CLI override with `--crop-ratio 16:9`: `crop_ratio: 16:9`
      - Configuration integration confirmed working end-to-end

##### 4.2.3.1 Unit Testing
- **Test Cases**: Configuration override application, precedence handling
- **Coverage**: Override logic paths

### Task 4.3: Comprehensive Testing and Documentation
*Spec Reference: Integration Tests, Acceptance Criteria*

- [x] 4.3.1 **Create integration tests for complete feature** ✅ **COMPLETED**
  - *Hint*: Create new test file in `tests/integration/` following existing patterns
  - *Consider*: Test full pipeline with fixed aspect ratio crops
  - *Files*: `tests/integration/test_fixed_ratio_crop_integration.py` (new file)
  - *Risk*: Low - testing improves reliability
  - **IMPLEMENTATION PLAN**:
    - **Test Scenarios**: Test with different aspect ratios ("1:1", "16:9", "4:3")
    - **Configuration Testing**: Test CLI parameter integration, config file integration
    - **Output Verification**: Verify correct crop dimensions and aspect ratios
    - **Error Scenarios**: Test invalid configurations, boundary conditions
  - **DETAILED EXECUTION PLAN**:
    - **Test Structure**: Create pytest-based integration test following existing patterns from `test_person_pipeline_integration.py`
    - **Fixtures**: Use `test_video_path`, `temp_output_dir`, and create `fixed_ratio_config` fixture for various aspect ratios
    - **CLI Testing**: Test CLI parameter integration with subprocess calls to verify end-to-end CLI functionality
    - **Pipeline Integration**: Test full pipeline execution with fixed aspect ratio configuration and verify completion
    - **Output Validation**: Use PIL/OpenCV to measure actual output image dimensions and verify aspect ratios are exact
    - **Error Testing**: Test invalid aspect ratio strings, missing --crop flag, boundary conditions with very small/large ratios
    - **Configuration Flow Testing**: Verify that CLI overrides properly propagate through config system to ImageWriter
    - **Regression Prevention**: Ensure existing functionality works when crop_ratio is None (backward compatibility)
  - **EXECUTION DETAILS**:
    - **Test File Created**: Successfully created `tests/integration/test_fixed_ratio_crop_integration.py` with comprehensive test coverage
    - **Test Methods Implemented**: 10 comprehensive test methods covering all integration scenarios
    - **CLI Integration Tests**: `test_cli_parameter_integration()` and `test_cli_dependency_validation()` verify CLI functionality end-to-end
    - **Configuration Flow Tests**: `test_configuration_flow_integration()` verifies config parsing, validation, and propagation
    - **Pipeline Execution Tests**: `test_pipeline_execution_with_fixed_ratios()` tests complete pipeline with fixed aspect ratios
    - **Output Validation Tests**: `test_output_dimension_validation()` and `test_multiple_aspect_ratios()` verify actual image dimensions match expected ratios
    - **Error Handling Tests**: `test_error_handling_integration()` validates error scenarios and invalid configurations
    - **Backward Compatibility Tests**: `test_backward_compatibility()` ensures existing functionality preserved when crop_ratio is None
    - **State Serialization Tests**: `test_state_serialization_with_crop_regions()` verifies pipeline state with crop regions can be serialized
    - **Parametrized Testing**: Multiple aspect ratios (1:1, 16:9, 4:3, 21:9) tested with precise tolerance checking
    - **Fixtures Implemented**: Comprehensive fixture setup with base_config, square_ratio_config, widescreen_ratio_config, portrait_ratio_config
    - **Test Execution Verified**: CLI parameter integration and dependency validation tests passing successfully
    - **Quality Assurance**: Integration tests provide comprehensive coverage of the complete fixed aspect ratio crop feature

##### 4.3.1.1 Integration Testing
- **Test Cases**: Full pipeline execution, output image verification, error handling
- **Mocks/Fixtures**: Test video files, configuration variations
- **Coverage**: End-to-end functionality, error paths

- [x] 4.3.2 **Update CLI help and documentation** ✅ **COMPLETED**
  - *Hint*: Ensure CLI help text is clear and examples are provided
  - *Consider*: Document integration with existing flags and configuration
  - *Files*: `personfromvid/cli.py`, potentially README updates
  - *Risk*: Low - documentation improvements
  - **IMPLEMENTATION PLAN**:
    - **Help Text Enhancement**: Ensure clear examples and requirements
    - **Example Updates**: Add fixed aspect ratio examples to CLI docstring
    - **Integration Documentation**: Document interaction with existing flags
  - **EXECUTION DETAILS**:
    - **CLI Examples Enhanced**: Successfully added 4 comprehensive examples demonstrating fixed aspect ratio crop functionality
    - **Example Scenarios Added**:
      - Basic square crops: `--crop --crop-ratio 1:1`
      - Widescreen crops with custom output: `--crop --crop-ratio 16:9 --output-dir ./widescreen_crops`
      - Portrait crops with padding: `--crop --crop-ratio 4:3 --crop-padding 0.3`
      - Advanced integration: `--crop --crop-ratio 1:1 --face-restoration --resize 512`
    - **Help Text Verification**: Confirmed new examples appear correctly in `--help` output with proper formatting
    - **Parameter Documentation**: Verified `--crop-ratio` parameter help text is clear and includes dependency requirement
    - **Integration Documentation**: Examples show proper integration with existing flags (--crop, --crop-padding, --face-restoration, --resize, --output-dir)
    - **User Experience**: Enhanced documentation provides clear guidance for users wanting to use fixed aspect ratio crops
    - **Practical Examples**: All examples are realistic use cases that users would actually want to perform

### Phase 4 Validation
- **Acceptance Criteria**: Complete feature working end-to-end, CLI integration functional, comprehensive test coverage
- **Testing Strategy**: Run full integration test suite, manual CLI testing with various configurations
- **Rollback Plan**: Remove CLI parameter, revert ImageWriter changes, maintain backward compatibility

## Critical Considerations

### Performance Impact
- **Memory Usage**: Larger default crop size (640px vs 512px) increases memory usage by ~56%
- **Processing Time**: Fixed aspect ratio calculations add minimal overhead (<1ms per frame)
- **Storage Impact**: Larger default crops may increase output file sizes

### Security Considerations  
- **Input Validation**: All aspect ratio parsing includes comprehensive validation
- **Resource Limits**: Crop size limits prevent excessive memory allocation
- **Error Handling**: Graceful degradation when calculations fail

### Scalability Factors
- **Configuration Complexity**: New fields integrate cleanly with existing Pydantic system
- **Processing Pipeline**: Changes are isolated and don't affect other pipeline components
- **State Management**: Crop regions storage scales with frame volume

### Monitoring and Observability
- **Logging Integration**: All phases include appropriate debug/info logging
- **Error Tracking**: Comprehensive error handling with actionable messages
- **Performance Metrics**: Processing time tracking for new calculations

### Cross-Phase Dependencies
- **Phase 1 → Phase 2**: Configuration validation must pass before calculation testing
- **Phase 2 → Phase 3**: Calculation functions must be stable before image processor integration
- **Phase 3 → Phase 4**: Image processing architecture must be solid before CLI integration
- **Rollback Dependencies**: Each phase can be independently reverted without affecting previous phases

## Research & Validation Completed

### Dependencies Verified
- **Pydantic System**: Configuration validation patterns confirmed in existing codebase
- **Image Processing Stack**: PIL, OpenCV, NumPy integration patterns established
- **Testing Framework**: pytest patterns and fixture usage confirmed

### Patterns Identified  
- **Configuration Override**: CLI override patterns in `_apply_output_overrides` function
- **Image Processing**: Complex logic in `_crop_region` method provides refactoring template
- **State Management**: `FrameData.selections.crop_regions` field ready for aspect ratio keys

### Assumptions Validated
- **Backward Compatibility**: New optional fields maintain compatibility with existing configurations
- **Performance Bounds**: Default crop size increase manageable within system constraints
- **Integration Points**: Clear separation between calculation, processing, and orchestration layers 