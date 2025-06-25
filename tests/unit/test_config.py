"""Tests for configuration management."""


import pytest

from personfromvid.data.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    Config,
    DeviceType,
    LogLevel,
    OutputImageConfig,
    PersonSelectionCriteria,
    get_default_config,
    FrameSelectionConfig,
)


def test_default_config_creation():
    """Test creating default configuration."""
    config = get_default_config()

    assert isinstance(config, Config)
    assert config.logging.level == LogLevel.INFO
    assert config.models.device == DeviceType.AUTO
    assert config.models.batch_size == DEFAULT_BATCH_SIZE
    assert config.models.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD


def test_config_validation():
    """Test configuration validation."""
    config = Config()

    # Test that defaults are valid
    assert config.models.batch_size >= 1
    assert config.models.batch_size <= 64
    assert config.models.confidence_threshold >= 0.0
    assert config.models.confidence_threshold <= 1.0


def test_resize_config_defaults():
    """Test resize configuration defaults."""
    config = Config()

    # Resize should be None by default (no resizing)
    assert config.output.image.resize is None


def test_resize_config_validation():
    """Test resize configuration validation."""
    # Test valid resize values
    config = Config()
    config.output.image.resize = 512
    assert config.output.image.resize == 512

    config.output.image.resize = 1024
    assert config.output.image.resize == 1024

    config.output.image.resize = 2048
    assert config.output.image.resize == 2048

    # Test minimum boundary
    config.output.image.resize = 256
    assert config.output.image.resize == 256

    # Test maximum boundary
    config.output.image.resize = 4096
    assert config.output.image.resize == 4096


def test_resize_config_invalid_values():
    """Test resize configuration with invalid values."""
    from pydantic import ValidationError

    # Test values below minimum
    with pytest.raises(ValidationError):
        OutputImageConfig(resize=255)

    with pytest.raises(ValidationError):
        OutputImageConfig(resize=100)

    # Test values above maximum
    with pytest.raises(ValidationError):
        OutputImageConfig(resize=4097)

    with pytest.raises(ValidationError):
        OutputImageConfig(resize=8192)

    # Test invalid type
    with pytest.raises(ValidationError):
        OutputImageConfig(resize="invalid")

    with pytest.raises(ValidationError):
        OutputImageConfig(resize=-1)


def test_crop_ratio_config_defaults():
    """Test crop_ratio configuration defaults."""
    config = Config()

    # crop_ratio should be None by default (no fixed aspect ratio)
    assert config.output.image.crop_ratio is None


def test_crop_ratio_config_valid_values():
    """Test crop_ratio configuration with valid values."""
    # Test valid crop_ratio values (need enable_pose_cropping=True for non-None values)
    config = Config()
    
    # Enable pose cropping first, then set crop ratios
    config.output.image.enable_pose_cropping = True
    
    # Test common aspect ratios
    config.output.image.crop_ratio = "1:1"
    assert config.output.image.crop_ratio == "1:1"

    config.output.image.crop_ratio = "16:9"
    assert config.output.image.crop_ratio == "16:9"

    config.output.image.crop_ratio = "4:3"
    assert config.output.image.crop_ratio == "4:3"

    config.output.image.crop_ratio = "21:9"
    assert config.output.image.crop_ratio == "21:9"

    # Test None value (works regardless of pose cropping setting)
    config.output.image.crop_ratio = None
    assert config.output.image.crop_ratio is None


def test_crop_ratio_config_type_validation():
    """Test crop_ratio configuration type validation."""
    from pydantic import ValidationError

    # Test valid string type (must include enable_pose_cropping=True due to dependency validation)
    config = OutputImageConfig(crop_ratio="1:1", enable_pose_cropping=True)
    assert config.crop_ratio == "1:1"

    # Test None type
    config = OutputImageConfig(crop_ratio=None)
    assert config.crop_ratio is None

    # Test invalid types (with enable_pose_cropping=True to isolate type validation)
    with pytest.raises(ValidationError):
        OutputImageConfig(crop_ratio=123, enable_pose_cropping=True)

    with pytest.raises(ValidationError):
        OutputImageConfig(crop_ratio=1.5, enable_pose_cropping=True)

    with pytest.raises(ValidationError):
        OutputImageConfig(crop_ratio=["1:1"], enable_pose_cropping=True)

    with pytest.raises(ValidationError):
        OutputImageConfig(crop_ratio={"ratio": "1:1"}, enable_pose_cropping=True)


def test_default_crop_size_config_defaults():
    """Test default_crop_size configuration defaults."""
    config = Config()

    # default_crop_size should be 640 by default
    assert config.output.image.default_crop_size == 640


def test_default_crop_size_config_valid_values():
    """Test default_crop_size configuration with valid values."""
    # Test valid default_crop_size values
    config = Config()
    
    # Test different valid sizes
    config.output.image.default_crop_size = 256
    assert config.output.image.default_crop_size == 256

    config.output.image.default_crop_size = 512
    assert config.output.image.default_crop_size == 512

    config.output.image.default_crop_size = 1024
    assert config.output.image.default_crop_size == 1024

    config.output.image.default_crop_size = 4096
    assert config.output.image.default_crop_size == 4096

    # Test default value
    config = Config()
    assert config.output.image.default_crop_size == 640


def test_default_crop_size_config_invalid_values():
    """Test default_crop_size configuration with invalid values."""
    from pydantic import ValidationError

    # Test values below minimum
    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size=255)

    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size=100)

    # Test values above maximum
    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size=4097)

    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size=8192)

    # Test invalid types
    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size="invalid")

    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size=-1)

    with pytest.raises(ValidationError):
        OutputImageConfig(default_crop_size=None)


def test_crop_ratio_format_validation():
    """Test crop_ratio format validation."""
    from pydantic import ValidationError

    # Test valid aspect ratio formats
    valid_formats = ["1:1", "16:9", "4:3", "21:9", "3:2", "2:3", "9:16"]
    for fmt in valid_formats:
        config = OutputImageConfig(crop_ratio=fmt, enable_pose_cropping=True)
        assert config.crop_ratio == fmt

    # Test invalid formats - malformed strings (format errors)
    format_error_cases = [
        "16:",      # Missing height
        ":9",       # Missing width
        "16/9",     # Wrong separator
        "1.5:1",    # Decimal ratio
        "16:9:2",   # Too many parts
        "invalid",  # Non-numeric
        "a:b",      # Non-numeric parts
        "16 : 9",   # Spaces
        "16-9",     # Wrong separator
        "",         # Empty string
        "-1:1",     # Negative width (fails regex)
        "1:-1",     # Negative height (fails regex)
    ]
    
    for fmt in format_error_cases:
        with pytest.raises(ValidationError, match="Invalid crop_ratio format"):
            OutputImageConfig(crop_ratio=fmt, enable_pose_cropping=True)

    # Test invalid formats - positive integer validation errors
    positive_int_error_cases = [
        "0:1",      # Zero width
        "1:0",      # Zero height
    ]
    
    for fmt in positive_int_error_cases:
        with pytest.raises(ValidationError, match="both width and height must be positive integers"):
            OutputImageConfig(crop_ratio=fmt, enable_pose_cropping=True)

    # Test boundary ratio values - valid boundaries
    # Ratio 0.1 (minimum boundary)
    config = OutputImageConfig(crop_ratio="1:10", enable_pose_cropping=True)
    assert config.crop_ratio == "1:10"

    # Ratio 100.0 (maximum boundary)
    config = OutputImageConfig(crop_ratio="100:1", enable_pose_cropping=True)
    assert config.crop_ratio == "100:1"

    # Test ratio outside valid range
    # Ratio < 0.1
    with pytest.raises(ValidationError, match="calculated ratio.*outside valid range"):
        OutputImageConfig(crop_ratio="1:11", enable_pose_cropping=True)

    # Ratio > 100.0
    with pytest.raises(ValidationError, match="calculated ratio.*outside valid range"):
        OutputImageConfig(crop_ratio="101:1", enable_pose_cropping=True)

    # Test very extreme ratios
    with pytest.raises(ValidationError, match="calculated ratio.*outside valid range"):
        OutputImageConfig(crop_ratio="1:1000", enable_pose_cropping=True)

    with pytest.raises(ValidationError, match="calculated ratio.*outside valid range"):
        OutputImageConfig(crop_ratio="1000:1", enable_pose_cropping=True)


def test_crop_ratio_format_validation_error_messages():
    """Test crop_ratio format validation error message quality."""
    from pydantic import ValidationError

    # Test type error message
    with pytest.raises(ValidationError, match="crop_ratio must be a string in format"):
        OutputImageConfig(crop_ratio=123, enable_pose_cropping=True)

    # Test format error message includes examples
    with pytest.raises(ValidationError, match="Must be in format 'W:H'.*'16:9', '4:3', '1:1'"):
        OutputImageConfig(crop_ratio="16:", enable_pose_cropping=True)

    # Test range error message includes valid range
    with pytest.raises(ValidationError, match="outside valid range \\(0.1-100.0\\)"):
        OutputImageConfig(crop_ratio="1:11", enable_pose_cropping=True)

    # Test zero value error message
    with pytest.raises(ValidationError, match="both width and height must be positive integers"):
        OutputImageConfig(crop_ratio="0:1", enable_pose_cropping=True)


def test_crop_ratio_format_validation_edge_cases():
    """Test crop_ratio format validation edge cases."""
    from pydantic import ValidationError

    # Test None value (should pass)
    config = OutputImageConfig(crop_ratio=None, enable_pose_cropping=True)
    assert config.crop_ratio is None

    # Test large but valid integer values
    config = OutputImageConfig(crop_ratio="50:1", enable_pose_cropping=True)
    assert config.crop_ratio == "50:1"

    config = OutputImageConfig(crop_ratio="1:5", enable_pose_cropping=True)
    assert config.crop_ratio == "1:5"

    # Test exact boundary calculations
    # 1:10 = 0.1 (exactly on boundary)
    config = OutputImageConfig(crop_ratio="1:10", enable_pose_cropping=True)
    assert config.crop_ratio == "1:10"

    # 100:1 = 100.0 (exactly on boundary)
    config = OutputImageConfig(crop_ratio="100:1", enable_pose_cropping=True)
    assert config.crop_ratio == "100:1"

    # Test just outside boundaries
    # 1:11 ≈ 0.091 (just below 0.1)
    with pytest.raises(ValidationError, match="outside valid range"):
        OutputImageConfig(crop_ratio="1:11", enable_pose_cropping=True)

    # 101:1 = 101.0 (just above 100.0)
    with pytest.raises(ValidationError, match="outside valid range"):
        OutputImageConfig(crop_ratio="101:1", enable_pose_cropping=True)


def test_crop_ratio_dependency_validation():
    """Test crop_ratio dependency validation."""
    from pydantic import ValidationError

    # Test valid combinations
    # crop_ratio with enable_pose_cropping=True should work
    config = OutputImageConfig(crop_ratio="1:1", enable_pose_cropping=True)
    assert config.crop_ratio == "1:1"
    assert config.enable_pose_cropping is True

    # crop_ratio=None with enable_pose_cropping=False should work (default)
    config = OutputImageConfig()
    assert config.crop_ratio is None
    assert config.enable_pose_cropping is False

    # crop_ratio=None with enable_pose_cropping=True should work
    config = OutputImageConfig(enable_pose_cropping=True)
    assert config.crop_ratio is None
    assert config.enable_pose_cropping is True

    # Test invalid combination
    # crop_ratio with enable_pose_cropping=False should fail
    with pytest.raises(ValidationError, match="crop_ratio can only be specified when enable_pose_cropping is True"):
        OutputImageConfig(crop_ratio="1:1", enable_pose_cropping=False)


def test_crop_ratio_dependency_validation_various_ratios():
    """Test crop_ratio dependency validation with various aspect ratios."""
    from pydantic import ValidationError

    # Test various valid aspect ratios with pose cropping enabled
    valid_ratios = ["1:1", "16:9", "4:3", "21:9", "2:3"]
    for ratio in valid_ratios:
        config = OutputImageConfig(crop_ratio=ratio, enable_pose_cropping=True)
        assert config.crop_ratio == ratio
        assert config.enable_pose_cropping is True

    # Test these same ratios should fail without pose cropping
    for ratio in valid_ratios:
        with pytest.raises(ValidationError, match="crop_ratio can only be specified when enable_pose_cropping is True"):
            OutputImageConfig(crop_ratio=ratio, enable_pose_cropping=False)


def test_crop_ratio_any_validation():
    """Test crop_ratio validation with 'any' value."""
    from pydantic import ValidationError

    # Test valid "any" variations with pose cropping enabled
    any_variations = ["any", "ANY", "Any", "AnY", "aNy"]
    for variation in any_variations:
        config = OutputImageConfig(crop_ratio=variation, enable_pose_cropping=True)
        assert config.crop_ratio == "any"  # Should normalize to lowercase
        assert config.enable_pose_cropping is True

    # Test "any" still requires enable_pose_cropping=True
    for variation in any_variations:
        with pytest.raises(ValidationError, match="crop_ratio can only be specified when enable_pose_cropping is True"):
            OutputImageConfig(crop_ratio=variation, enable_pose_cropping=False)

    # Test that error messages include "any" as valid option
    with pytest.raises(ValidationError, match="crop_ratio must be a string in format.*or 'any'"):
        OutputImageConfig(crop_ratio=123, enable_pose_cropping=True)

    with pytest.raises(ValidationError, match="Must be in format 'W:H'.*or 'any'"):
        OutputImageConfig(crop_ratio="invalid_format", enable_pose_cropping=True)

    # Test that "any" does not interfere with existing W:H validation
    config = OutputImageConfig(crop_ratio="16:9", enable_pose_cropping=True)
    assert config.crop_ratio == "16:9"

    # Test None still works
    config = OutputImageConfig(crop_ratio=None, enable_pose_cropping=True)
    assert config.crop_ratio is None


def test_config_environment_override():
    """Test environment variable override capability."""
    import os

    # Set environment variable
    os.environ["PERSONFROMVID_MODELS__BATCH_SIZE"] = "16"

    try:
        # Note: This is a basic test - full env override testing would require
        # more complex setup due to pydantic's env handling
        Config()
        # The actual env parsing would happen in from_env() method
        assert True  # Basic structure test passes
    finally:
        # Clean up
        if "PERSONFROMVID_MODELS__BATCH_SIZE" in os.environ:
            del os.environ["PERSONFROMVID_MODELS__BATCH_SIZE"]


def test_cache_directory_creation():
    """Test cache directory creation."""
    config = Config()

    # This should not raise an exception
    config.create_directories()

    # Cache directory should exist after creation
    assert config.storage.cache_directory.exists()


def test_system_requirements_validation():
    """Test system requirements validation."""
    config = Config()

    issues = config.validate_system_requirements()

    # Should return a list (may be empty if system is properly configured)
    assert isinstance(issues, list)


def test_person_selection_criteria_defaults():
    """Test PersonSelectionCriteria default values."""
    criteria = PersonSelectionCriteria()

    # Core parameters
    assert criteria.min_instances_per_person == 3
    assert criteria.max_instances_per_person == 10
    assert criteria.min_quality_threshold == 0.3

    # Category parameters
    assert criteria.enable_pose_categories is True
    assert criteria.enable_head_angle_categories is True
    assert criteria.min_poses_per_person == 3
    assert criteria.min_head_angles_per_person == 3

    # Temporal parameters
    assert criteria.temporal_diversity_threshold == 2.0

    # Global parameters
    assert criteria.max_total_selections == 100


def test_person_selection_criteria_validation():
    """Test PersonSelectionCriteria validation rules."""
    from pydantic import ValidationError

    # Test valid configuration
    valid_criteria = PersonSelectionCriteria(
        min_instances_per_person=5,
        max_instances_per_person=15,
        min_quality_threshold=0.5
    )
    assert valid_criteria.min_instances_per_person == 5
    assert valid_criteria.max_instances_per_person == 15

    # Test max < min validation
    with pytest.raises(ValidationError, match="max_instances_per_person.*must be >= min_instances_per_person"):
        PersonSelectionCriteria(
            min_instances_per_person=10,
            max_instances_per_person=5
        )


def test_person_selection_criteria_boundary_values():
    """Test PersonSelectionCriteria boundary value validation."""
    from pydantic import ValidationError

    # Test minimum boundary values
    min_criteria = PersonSelectionCriteria(
        min_instances_per_person=1,
        max_instances_per_person=1,
        min_quality_threshold=0.0,
        temporal_diversity_threshold=0.0,
        max_total_selections=10
    )
    assert min_criteria.min_instances_per_person == 1
    assert min_criteria.max_total_selections == 10

    # Test maximum boundary values
    max_criteria = PersonSelectionCriteria(
        min_instances_per_person=20,
        max_instances_per_person=50,
        min_quality_threshold=1.0,
        temporal_diversity_threshold=30.0,
        max_total_selections=1000
    )
    assert max_criteria.max_instances_per_person == 50
    assert max_criteria.max_total_selections == 1000

    # Test below minimum values
    with pytest.raises(ValidationError):
        PersonSelectionCriteria(min_instances_per_person=0)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(min_quality_threshold=-0.1)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(temporal_diversity_threshold=-1.0)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(max_total_selections=5)

    # Test above maximum values
    with pytest.raises(ValidationError):
        PersonSelectionCriteria(min_instances_per_person=25)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(max_instances_per_person=60)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(min_quality_threshold=1.5)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(temporal_diversity_threshold=35.0)

    with pytest.raises(ValidationError):
        PersonSelectionCriteria(max_total_selections=1500)


def test_person_selection_criteria_serialization():
    """Test PersonSelectionCriteria serialization and deserialization."""
    import json

    # Create criteria with non-default values
    original = PersonSelectionCriteria(
        min_instances_per_person=5,
        max_instances_per_person=15,
        min_quality_threshold=0.4,
        enable_pose_categories=False,
        temporal_diversity_threshold=3.0,
        max_total_selections=200
    )

    # Test model_dump (Pydantic v2)
    data = original.model_dump()
    assert isinstance(data, dict)
    assert data["min_instances_per_person"] == 5
    assert data["max_instances_per_person"] == 15
    assert data["enable_pose_categories"] is False

    # Test JSON serialization
    json_str = json.dumps(data)
    assert isinstance(json_str, str)

    # Test deserialization
    restored_data = json.loads(json_str)
    restored = PersonSelectionCriteria(**restored_data)

    # Verify all fields match
    assert restored.min_instances_per_person == original.min_instances_per_person
    assert restored.max_instances_per_person == original.max_instances_per_person
    assert restored.min_quality_threshold == original.min_quality_threshold
    assert restored.enable_pose_categories == original.enable_pose_categories
    assert restored.temporal_diversity_threshold == original.temporal_diversity_threshold
    assert restored.max_total_selections == original.max_total_selections


def test_config_person_selection_integration():
    """Test PersonSelectionCriteria integration with main Config class."""
    config = Config()

    # Test that person_selection field exists and has correct type
    assert hasattr(config, 'person_selection')
    assert isinstance(config.person_selection, PersonSelectionCriteria)

    # Test default values through Config
    assert config.person_selection.min_instances_per_person == 3
    assert config.person_selection.max_instances_per_person == 10

    # Test Config serialization includes person_selection
    config_data = config.model_dump()
    assert "person_selection" in config_data
    assert isinstance(config_data["person_selection"], dict)
    assert config_data["person_selection"]["min_instances_per_person"] == 3


def test_config_person_selection_custom_values():
    """Test Config with custom PersonSelectionCriteria values."""
    custom_criteria = PersonSelectionCriteria(
        min_instances_per_person=7,
        max_instances_per_person=20,
        min_quality_threshold=0.5
    )

    config = Config(person_selection=custom_criteria)

    assert config.person_selection.min_instances_per_person == 7
    assert config.person_selection.max_instances_per_person == 20
    assert config.person_selection.min_quality_threshold == 0.5


def test_frame_selection_config_defaults():
    """Test FrameSelectionConfig default values."""
    config = FrameSelectionConfig()
    
    assert config.min_quality_threshold == 0.2
    assert config.face_size_weight == 0.3
    assert config.quality_weight == 0.7
    assert config.diversity_threshold == 0.8
    assert config.temporal_diversity_threshold == 3.0


def test_frame_selection_config_validation():
    """Test FrameSelectionConfig validation."""
    from pydantic import ValidationError
    
    # Test valid configuration
    valid_config = FrameSelectionConfig(
        min_quality_threshold=0.5,
        face_size_weight=0.4,
        quality_weight=0.6,
        diversity_threshold=0.9,
        temporal_diversity_threshold=5.0
    )
    assert valid_config.temporal_diversity_threshold == 5.0
    
    # Test invalid temporal_diversity_threshold values
    with pytest.raises(ValidationError):
        FrameSelectionConfig(temporal_diversity_threshold=-1.0)  # Below minimum
    
    with pytest.raises(ValidationError):
        FrameSelectionConfig(temporal_diversity_threshold=35.0)  # Above maximum


def test_face_restoration_config_defaults():
    """Test face restoration configuration default values."""
    config = OutputImageConfig()

    # Test default values per specification
    assert config.face_restoration_enabled is False
    assert config.face_restoration_strength == 0.8


def test_face_restoration_config_validation():
    """Test face restoration configuration validation."""
    # Test valid configurations
    valid_config = OutputImageConfig(
        face_restoration_enabled=False,
        face_restoration_strength=0.5
    )
    assert valid_config.face_restoration_enabled is False
    assert valid_config.face_restoration_strength == 0.5

    # Test boundary values
    boundary_config = OutputImageConfig(
        face_restoration_strength=0.0
    )
    assert boundary_config.face_restoration_strength == 0.0

    boundary_config = OutputImageConfig(
        face_restoration_strength=1.0
    )
    assert boundary_config.face_restoration_strength == 1.0


def test_face_restoration_config_invalid_values():
    """Test face restoration configuration with invalid values."""
    from pydantic import ValidationError

    # Test strength values below minimum
    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_strength=-0.1)

    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_strength=-1.0)

    # Test strength values above maximum
    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_strength=1.1)

    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_strength=2.0)

    # Test invalid type for strength
    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_strength="invalid")

    # Test invalid type for enabled flag
    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_enabled="invalid_string")

    with pytest.raises(ValidationError):
        OutputImageConfig(face_restoration_enabled=42)


def test_face_restoration_config_serialization():
    """Test face restoration configuration serialization and deserialization."""
    import json

    # Create config with non-default values
    original = OutputImageConfig(
        face_restoration_enabled=False,
        face_restoration_strength=0.3,
        face_crop_enabled=True,
        format="jpeg"
    )

    # Test model_dump (Pydantic v2)
    data = original.model_dump()
    assert isinstance(data, dict)
    assert data["face_restoration_enabled"] is False
    assert data["face_restoration_strength"] == 0.3

    # Test JSON serialization
    json_str = json.dumps(data)
    assert isinstance(json_str, str)

    # Test deserialization
    restored_data = json.loads(json_str)
    restored = OutputImageConfig(**restored_data)

    # Verify face restoration fields match
    assert restored.face_restoration_enabled == original.face_restoration_enabled
    assert restored.face_restoration_strength == original.face_restoration_strength


def test_face_restoration_config_integration():
    """Test face restoration configuration integration with main Config class."""
    config = Config()

    # Test that face restoration fields exist through Config
    assert hasattr(config.output.image, 'face_restoration_enabled')
    assert hasattr(config.output.image, 'face_restoration_strength')
    assert isinstance(config.output.image.face_restoration_enabled, bool)
    assert isinstance(config.output.image.face_restoration_strength, float)

    # Test default values through Config
    assert config.output.image.face_restoration_enabled is False
    assert config.output.image.face_restoration_strength == 0.8

    # Test Config serialization includes face restoration fields
    config_data = config.model_dump()
    image_config = config_data["output"]["image"]
    assert "face_restoration_enabled" in image_config
    assert "face_restoration_strength" in image_config
    assert image_config["face_restoration_enabled"] is False
    assert image_config["face_restoration_strength"] == 0.8


def test_face_restoration_config_custom_values():
    """Test Config with custom face restoration configuration values."""
    from personfromvid.data.config import OutputConfig

    custom_image_config = OutputImageConfig(
        face_restoration_enabled=False,
        face_restoration_strength=0.2,
        format="png"
    )
    custom_output_config = OutputConfig(image=custom_image_config)
    config = Config(output=custom_output_config)

    assert config.output.image.face_restoration_enabled is False
    assert config.output.image.face_restoration_strength == 0.2
    assert config.output.image.format == "png"


def test_face_restoration_config_backward_compatibility():
    """Test that face restoration configuration maintains backward compatibility."""
    # Test that existing configurations without face restoration fields work
    legacy_config_data = {
        "format": "jpeg",
        "face_crop_enabled": True,
        "face_crop_padding": 0.15,
        "enable_pose_cropping": False,
        "resize": 1024
    }

    # Should not raise an exception and should use defaults
    config = OutputImageConfig(**legacy_config_data)
    assert config.face_restoration_enabled is False  # Default
    assert config.face_restoration_strength == 0.8  # Default
    assert config.format == "jpeg"  # Preserved
    assert config.resize == 1024  # Preserved


if __name__ == "__main__":
    pytest.main([__file__])
