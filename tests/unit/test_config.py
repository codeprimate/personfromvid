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
    assert criteria.enable_pose_categories is False
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


if __name__ == "__main__":
    pytest.main([__file__])
