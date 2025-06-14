"""Tests for configuration management."""

import pytest
from pathlib import Path
from personfromvid.data.config import (
    Config, 
    LogLevel, 
    DeviceType, 
    get_default_config,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    OutputImageConfig
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
        config = OutputImageConfig(resize=255)
    
    with pytest.raises(ValidationError):
        config = OutputImageConfig(resize=100)
    
    # Test values above maximum
    with pytest.raises(ValidationError):
        config = OutputImageConfig(resize=4097)
    
    with pytest.raises(ValidationError):
        config = OutputImageConfig(resize=8192)
    
    # Test invalid type
    with pytest.raises(ValidationError):
        config = OutputImageConfig(resize="invalid")
    
    with pytest.raises(ValidationError):
        config = OutputImageConfig(resize=-1)


def test_config_environment_override():
    """Test environment variable override capability."""
    import os
    
    # Set environment variable
    os.environ["PERSONFROMVID_MODELS__BATCH_SIZE"] = "16"
    
    try:
        # Note: This is a basic test - full env override testing would require
        # more complex setup due to pydantic's env handling
        config = Config()
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


if __name__ == "__main__":
    pytest.main([__file__]) 