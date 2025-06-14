# Person From Vid

[![PyPI version](https://badge.fury.io/py/personfromvid.svg)](https://badge.fury.io/py/personfromvid) [![Python versions](https://img.shields.io/pypi/pyversions/personfromvid.svg)](https://pypi.org/project/personfromvid) [![License: GPL-3.0-or-later](https://img.shields.io/pypi/l/personfromvid.svg)](https://github.com/personfromvid/personfromvid/blob/main/LICENSE)

AI-powered video frame extraction and pose categorization tool that analyzes video files to identify and extract high-quality frames containing people in specific poses and head orientations.

## Features

- 🎥 **Video Analysis**: Supports multiple video formats (MP4, AVI, MOV, MKV, WebM, etc.).
- 🤖 **AI-Powered Detection**: Uses state-of-the-art models for face detection (`yolov8s-face`), pose estimation (`yolov8s-pose`), and head pose analysis (`sixdrepnet`).
- 🧠 **Smart Frame Selection**:
    - **Keyframe Detection**: Prioritizes information-rich I-frames.
    - **Temporal Sampling**: Extracts frames at regular intervals to ensure coverage.
    - **Deduplication**: Avoids saving visually similar frames.
- 📐 **Pose & Shot Classification**:
    - Automatically categorizes poses into **standing, sitting, and squatting**.
    - Classifies shot types like **closeup, medium shot, and full body**.
- 👤 **Head Orientation**: Classifies head directions into 9 cardinal orientations (front, profile, looking up/down, etc.).
- 🖼️ **Advanced Quality Assessment**: Uses multiple metrics like blur, brightness, and contrast to select the sharpest, best-lit frames.
- ⚡ **GPU Acceleration**: Optional CUDA/MPS support for significantly faster processing.
- 📊 **Rich Progress Tracking**: Modern console interface with real-time progress displays and detailed status.
- 🔄 **Resumable Processing**: Automatically saves progress and allows resuming interrupted sessions.
- ⚙️ **Highly Configurable**: Extensive configuration options via CLI, YAML files, or environment variables.

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for video processing)

#### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html) or use:
```bash
choco install ffmpeg  # Using Chocolatey
```

### Install Person From Vid

#### From PyPI
The recommended way to install is via `pip`:
```bash
pip install personfromvid
```

#### From Source
Alternatively, to install from source:
```bash
git clone https://github.com/personfromvid/personfromvid.git
cd personfromvid
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Process a video file, saving results to the same directory
personfromvid video.mp4

# Specify a different output directory
personfromvid video.mp4 --output-dir ./extracted_frames

# Enable verbose logging for detailed information
personfromvid video.mp4 --verbose

# Use GPU for faster processing (if available)
personfromvid video.mp4 --device gpu
```

### Advanced Usage

```bash
# High-quality processing with custom settings
personfromvid video.mp4 \
    --output-dir ./custom_output \
    --output-jpeg-quality 98 \
    --confidence 0.5 \
    --batch-size 16 \
    --max-frames 1000

# Resize output images to a maximum of 1024 pixels
personfromvid video.mp4 --resize 1024

# Resume an interrupted process
personfromvid video.mp4 --resume
```

## Command-line Options

`personfromvid` offers many options to customize its behavior. Here are some of the most common ones:

| Option | Alias | Description | Default |
| --- | --- | --- | --- |
| `--config` | `-c` | Path to a YAML configuration file. | `None` |
| `--output-dir` | `-o` | Directory to save output files. | Video's directory |
| `--device` | | Device to use for AI models (`auto`, `cpu`, `gpu`). | `auto` |
| `--log-level` | `-l` | Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `--verbose` | `-v` | Enable verbose output (sets log level to `DEBUG`). | `False` |
| `--quiet` | `-q` | Suppress non-essential output. | `False` |
| `--resume` | | Resume from the last saved state. | `True` |
| `--batch-size`| | Batch size for AI model inference. | `8` |
| `--confidence`| | Confidence threshold for detections. | `0.3` |
| `--max-frames`| | Maximum frames to process from the video. | `None` |
| `--output-format`| | Output image format (`jpeg` or `png`). | `jpeg` |
| `--output-jpeg-quality`| | Quality for JPEG output (70-100). | `95` |
| `--resize`| | Maximum dimension for proportional image resizing (256-4096 pixels). | `None` |
| `--min-frames-per-category`| | Minimum frames to output per pose/angle category (1-10). | `3` |
| `--no-output-face-crop-enabled`| | Disable generation of cropped face images. | `False` |
| `--no-output-full-frame-enabled`| | Disable saving of full-frame images. | `False` |
| `--force` | | Force cleanup of existing temp directory before starting. | `False` |
| `--keep-temp` | | Keep temporary files after processing for debugging. | `False` |
| `--version` | | Show version information and exit. | `False` |

For a full list of options, run `personfromvid --help`.

## Output Structure

By default, Person From Vid saves all output files into the same directory as the input video. You can specify a different location with the `--output-dir` option. All files are prefixed with the base name of the video file.

Here is an example of the output for a video named `interview.mp4`:

```
interview_info.json                     # Detailed processing metadata and results
interview_standing_front_closeup_001.jpg  # Full frame: {video}_{pose}_{head}_{shot}_{rank}.jpg
interview_sitting_profile-left_medium-shot_002.jpg
interview_face_front_001.jpg              # Face crop: {video}_face_{head-angle}_{rank}.jpg
interview_face_profile-right_002.jpg
```

- **`{video_base_name}_info.json`**: A detailed JSON file containing the configuration used, video metadata, and data for every selected frame.
- **Full Frame Images**: Saved if `output.image.full_frame_enabled` is `true` (default). The filename captures the detected pose, head orientation, and shot type.
- **Face Crop Images**: Saved if `output.image.face_crop_enabled` is `true` (default). These files contain only the cropped face for easier analysis. All images are saved in a single flat directory.

## Configuration

Person From Vid can be configured via a YAML file, environment variables, or command-line arguments.

### Configuration File

Create a YAML file (e.g., `config.yaml`) to manage settings. CLI arguments will override file settings.

```yaml
# config.yaml

# Models and device settings
models:
  device: "auto"  # "cpu", "gpu", or "auto"
  batch_size: 8
  confidence_threshold: 0.3
  face_detection_model: "yolov8s-face"
  pose_estimation_model: "yolov8s-pose"
  head_pose_model: "sixdrepnet"

# Frame extraction strategy
frame_extraction:
  temporal_sampling_interval: 0.25 # Seconds between samples
  enable_keyframe_detection: true
  max_frames_per_video: null # No limit

# Quality assessment thresholds
quality:
  blur_threshold: 100.0
  brightness_min: 30.0
  brightness_max: 225.0
  contrast_min: 20.0

# Output settings
output:
  min_frames_per_category: 3
  image:
    format: "jpeg" # 'jpeg' or 'png'
    jpeg:
      quality: 95
    png:
      optimize: true
    face_crop_enabled: true
    full_frame_enabled: true
    face_crop_padding: 0.2 # 20% padding

# Processing and storage behavior
processing:
  enable_resume: true
storage:
  cache_directory: "~/.cache/personfromvid"  # Override default cache location
  keep_temp: false                           # Keep temporary files after processing
  force_temp_cleanup: false                  # Force cleanup before starting
  cleanup_temp_on_success: true              # Clean up temp files on success
  cleanup_temp_on_failure: false             # Keep temp files if processing fails

# Logging configuration
logging:
  level: "INFO" # DEBUG, INFO, WARNING, ERROR
  enable_structured_output: true
```

Use with:
```bash
personfromvid video.mp4 --config config.yaml
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/personfromvid/personfromvid.git
cd personfromvid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Project Structure

```
personfromvid/
├── personfromvid/           # Main package
│   ├── cli.py              # Command-line interface
│   ├── core/               # Core processing modules
│   ├── models/             # AI model management
│   ├── analysis/           # Image analysis and classification
│   ├── output/             # Output generation
│   ├── utils/              # Utility modules
│   └── data/               # Data models and configuration
├── tests/                  # Test suite
├── docs/                   # Documentation
└── scripts/                # Development scripts
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=personfromvid

# Run specific test modules
pytest tests/unit/test_config.py
```

### Code Quality

```bash
# Format code
black personfromvid/

# Check linting
flake8 personfromvid/

# Type checking
mypy personfromvid/
```

### Cleaning Up

To remove temporary files, build artifacts, and caches, run the cleaning script:

```bash
python scripts/clean.py
```

## System Requirements

### Minimum Requirements
- Python 3.10+
- 4GB RAM
- 1GB disk space for dependencies and cache
- FFmpeg

### Recommended Requirements
- Python 3.11+
- 8GB+ RAM
- 5GB+ disk space for cache
- NVIDIA GPU with CUDA support for acceleration
- FFmpeg with hardware acceleration support

## Supported Formats

### Video Formats
- MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP, OGV

### Output Formats
- PNG images (configurable quality)
- JPEG images (configurable quality)
- JSON metadata files

## Cache and Temporary Files

Person From Vid uses a centralized cache directory to store both AI models and temporary files during video processing. This keeps your video directories clean and makes cache management easier.

### Cache Directory Locations

The cache directory is automatically determined based on your operating system:

- **Linux**: `~/.cache/personfromvid/`
- **macOS**: `~/Library/Caches/personfromvid/`
- **Windows**: `C:\Users\{username}\AppData\Local\codeprimate\personfromvid\Cache\`

### Cache Structure

```
personfromvid/                  # Base cache directory
├── models/                     # AI model files
│   ├── yolov8s-face/          # Face detection model
│   ├── yolov8s-pose/          # Pose estimation model
│   └── sixdrepnet/            # Head pose model
└── temp/                      # Temporary processing files
    └── temp_{video_name}/     # Per-video temporary directory
        └── frames/            # Extracted frames during processing
```

### Temporary Files

During video processing, temporary files (extracted frames, intermediate data) are stored in the cache directory under `temp/temp_{video_name}/`. These files are:

- **Automatically cleaned up** after successful processing (configurable)
- **Kept for debugging** if processing fails or if `--keep-temp` is used
- **Isolated per video** to allow concurrent processing of multiple videos

### Cache Management

```bash
# Keep temporary files after processing (for debugging)
personfromvid video.mp4 --keep-temp

# Force cleanup of existing temp files before starting
personfromvid video.mp4 --force

# Configure cache location via config file
personfromvid video.mp4 --config custom_config.yaml
```

You can manually clean the cache directory to free up disk space, or configure automatic cleanup in your configuration file.

## AI Models

Person From Vid uses the following default AI models, which are automatically downloaded and cached on first use in the cache directory described above.

- **Face Detection**: `yolov8s-face` - A YOLOv8 model trained for face detection.
- **Pose Estimation**: `yolov8s-pose` - A YOLOv8 model for human pose estimation.
- **Head Pose**: `sixdrepnet` - A model for 6DoF head pose estimation.

Alternative models can be configured.

## Performance Tips

1. **Use a GPU**: The single most effective way to speed up processing is to use an NVIDIA GPU with `--device gpu`.
2. **Adjust Batch Size**: Increase `--batch-size` to improve GPU utilization. A size of 8 or 16 is a good starting point.
3. **Limit Frame Extraction**: Use `--max-frames` on very long videos to get results faster.

## Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# Check if FFmpeg is installed
ffmpeg -version
# Install if missing (see Prerequisites section)
```

**CUDA/GPU issues:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
# Fall back to CPU processing
personfromvid video.mp4 --device cpu
```

**Memory issues:**
```bash
# Reduce batch size
personfromvid video.mp4 --batch-size 1
```

**Permission errors:**
```bash
# Check output directory permissions
ls -la /path/to/output/directory
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the GPL-3.0-or-later - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://github.com/codeprimate/personfromvid/docs)
- 🐛 [Issue Tracker](https://github.com/codeprimate/personfromvid/issues)
- 💬 [Discussions](https://github.com/codeprimate/personfromvid/discussions)

---

**Person From Vid** - Extracting moments, categorizing poses, powered by AI. 