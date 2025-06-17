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
- 🔄 **Resumable Processing**: Automatically saves progress and resumes interrupted sessions (use `--force` to restart from scratch).
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

# Force restart processing (clears previous state)
personfromvid video.mp4 --force

# Keep temporary files for debugging
personfromvid video.mp4 --keep-temp

# Disable structured output (use basic logging)
personfromvid video.mp4 --no-structured-output
```

## Command-line Options

`personfromvid` offers many options to customize its behavior. Here are the available options:

### General Options
| Option | Alias | Description | Default |
| --- | --- | --- | --- |
| `--config` | `-c` | Path to a YAML or JSON configuration file. | `None` |
| `--output-dir` | `-o` | Directory to save output files. | Video's directory |
| `--log-level` | `-l` | Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `--verbose` | `-v` | Enable verbose output (sets log level to `DEBUG`). | `False` |
| `--quiet` | `-q` | Suppress non-essential output. | `False` |
| `--no-structured-output` | | Disable structured output format (use basic logging). | `False` |
| `--version` | | Show version information and exit. | `False` |

### AI Model Options
| Option | Description | Default |
| --- | --- | --- |
| `--device` | Device to use for AI models (`auto`, `cpu`, `gpu`). | `auto` |
| `--batch-size` | Batch size for AI model inference (1-64). | `1` |
| `--confidence` | Confidence threshold for detections (0.0-1.0). | `0.3` |

### Frame Processing Options
| Option | Description | Default |
| --- | --- | --- |
| `--max-frames` | Maximum frames to extract per video. | `None` |
| `--quality-threshold` | Quality threshold for frame selection (0.0-1.0). | `0.2` |

### Output Options
| Option | Description | Default |
| --- | --- | --- |
| `--output-format` | Output image format (`jpeg` or `png`). | `png` |
| `--output-jpeg-quality` | Quality for JPEG output (70-100). | `95` |
| `--output-face-crop-enabled` / `--no-output-face-crop-enabled` | Enable or disable generation of cropped face images. | `True` |
| `--output-face-crop-padding` | Padding around face bounding box (0.0-1.0). | `0.3` |
| `--crop` | Enable generation of cropped pose images. | `False` |
| `--crop-padding` | Padding around pose bounding box for crops (0.0-1.0). | `0.1` |
| `--output-png-optimize` / `--no-output-png-optimize` | Enable or disable PNG optimization. | `True` |
| `--resize` | Maximum dimension for proportional resizing (256-4096 pixels). | `None` |
| `--min-frames-per-category` | Minimum frames to output per pose/angle category (1-10). | `3` |
| `--max-frames-per-category` | Maximum frames to output per pose/angle category (1-100). | `5` |

### Processing Control Options
| Option | Description | Default |
| --- | --- | --- |
| `--force` | Force restart analysis by deleting existing state. | `False` |
| `--keep-temp` | Keep temporary files after processing. | `False` |

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
- **Full Frame Images**: The filename captures the detected pose, head orientation, and shot type.
- **Cropped Face Images**: Saved if `output.image.face_crop_enabled` is `true`. The filename includes head orientation details.
- **Cropped Pose Images**: Saved if `output.image.enable_pose_cropping` is `true`. A `_crop` suffix is added to the original filename.

## Configuration

Person From Vid can be configured via a YAML file, environment variables, or command-line arguments.

### Configuration File

Create a YAML file (e.g., `config.yaml`) to manage settings. CLI arguments will override file settings.

```yaml
# config.yaml

# AI Models and device settings
models:
  device: "auto"  # "cpu", "gpu", or "auto"
  batch_size: 1
  confidence_threshold: 0.3
  face_detection_model: "yolov8s-face"
  pose_estimation_model: "yolov8s-pose"
  head_pose_model: "sixdrepnet"

# Frame extraction strategy
frame_extraction:
  temporal_sampling_interval: 0.25 # Seconds between samples
  enable_keyframe_detection: true
  enable_temporal_sampling: true
  max_frames_per_video: null # No limit
  deduplication_enabled: true

# Quality assessment thresholds
quality:
  blur_threshold: 100.0
  brightness_min: 30.0
  brightness_max: 225.0
  contrast_min: 20.0
  enable_multiple_metrics: true

# Pose classification thresholds
pose_classification:
  standing_hip_knee_angle_min: 160.0
  sitting_hip_knee_angle_min: 80.0
  sitting_hip_knee_angle_max: 120.0
  squatting_hip_knee_angle_max: 90.0
  closeup_face_area_threshold: 0.15

# Head angle classification
head_angle:
  yaw_threshold_degrees: 22.5
  pitch_threshold_degrees: 22.5
  max_roll_degrees: 30.0
  profile_yaw_threshold: 67.5

# Closeup detection settings
closeup_detection:
  extreme_closeup_threshold: 0.25
  closeup_threshold: 0.15
  medium_closeup_threshold: 0.08
  medium_shot_threshold: 0.03
  shoulder_width_threshold: 0.35
  enable_distance_estimation: true

# Frame selection criteria
frame_selection:
  min_quality_threshold: 0.2
  face_size_weight: 0.3
  quality_weight: 0.7
  diversity_threshold: 0.8

# Output settings
output:
  min_frames_per_category: 3
  max_frames_per_category: 5
  preserve_metadata: true
  image:
    format: "jpeg"
    jpeg:
      quality: 98
    png:
      optimize: true
    face_crop_enabled: true
    face_crop_padding: 0.3
    enable_pose_cropping: true

# Storage and caching
storage:
  cache_directory: "~/.cache/personfromvid"  # Override default cache location
  temp_directory: null                       # Auto-generated if null
  keep_temp: false                           # Keep temporary files after processing
  force_temp_cleanup: false                  # Force cleanup before starting
  cleanup_temp_on_success: true              # Clean up temp files on success
  cleanup_temp_on_failure: false             # Keep temp files if processing fails
  max_cache_size_gb: 5.0

# Processing behavior
processing:
  force_restart: false                       # Force restart by deleting existing state
  save_intermediate_results: true
  max_processing_time_minutes: null         # No time limit
  parallel_workers: 1

# Logging configuration
logging:
  level: "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
  enable_file_logging: false
  log_file: null
  enable_rich_console: true
  enable_structured_output: true
  verbose: false
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
2. **Adjust Batch Size**: Increase `--batch-size` to improve GPU utilization. Start with 4 or 8, then try 16 if you have sufficient GPU memory. Default is 1 for maximum compatibility.
3. **Limit Frame Extraction**: Use `--max-frames` on very long videos to get results faster.
4. **Use Structured Output**: The default structured output (`--no-structured-output` to disable) provides better progress tracking and user experience.

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

**Processing seems stuck or interrupted:**
```bash
# Force restart from the beginning (clears saved state)
personfromvid video.mp4 --force

# Keep temporary files for debugging
personfromvid video.mp4 --keep-temp
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