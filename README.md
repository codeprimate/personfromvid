# Person From Vid

AI-powered video frame extraction and pose categorization tool that analyzes video files to identify and extract high-quality frames containing people in specific poses and head orientations.

## Features

- 🎥 **Video Analysis**: Supports multiple video formats (MP4, AVI, MOV, MKV, WebM, etc.)
- 🤖 **AI-Powered Detection**: Uses state-of-the-art models for face detection, pose estimation, and head pose analysis
- 📐 **Pose Classification**: Automatically categorizes poses into standing, sitting, and squatting
- 👤 **Head Orientation**: Classifies head directions into 9 cardinal orientations
- 🖼️ **Quality Assessment**: Advanced image quality metrics for selecting the best frames
- ⚡ **GPU Acceleration**: Optional CUDA support for faster processing
- 📊 **Progress Tracking**: Rich console interface with real-time progress displays
- 🔄 **Resumable Processing**: Automatic state management with interruption recovery
- ⚙️ **Configurable**: Extensive configuration options via CLI, files, or environment variables

## Installation

### Prerequisites

- Python 3.8 or higher
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

#### From PyPI (when available)
```bash
pip install personfromvid
```

#### From Source
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
# Process a video file
personfromvid video.mp4

# Specify output directory
personfromvid video.mp4 --output-dir ./extracted_frames

# Enable verbose logging
personfromvid video.mp4 --verbose

# Use GPU acceleration
personfromvid video.mp4 --device gpu
```

### Advanced Options

```bash
# High-quality processing with custom settings
personfromvid video.mp4 \
    --jpeg-quality 98 \
    --confidence-threshold 0.8 \
    --batch-size 16 \
    --max-frames 1000

# Resume interrupted processing
personfromvid video.mp4 --resume

# Dry run to validate inputs
personfromvid video.mp4 --dry-run
```

## Output Structure

Person From Vid creates the following output structure:

```
video_name/
├── video_name_info.json          # Processing metadata and results
├── standing_front_001.jpg        # Standing pose, front view
├── standing_front_002.jpg
├── standing_profile_left_001.jpg # Standing pose, left profile
├── sitting_front_001.jpg         # Sitting pose, front view
├── squatting_front_001.jpg       # Squatting pose, front view
└── faces/                        # Face crops for head angle analysis
    ├── front_001.jpg
    ├── profile_left_001.jpg
    └── looking_up_001.jpg
```

## Configuration

### Configuration File

Create a YAML configuration file:

```yaml
# config.yaml
models:
  device: "auto"
  batch_size: 8
  confidence_threshold: 0.7

quality:
  blur_threshold: 100.0
  brightness_min: 30.0
  brightness_max: 225.0

output:
  jpeg_quality: 95
  max_frames_per_category: 3
  create_face_crops: true

processing:
  enable_resume: true
  parallel_workers: 1
```

Use with:
```bash
personfromvid video.mp4 --config config.yaml
```

### Environment Variables

Set configuration via environment variables:

```bash
export PERSONFROMVID_MODELS__DEVICE="gpu"
export PERSONFROMVID_MODELS__BATCH_SIZE=16
export PERSONFROMVID_QUALITY__BLUR_THRESHOLD=150.0
personfromvid video.mp4
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

## System Requirements

### Minimum Requirements
- Python 3.8+
- 2GB RAM
- 500MB disk space
- FFmpeg

### Recommended Requirements
- Python 3.10+
- 8GB RAM
- 5GB disk space
- NVIDIA GPU with CUDA support
- FFmpeg with hardware acceleration

## Supported Formats

### Video Formats
- MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP, OGV

### Output Formats
- JPEG images (configurable quality)
- JSON metadata files

## AI Models

Person From Vid uses the following AI models:

- **Face Detection**: SCRFD or YOLO-based models
- **Pose Estimation**: YOLOv8-Pose or ViTPose
- **Head Pose**: HopeNet or similar models

Models are automatically downloaded and cached on first use.

## Performance Tips

1. **GPU Acceleration**: Use `--device gpu` for faster processing
2. **Batch Size**: Increase `--batch-size` for better GPU utilization
3. **Quality Threshold**: Adjust `--quality-threshold` to filter low-quality frames
4. **Max Frames**: Limit processing with `--max-frames` for large videos

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Person From Vid in your research, please cite:

```bibtex
@software{personfromvid2024,
  title={Person From Vid: AI-powered video frame extraction and pose categorization},
  author={Person From Vid Project},
  year={2024},
  url={https://github.com/personfromvid/personfromvid}
}
```

## Support

- 📖 [Documentation](https://github.com/personfromvid/personfromvid/docs)
- 🐛 [Issue Tracker](https://github.com/personfromvid/personfromvid/issues)
- 💬 [Discussions](https://github.com/personfromvid/personfromvid/discussions)

---

**Person From Vid** - Extracting moments, categorizing poses, powered by AI. 