import os
import subprocess
from pathlib import Path

# Configuration section
CONFIG = {
    'input_dir': 'input',
    'output_base_dir': 'input',
    'device': 'gpu',
    'log_level': 'INFO',
    'verbose': False,
    'quiet': False,
    'resume': True,
    'batch_size': 8,
    'confidence': 0.3,
    'max_frames': None,
    'output_format': 'jpeg',
    'output_jpeg_quality': 95,
    'resize': None,
    'min_frames_per_category': 3,
    'no_output_face_crop_enabled': False,
    'no_output_full_frame_enabled': False,
    'force': False,
    'keep_temp': False,
    'overwrite': False  # If True, process even if output directory exists
}

def run_personfromvid(video_path: str, output_dir: str, current: int, total: int) -> None:
    """Run personfromvid command with configured options for a single video."""
    print(f"\nStart processing video {current}/{total}: {video_path.name}")
    
    cmd = ['personfromvid', str(video_path)]
    
    # Add output directory
    cmd.extend(['--output-dir', str(output_dir)])
    
    # Add other configured options
    cmd.extend(['--device', CONFIG['device']])
    cmd.extend(['--log-level', CONFIG['log_level']])
    
    if CONFIG['verbose']:
        cmd.append('--verbose')
    if CONFIG['quiet']:
        cmd.append('--quiet')
    if CONFIG['resume']:
        cmd.append('--resume')
    if CONFIG['force']:
        cmd.append('--force')
    if CONFIG['keep_temp']:
        cmd.append('--keep_temp')
    if CONFIG['no_output_face_crop_enabled']:
        cmd.append('--no-output-face-crop-enabled')
    if CONFIG['no_output_full_frame_enabled']:
        cmd.append('--no-output-full-frame-enabled')
        
    if CONFIG['batch_size'] is not None:
        cmd.extend(['--batch-size', str(CONFIG['batch_size'])])
    if CONFIG['confidence'] is not None:
        cmd.extend(['--confidence', str(CONFIG['confidence'])])
    if CONFIG['max_frames'] is not None:
        cmd.extend(['--max-frames', str(CONFIG['max_frames'])])
    if CONFIG['output_format'] is not None:
        cmd.extend(['--output-format', CONFIG['output_format']])
    if CONFIG['output_jpeg_quality'] is not None:
        cmd.extend(['--output-jpeg-quality', str(CONFIG['output_jpeg_quality'])])
    if CONFIG['resize'] is not None:
        cmd.extend(['--resize', str(CONFIG['resize'])])
    if CONFIG['min_frames_per_category'] is not None:
        cmd.extend(['--min-frames-per-category', str(CONFIG['min_frames_per_category'])])

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Video {current}/{total} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error processing video {current}/{total}: {e}")

def main():
    """Process all video files in the input directory."""
    input_dir = Path(CONFIG['input_dir'])
    output_base_dir = Path(CONFIG['output_base_dir'])
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return
    
    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    # Check if input directory is empty
    if not video_files:
        print(f"No video files found in {input_dir}.")
        return
    
    # Count videos to process (considering overwrite option)
    total_videos = len(video_files)
    skipped_videos = 0
    to_process = []
    
    for video in video_files:
        output_dir = output_base_dir / video.stem
        if not CONFIG['overwrite'] and output_dir.exists():
            skipped_videos += 1
        else:
            to_process.append(video)
    
    # Print overall progress
    print(f"\n{'='*50}")
    print(f"Detected {total_videos} input {'video' if total_videos == 1 else 'videos'}")
    if skipped_videos > 0:
        print(f"{skipped_videos} already {'has' if skipped_videos == 1 else 'have'} an output folder")
    print(f"Processing {len(to_process)} {'video' if len(to_process) == 1 else 'videos'}")
    print(f"{'='*50}\n")
    
    # Process each video file
    for i, video in enumerate(to_process, 1):
        output_dir = output_base_dir / video.stem
        run_personfromvid(video, output_dir, i, len(to_process))
    
    print(f"\n{'='*50}")
    print("Processing completed!")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()