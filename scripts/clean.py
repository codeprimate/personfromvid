#!/usr/bin/env python3
"""
Description: This script cleans up temporary files, build artifacts, and caches.
Usage: python scripts/clean.py
"""
import os
import shutil
from pathlib import Path


def main():
    """Main function to clean the project."""
    # Change to the project root directory
    project_root = Path(__file__).parent.parent.resolve()
    os.chdir(project_root)

    print("🧹 Cleaning project...")

    # Directories to remove
    directories_to_remove = [
        "build",
        "dist",
        "htmlcov",
        "personfromvid.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache"
    ]

    for directory in directories_to_remove:
        dir_path = Path(directory)
        if dir_path.is_dir():
            print(f"  - Removing directory: {directory}")
            shutil.rmtree(dir_path)

    # Find and remove files by pattern
    file_patterns_to_remove = [".coverage*", ".DS_Store"]

    for pattern in file_patterns_to_remove:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                print(f"  - Removing file: {file_path.name}")
                file_path.unlink()

    # Find and remove __pycache__ directories
    print("  - Removing Python cache files...")
    for pycache in project_root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)

    # Find and remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        if pyc_file.is_file():
            pyc_file.unlink()

    # Remove test cache if it exists and is not empty
    test_cache_dir = Path("test_cache")
    if test_cache_dir.is_dir():
        print("  - Removing test_cache contents")
        for item in test_cache_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    print("✅ Cleaning complete.")

if __name__ == "__main__":
    main()
