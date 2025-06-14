#!/usr/bin/env bash
#
# Description: This script cleans up temporary files, build artifacts, and caches.
# Usage: ./scripts/clean.sh

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.
set -o pipefail # Return the exit status of the last command in the pipe that failed.

# Change to the project root directory
cd "$(dirname "$0")/.."

echo "🧹 Cleaning project..."

# Directories to remove
directories_to_remove=(
    "build"
    "dist"
    "htmlcov"
    "personfromvid.egg-info"
    ".pytest_cache"
    ".mypy_cache"
    "tmp"
    ".coverage"
)

for dir in "${directories_to_remove[@]}"; do
    if [ -d "$dir" ]; then
        echo "  - Removing directory: $dir"
        rm -rf "$dir"
    fi
done

# Find and remove __pycache__ directories and .pyc files
echo "  - Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete


# Remove test cache if it exists and is not empty
if [ -d "test_cache" ] && [ "$(ls -A "test_cache")" ]; then
    echo "  - Removing test_cache contents"
    rm -rf test_cache/*
fi


echo "✅ Cleaning complete." 