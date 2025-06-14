"""Entry point for running personfromvid as a module.

This allows the package to be executed via:
    python -m personfromvid video.mp4
"""

from .cli import main

if __name__ == "__main__":
    main()
