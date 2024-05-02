"""Aliases for commonly used types"""

from pathlib import Path

PathLike = str | Path
PathOrPaths = PathLike | list[PathLike]
