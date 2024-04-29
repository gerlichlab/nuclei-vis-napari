"""Aliases for commonly used types"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

PathLike = str | Path
PathOrPaths = PathLike | list[PathLike]
PixelArray = npt.NDArray[np.uint8 | np.uint16]
