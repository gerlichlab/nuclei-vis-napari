"""Reading looptrace-written, ZARR-stored data"""

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from operator import itemgetter
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from gertils.geometry import ImagePoint2D
from gertils.pathtools import find_single_path_by_fov
from gertils.types import FieldOfViewFrom1, NucleusNumber
from gertils.zarr_tools import read_zarr
from numpydoc_decorator import doc  # type: ignore[import-untyped]

# Specific layer types
LayerParams = dict[str, object]
ImageLayer = tuple[npt.ArrayLike, LayerParams, Literal["image"]]
MasksLayer = tuple[npt.ArrayLike, LayerParams, Literal["labels"]]
CentroidsLayer = tuple[npt.ArrayLike, LayerParams, Literal["points"]]
FullDataLayer = ImageLayer | MasksLayer | CentroidsLayer

# Other type aliases
PathLike = str | Path
PathOrPaths = PathLike | list[PathLike]
PixelArray = npt.NDArray[np.uint8 | np.uint16]

Reader = Callable[[PathOrPaths], list[FullDataLayer]]

# Environment variable for finding nuclei channel if needed.
LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR = "LOOPTRACE_NAPARI_NUCLEI_CHANNEL"


@doc(
    summary=(
        "This is the main hook required by napari / napari plugins to provide a Reader" " plugin."
    ),
    parameters=dict(path="Path to the folder (with proper substructure) with nuclei data to view"),
    returns="A callable that accepts a list of paths and returns the layers for nuclei vis",
)
def get_reader(path: PathOrPaths) -> Optional[Reader]:  # noqa: D103
    def do_not_parse(why: str, *, level: int = logging.DEBUG) -> None:
        logging.log(level, "%s, cannot read looptrace nuclei visualisation data: %s", why, path)

    # Input should be a single extant folder.
    if isinstance(path, list):
        do_not_parse("Cannot parse multiple paths for nuclei data, just 1 folder")
        return None
    if not _is_path_like(path):
        do_not_parse(f"Not a path-like: {path}")
        return None
    if not os.path.isdir(path):  # noqa: PTH112
        do_not_parse(f"Not an extant directory: {path}")
        return None
    path: Path = Path(path)  # type: ignore[no-redef]

    # Each of the subpaths to parse must be extant folder.
    if not NucleiDataSubfolders.all_present_within(path):
        do_not_parse(
            "At least one subpath to parse isn't a folder!"
            f" {NucleiDataSubfolders.relpaths(path)}."
        )
        return None

    def parse(root: PathOrPaths) -> list[FullDataLayer]:
        if not _is_path_like(root):
            raise TypeError(f"Non-path-like as nuclei data: {type(root).__name__}")
        # Ignore type warning here b/c we've conditionally proven arg type is correct.
        return list(build_layers(NucleiDataSubfolders.read_all_from_root(root)))  # type: ignore[arg-type]

    return parse


class NucleiDataSubfolders(Enum):
    """The subfolders of nuclei data"""

    IMAGES = "nuc_images"
    MASKS = "nuc_masks"
    CENTERS = "_nuclear_masks_visualisation"

    @classmethod
    def all_present_within(cls, p: PathLike) -> bool:
        """Determine whether all subfolders are present directly in given folder."""
        return all(m.is_present_within(p) for m in cls)

    @classmethod
    def read_all_from_root(cls, p: PathLike) -> dict[FieldOfViewFrom1, "NucleiVisualisationData"]:
        """For each field of view in the given folder, deter,ome the nuclei data paths."""
        image_paths = find_single_path_by_fov(cls.IMAGES.relpath(p), extension=".zarr")
        masks_paths = find_single_path_by_fov(cls.MASKS.relpath(p), extension=".zarr")
        centers_paths = find_single_path_by_fov(
            cls.CENTERS.relpath(p), extension=".nuclear_masks.csv"
        )
        fields_of_view = (
            set(image_paths.keys()) & set(masks_paths.keys()) & set(centers_paths.keys())
        )
        logging.debug("Image paths count: %d", len(image_paths))
        logging.debug("Masks paths count: %d", len(masks_paths))
        logging.debug("Centers paths count: %d", len(centers_paths))
        bundles: dict[FieldOfViewFrom1, "NucleiVisualisationData"] = {}
        for fov in sorted(fields_of_view):
            logging.debug("Reading data for FOV: %d", fov.get)
            image_fp: Path = image_paths[fov]
            logging.debug("Reading nuclei image: %s", image_fp)
            image = read_zarr(image_fp)
            masks_fp: Path = masks_paths[fov]
            logging.debug("Reading nuclei masks: %s", masks_fp)
            masks = read_zarr(masks_fp)
            centers_fp: Path = centers_paths[fov]
            logging.debug("Reading nuclei centers: %s", centers_fp)
            centers = _read_csv(centers_fp)
            bundles[fov] = NucleiVisualisationData(image=image, masks=masks, centers=centers)
        return bundles

    @classmethod
    def relpaths(cls, p: PathLike) -> dict[str, Path]:
        """Give the path to each subfolder, relative to the given parent."""
        return {m.value: m.relpath(p) for m in cls}

    def is_present_within(self, p: PathLike) -> bool:
        """Determine whether this subfolder is directly within given folder."""
        return self.relpath(p).is_dir()

    def relpath(self, p: PathLike) -> Path:
        """Get the path of this subfolder, relative to the given parent."""
        return Path(p) / self.value


@doc(
    summary="Bundle the data needed to visualise nuclei.",
    parameters=dict(
        image="The array of pixel values, e.g. of DAPI staining in a particular FOV",
        masks="""
            The array of region-defining label indicators which represent nuclei regions.
            The values should be nonnegative integers, with 0 representing portions of the
            image outside a nucleus, and nonzero values corresponding to a nucleus.
        """,
        centers="The list of centroids, one for each nuclear mask",
    ),
)
@dataclass(frozen=True, kw_only=True)
class NucleiVisualisationData:  # noqa: D101
    image: PixelArray
    masks: PixelArray
    centers: list[tuple[NucleusNumber, ImagePoint2D]]

    def __post_init__(self) -> None:  # noqa: C901
        # First, handle the raw image.
        if len(self.image.shape) == 5:  # noqa: PLR2004
            if self.image.shape[0] != 1:
                raise ValueError(
                    "5D image for nuclei visualisation must have trivial first"
                    f" dimension; got {self.image.shape[0]} (not 1)"
                )
            object.__setattr__(self, "image", self.image[0])
        if len(self.image.shape) == 4:  # noqa: PLR2004
            if self.image.shape[0] == 1:
                logging.debug("Collapsing trivial channel axis for nuclei image")
                object.__setattr__(self, "image", self.image[0])
            else:
                logging.debug(
                    "Multiple channels in nuclei image; attempting to determine which" " to use"
                )
                nuc_channel: int = determine_nuclei_channel()
                if nuc_channel >= self.image.shape[0]:
                    raise ValueError(
                        "Illegal nuclei channel value (from"
                        f" {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}), {nuc_channel},"
                        f" for channel axis of length {self.image.shape[0]}"
                    )
                logging.debug(
                    "Using nuclei channel (from %s): %d",
                    LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR,
                    nuc_channel,
                )
                object.__setattr__(self, "image", self.image[nuc_channel])
        if len(self.image.shape) == 3:  # noqa: PLR2004
            if self.image.shape[0] == 1:
                logging.debug("Collapsing trivial z-axis for nuclei image")
                object.__setattr__(self, "image", self.image[0])
            else:
                logging.debug("Max projecting along z for nuclei image")
                object.__setattr__(self, "image", max_project_z(self.image))
        if len(self.image.shape) == 2:  # noqa: PLR2004
            # All good, 2D image
            pass
        else:
            raise ValueError(
                f"Cannot use image with {len(self.image.shape)} dimension(s) for nuclei"
                " visualisation"
            )

        # Then, handle the masks image.
        if len(self.masks.shape) == 5:  # noqa: PLR2004
            if any(d != 1 for d in self.masks.shape[:3]):
                raise ValueError(
                    "5D nuclear masks image with at least 1 nontrivial (t, c, z) axis!"
                    f" {self.masks.shape}"
                )
            logging.debug("Reducing 5D nuclear masks to 2D")
            object.__setattr__(self, "masks", self.masks[0, 0, 0])
        if len(self.masks.shape) != 2:  # noqa: PLR2004
            raise ValueError(
                f"Need 2D image for nuclear masks! Got {len(self.masks.shape)}:"
                f" {self.masks.shape}"
            )


@doc(
    summary="Build the multiple layers (image, masks, points) to look at nuclei in napari.",
    parameters=dict(bundles="Mapping from FOV to the bundle of data needed to visualise its data"),
    returns="The layers needed to visualise nuclei across all FOVs",
)
def build_layers(  # noqa: D103
    bundles: Mapping[FieldOfViewFrom1, NucleiVisualisationData],
) -> tuple[ImageLayer, MasksLayer, CentroidsLayer]:
    images = []
    masks = []
    nuclei_points = []
    nuclei_labels = []
    for i, (_, visdata) in enumerate(sorted(bundles.items(), key=itemgetter(0))):
        images.append(visdata.image)
        masks.append(visdata.masks)
        for nuc, pt in visdata.centers:
            nuclei_points.append([i, pt.get_y_coordinate(), pt.get_x_coordinate()])
            nuclei_labels.append(nuc.get)

    # Prep the data for presentation as layers.
    images: PixelArray = np.stack(images)  # type: ignore[no-redef]
    logging.debug("Image layer data shape: %s", images.shape)  # type: ignore[attr-defined]
    masks: PixelArray = np.stack(masks)  # type: ignore[no-redef]
    logging.debug("Masks layer data shape: %s", masks.shape)  # type: ignore[attr-defined]

    labs_text = {
        "string": "{nucleus}",
        "size": 10,  # tested on FOVs of 2048 (x) x 2044 (y), with ~15-20 nuclei per FOV
        "color": "black",
    }
    points_params = {
        "name": "labels",
        "size": 0,
        "text": labs_text,
        "properties": {"nucleus": nuclei_labels},
    }

    images_layer = (images, {"name": "max_proj_z"}, "image")
    masks_layer = (masks, {"name": "masks"}, "labels")
    points_layer = (nuclei_points, points_params, "points")
    return images_layer, masks_layer, points_layer  # type: ignore[return-value]


@doc(
    summary=(
        "Read (from environment variable) the image channel in which to find nuclei" " signal."
    ),
    raises=dict(
        ValueError="When the environment variable value's set to a non-integer-like",
    ),
    returns="The integer value of image channel in which to find nuclei signal",
)
def determine_nuclei_channel() -> int:  # noqa: D103
    nuc_channel = os.getenv(LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR, "")
    if nuc_channel == "":
        raise ValueError(
            "When using nuclei images with multiple channels, nuclei channel must be"
            " specified through environment variable"
            f" {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}."
        )
    try:
        return int(nuc_channel)
    except TypeError as e:
        raise ValueError(
            "Illegal nuclei channel value (from"
            f" {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}): {nuc_channel}"
        ) from e


@doc(
    summary=("Max project images z-stack along the z-axis, assumed to be axis 0 (first" " axis)."),
    parameters=dict(img="The image stack to max-project along z"),
    raises=dict(ValueError="If the given image isn't 3D (z-stack of 2D images)"),
    returns="Array of 1 dimension less than input, reducing first dimension by taking maxima along it",
)
def max_project_z(img: PixelArray) -> PixelArray:  # noqa: D103
    if len(img.shape) != 3:  # noqa: PLR2004
        raise ValueError(f"Image to max-z-project must have 3 dimensions! Got {len(img.shape)}")
    return np.max(img, axis=0)  # type: ignore[no-any-return]


def _read_csv(fp: Path) -> list[tuple[NucleusNumber, ImagePoint2D]]:
    logging.debug("Reading CSV: %s", fp)
    df = pd.read_csv(fp, index_col=0)  # noqa: PD901
    nuclei = df[
        "label"
    ]  # preserves data type of this column / field, .iterrows() seems to lose it.
    ys = df["yc"]
    xs = df["xc"]
    return [
        (NucleusNumber(n), ImagePoint2D(y=y, x=x)) for n, y, x in zip(nuclei, ys, xs, strict=False)
    ]


def _is_path_like(obj: object) -> bool:
    return isinstance(obj, str | Path)
