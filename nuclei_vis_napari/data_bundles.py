"""Nuclei datta access and bundling for layer construction"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from gertils.geometry import ImagePoint2D
from gertils.pathtools import find_single_path_by_fov
from gertils.types import FieldOfViewFrom1, NucleusNumber, PixelArray
from gertils.zarr_tools import read_zarr
from numpydoc_decorator import doc  # type: ignore[import-untyped]

from .type_aliases import PathLike

# Environment variable for finding nuclei channel if needed.
LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR = "LOOPTRACE_NAPARI_NUCLEI_CHANNEL"


class NuclusDataKey(Enum):
    """Columns of CSV or similar 'key'-like to store coordinates of a point"""

    Label = "label"
    CenterY = "yc"
    CenterX = "xc"


class NucleiDataSubfolders(Enum):
    """The subfolders of nuclei data"""

    IMAGES = "nuc_images"
    MASKS = "nuc_masks"
    CENTERS = "nuclear_masks_visualisation"

    @classmethod
    def paths_by_fov(cls, p: PathLike) -> dict[str, dict[FieldOfViewFrom1, Path]]:
        """Per subfolder, the data file for each field of view found in it.

        Filenames only: no array is opened, so this is cheap enough for
        ``get_reader`` to consult before it claims it can read a folder.
        """
        return {
            cls.IMAGES.value: find_single_path_by_fov(cls.IMAGES.relpath(p), extension=".zarr"),
            cls.MASKS.value: find_single_path_by_fov(cls.MASKS.relpath(p), extension=".zarr"),
            cls.CENTERS.value: find_single_path_by_fov(
                cls.CENTERS.relpath(p), extension=".nuclear_masks.csv"
            ),
        }

    @staticmethod
    def shared_fields_of_view(
        by_fov: Mapping[str, Mapping[FieldOfViewFrom1, Path]],
    ) -> set[FieldOfViewFrom1]:
        """The fields of view for which ALL THREE subfolders have data.

        A field of view present in only some of them cannot be displayed -- the
        image, the mask and the centroids are one layer stack -- so the usable
        set is the intersection, and it can be empty even when all three folders
        exist and are full. That happens whenever the folders describe different
        subsets: a run restricted with `selected_fovs`, or three folders
        assembled by hand from different analyses.

        Takes the result of ``paths_by_fov`` rather than a path, so that a caller
        needing both it and the per-subfolder counts pays for one scan, not two.
        """
        return set.intersection(*(set(paths) for paths in by_fov.values()))

    @classmethod
    def read_all_from_root(cls, p: PathLike) -> dict[FieldOfViewFrom1, "NucleiVisualisationData"]:
        """For each field of view in the given folder, deter,ome the nuclei data paths."""
        by_fov = cls.paths_by_fov(p)
        image_paths = by_fov[cls.IMAGES.value]
        masks_paths = by_fov[cls.MASKS.value]
        centers_paths = by_fov[cls.CENTERS.value]
        fields_of_view = cls.shared_fields_of_view(by_fov)
        logging.debug("Image paths count: %d", len(image_paths))
        logging.debug("Masks paths count: %d", len(masks_paths))
        logging.debug("Centers paths count: %d", len(centers_paths))
        bundles: dict[FieldOfViewFrom1, NucleiVisualisationData] = {}
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

    @property
    def names(self) -> tuple[str, ...]:
        """The accepted names for this subfolder, in order of preference."""
        # Older looptrace wrote the centers folder with an underscore prefix, into the images folder;
        # the pipeline now publishes it without the prefix, in B03_NUCLEI_SEGMENTATION.
        if self is NucleiDataSubfolders.CENTERS:
            return (self.value, "_" + self.value)
        return (self.value,)

    def is_present_within(self, p: PathLike) -> bool:
        """Determine whether this subfolder is directly within given folder."""
        return self.relpath(p).is_dir()

    def relpath(self, p: PathLike) -> Path:
        """Give this subfolder's path within the given parent -- the one that EXISTS, where more than one name is accepted.

        Not a pure join: the centers folder has two accepted spellings, so which
        path is correct is a question about the filesystem, and this probes it.
        When none of the accepted names is present -- which is the case worth
        reporting on -- the preferred spelling is returned, so the caller has a
        path to name in its message even though nothing is there.
        """
        candidates = [Path(p) / name for name in self.names]
        return next((c for c in candidates if c.is_dir()), candidates[0])


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
@dataclass(frozen=True, kw_only=True, eq=False)
class NucleiVisualisationData:  # noqa: D101
    image: PixelArray
    masks: PixelArray
    centers: list[tuple[NucleusNumber, ImagePoint2D]]

    def __eq__(self, value: object) -> bool:
        return (
            type(value) is type(self)
            and np.array_equal(self.image, value.image)  # type: ignore[attr-defined]
            and np.array_equal(self.masks, value.masks)  # type: ignore[attr-defined]
            and self.centers == value.centers  # type: ignore[attr-defined]
        )

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
                    "Multiple channels in nuclei image; attempting to determine which to use"
                )
                nuc_channel: int = _determine_nuclei_channel()
                if nuc_channel < 0 or nuc_channel >= self.image.shape[0]:
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
                f"Need 2D image for nuclear masks! Got {len(self.masks.shape)}: {self.masks.shape}"
            )


@doc(
    summary=("Read (from environment variable) the image channel in which to find nuclei signal."),
    raises=dict(
        ValueError="When the environment variable value's set to a non-integer-like",
    ),
    returns="The integer value of image channel in which to find nuclei signal",
)
def _determine_nuclei_channel() -> int:
    nuc_channel = os.getenv(LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR)
    if nuc_channel is None or nuc_channel == "":
        raise ValueError(
            "When using nuclei images with multiple channels, nuclei channel must be"
            " specified through environment variable"
            f" {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}."
        )
    try:
        return int(nuc_channel)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Illegal nuclei channel value (from"
            f" {LOOPTRACE_NAPARI_NUCLEI_CHANNEL_ENV_VAR}): {nuc_channel}"
        ) from e


@doc(
    summary=("Max project images z-stack along the z-axis, assumed to be axis 0 (first axis)."),
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

    # preserves data type of this column / field, .iterrows() seems to lose it.
    nuclei = df[NuclusDataKey.Label.value]
    ys = df[NuclusDataKey.CenterY.value]
    xs = df[NuclusDataKey.CenterX.value]

    return [
        (NucleusNumber(n), ImagePoint2D(y=y, x=x)) for n, y, x in zip(nuclei, ys, xs, strict=False)
    ]
