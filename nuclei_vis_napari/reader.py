"""Reading looptrace-written, ZARR-stored data"""

import logging
import os
from collections.abc import Callable, Mapping
from operator import itemgetter
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import numpy as np
from gertils.types import FieldOfViewFrom1, PixelArray
from numpydoc_decorator import doc  # type: ignore[import-untyped]

from .data_bundles import NucleiDataSubfolders, NucleiVisualisationData
from .napari_layer import LayerData, LayerParams, NapariLayer, NapariLayerType
from .type_aliases import PathOrPaths

# Specific layer types
ImageLayer = tuple[LayerData, LayerParams, Literal["image"]]
MasksLayer = tuple[LayerData, LayerParams, Literal["labels"]]
CentroidsLayer = tuple[LayerData, LayerParams, Literal["points"]]
FullDataLayer = ImageLayer | MasksLayer | CentroidsLayer

# Other type aliases
Reader = Callable[[PathOrPaths], list[FullDataLayer]]


class _Refusal(NamedTuple):
    """A reason not to read a folder, and how loudly to say it.

    Most refusals are DEBUG, and must be: napari calls ``get_reader`` on every
    plugin for every drop, so declining someone else's data has to be silent.

    A refusal reached AFTER all three subfolders were found is different. That
    folder looks exactly like nuclei data, so it is almost certainly the one the
    user meant, and napari reports only "no reader available" -- which hides a
    cause they could act on (a permission bit, a duplicated field of view) behind
    a message about the plugin. Those are WARNING.
    """

    why: str
    level: int = logging.DEBUG


def _why_not_readable(path: Path) -> Optional[_Refusal]:
    """Why this folder cannot be read, or None if it can.

    Kept out of ``get_reader`` so each refusal is one return of a message rather
    than a log-and-return pair, and so the folder rules can be read -- and
    tested -- without napari's reader protocol in the way.
    """
    # Each of the subpaths to parse must be an extant folder.
    missing = [
        member.value for member in NucleiDataSubfolders if not member.is_present_within(path)
    ]
    if missing:
        # Name what is missing, not all three: for a while the common case will
        # be an analysis folder produced before looptrace published nuc_images,
        # where every other subfolder is present and correct. Listing all three
        # made that read like a malformed folder rather than an old one.
        why = f"Not a folder: {', '.join(missing)}, under {path}"
        if NucleiDataSubfolders.IMAGES.value in missing:
            why += (
                f". If this is a looptrace analysis folder,"
                f" {NucleiDataSubfolders.IMAGES.value} is published only by newer"
                " versions of the pipeline; resuming the run republishes it from"
                " cached task output, or use an analysis from a newer run"
            )
        return _Refusal(why)

    # ...and they must describe at least one field of view IN COMMON. Checked
    # here rather than left to the parse, because returning a reader is a claim
    # that the folder can be read: without this, napari accepted the drop and
    # then died in np.stack on an empty list, which names nothing the user can
    # act on. Filenames only, so this costs one listing per subfolder and opens
    # no array; the result is reused for the check and for its message.
    by_fov = NucleiDataSubfolders.paths_by_fov(path)
    if not NucleiDataSubfolders.shared_fields_of_view(by_fov):
        counts = {name: len(paths) for name, paths in by_fov.items()}
        return _Refusal(
            "No field of view is present in all three subfolders, so there is"
            f" nothing to display; data files found per subfolder: {counts}",
            logging.WARNING,
        )
    return None


@doc(
    summary=(
        "This is the main hook required by napari / napari plugins to provide a Reader plugin."
    ),
    parameters=dict(path="Path to the folder (with proper substructure) with nuclei data to view"),
    returns="A callable that accepts a list of paths and returns the layers for nuclei vis",
)
def get_reader(path: PathOrPaths) -> Optional[Reader]:  # noqa: D103
    def do_not_parse(why: str, *, level: int = logging.DEBUG) -> None:
        logging.log(level, "%s, cannot read looptrace nuclei visualisation data", why)

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

    # Declining must not THROW. Everything above is a decline; the folder checks
    # touch the filesystem, and napari calls this during reader SELECTION, where
    # an exception is a crash in the GUI rather than a decline that hands the
    # drop to the next plugin. Two shapes reach here, both meaning "this folder
    # cannot be read" rather than "this code is wrong":
    #   * RuntimeError, when two filenames parse to one field of view -- P1.zarr
    #     beside P0001.zarr, since both parse to 1;
    #   * OSError, when the filesystem will not say what a folder holds. On the
    #     group share that is likelier than it sounds: a subfolder present but
    #     unreadable (EACCES), or a stale NFS handle (ESTALE) on a flaky mount.
    # The guard sits here, around the whole check, because BOTH the listing and
    # the earlier is_dir probes can raise: pathlib re-raises EACCES rather than
    # reporting "not a directory", so a folder that is readable but not
    # traversable throws before any listing is attempted.
    try:
        refusal = _why_not_readable(path)  # type: ignore[arg-type]
    except (RuntimeError, OSError) as e:
        refusal = _Refusal(f"Cannot examine the contents of {path}: {e}", logging.WARNING)
    if refusal is not None:
        do_not_parse(refusal.why, level=refusal.level)
        return None

    def parse(root: PathOrPaths) -> list[FullDataLayer]:
        if not _is_path_like(root):
            # Impossibility should be assured by the above logic, so don't test for coverage.
            raise TypeError(
                f"Non-path-like as nuclei data: {type(root).__name__}"
            )  # pragma: no cover
        # Ignore type warning here b/c we've conditionally proven arg type is correct.
        data_by_fov = NucleiDataSubfolders.read_all_from_root(root)  # type: ignore[arg-type]
        image_layer, masks_layer, centroids_layer = build_layers(data_by_fov)
        return [image_layer.as_image, masks_layer.as_labels, centroids_layer.as_points]

    return parse


@doc(
    summary="Build the multiple layers (image, masks, points) to look at nuclei in napari.",
    parameters=dict(bundles="Mapping from FOV to the bundle of data needed to visualise its data"),
    raises=dict(RuntimeError="If images and masks aren't entirely uniform w.r.t. shape"),
    returns="The layers needed to visualise nuclei across all FOVs",
)
def build_layers(  # noqa: D103
    bundles: Mapping[FieldOfViewFrom1, NucleiVisualisationData],
) -> tuple[NapariLayer, NapariLayer, NapariLayer]:
    if not bundles:
        # get_reader refuses this case, so reaching here is a programming error;
        # it still gets its own message rather than numpy's "need at least one
        # array to stack", which says nothing about fields of view.
        raise ValueError(
            "Cannot build layers from no data bundles: no field of view had an"
            " image, a mask and centroids together"
        )
    images = []
    masks = []
    nuclei_points = []
    nuclei_labels = []
    image_shape: tuple[int, ...]
    for i, (_, visdata) in enumerate(sorted(bundles.items(), key=itemgetter(0))):
        img = visdata.image
        if i == 0:
            image_shape = img.shape
        if img.shape != image_shape:
            raise RuntimeError(
                f"Image shape for FOV {i} doesn't match previous: {img.shape} != {image_shape}"
            )
        if visdata.masks.shape != image_shape:
            raise RuntimeError(
                f"Masks shape for FOV {i} doesn't match previous: {visdata.masks.shape} != {image_shape}"
            )
        images.append(visdata.image)
        masks.append(visdata.masks)
        for nuc, pt in visdata.centers:
            nuclei_points.append([i, pt.y, pt.x])
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

    images_layer = NapariLayer(
        data=images,
        parameters={"name": "max_proj_z"},
        get_type=NapariLayerType.Image,
    )
    masks_layer = NapariLayer(
        data=masks,
        parameters={"name": "masks"},
        get_type=NapariLayerType.Labels,
    )
    points_layer = NapariLayer(
        data=nuclei_points,
        parameters=points_params,
        get_type=NapariLayerType.Points,
    )
    return images_layer, masks_layer, points_layer


def _is_path_like(obj: object) -> bool:
    return isinstance(obj, str | Path)
