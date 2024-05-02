"""Reading looptrace-written, ZARR-stored data"""

import logging
import os
from collections.abc import Callable, Mapping
from operator import itemgetter
from pathlib import Path
from typing import Literal, Optional

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


@doc(
    summary=(
        "This is the main hook required by napari / napari plugins to provide a Reader" " plugin."
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

    # Each of the subpaths to parse must be extant folder.
    if not NucleiDataSubfolders.all_present_within(path):
        do_not_parse(
            "At least one subpath to parse isn't a folder!"
            f" {NucleiDataSubfolders.relpaths(path)}."
        )
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
