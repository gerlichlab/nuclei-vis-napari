"""Tests for the accuracy of the layers' construction"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gertils.geometry import ImagePoint2D
from gertils.types import FieldOfViewFrom1, NucleusNumber, PixelArray
from gertils.zarr_tools import read_zarr

from nuclei_vis_napari import get_package_examples_folder
from nuclei_vis_napari.data_bundles import (
    NucleiDataSubfolders,
    NucleiVisualisationData,
    NuclusDataKey,
)
from nuclei_vis_napari.reader import get_reader

EXPECTED_FIELDS_OF_VIEW: list[FieldOfViewFrom1] = [FieldOfViewFrom1(raw_fov) for raw_fov in [1, 2]]


def test_read_all_from_root_is_correct_on_example(wrap_path):
    images_folder = get_images_folder()
    assert images_folder.is_dir(), f"Images folder isn't an extant directory: {images_folder}"
    masks_folder = get_masks_folder()
    assert masks_folder.is_dir(), f"Masks folder isn't an extant directory: {masks_folder}"
    read_centers_file = lambda fov: [  # noqa: E731
        (
            NucleusNumber(int(row[NuclusDataKey.Label.value])),
            ImagePoint2D(y=row[NuclusDataKey.CenterY.value], x=row[NuclusDataKey.CenterX.value]),
        )
        for _, row in pd.read_csv(get_centers_file(fov), index_col=0).iterrows()
    ]

    expected = {
        fov: NucleiVisualisationData(
            image=read_zarr(images_folder / (get_fov_name(fov) + ".zarr")),
            masks=read_zarr(masks_folder / (get_fov_name(fov) + ".zarr")),
            centers=read_centers_file(fov),
        )
        for fov in EXPECTED_FIELDS_OF_VIEW
    }

    observed = NucleiDataSubfolders.read_all_from_root(wrap_path(get_example_folder()))

    assert observed == expected


def test_layers_are_correct_on_example():
    """This is a big integration-like test."""  # noqa: D404
    example = get_example_folder()
    assert example.is_dir(), f"Example isn't an extant folder: {example}"
    read_data = get_reader(example)
    assert callable(read_data), f"Example ({example}) didn't return a callable reader"
    try:
        (
            (obs_image_data, obs_image_layer_params, obs_image_layer_name),
            (obs_masks_layer_data, obs_masks_layer_params, obs_masks_layer_name),
            (obs_centers_layer_data, obs_centers_layer_params, obs_centers_layer_name),
        ) = read_data(example)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to read example ({example}) or to unpack result. Got error: {e}")

    def read_points_data(fp: Path) -> tuple[list[float | np.float64], list[int]]:
        coordinates: list[list[float | np.float64]] = []
        labels: list[int] = []
        df = pd.read_csv(fp, index_col=0)  # noqa: PD901
        for _, row in df.iterrows():
            coordinates.append([row[NuclusDataKey.CenterY.value], row[NuclusDataKey.CenterX.value]])
            labels.append(row[NuclusDataKey.Label.value])
        return coordinates, labels

    # Validate the image layer.
    exp_image_data = np.stack([read_image_file(fov) for fov in EXPECTED_FIELDS_OF_VIEW])
    assert obs_image_layer_name == "image"
    assert obs_image_layer_params == {"name": "max_proj_z"}
    assert obs_image_data.shape == exp_image_data.shape
    assert np.array_equal(obs_image_data, exp_image_data)

    # Validate the masks layer.
    exp_masks_layer_data = np.stack([read_masks_file(fov) for fov in EXPECTED_FIELDS_OF_VIEW])
    assert obs_masks_layer_name == "labels"
    assert obs_masks_layer_params == {"name": "masks"}
    assert obs_masks_layer_data.shape == exp_masks_layer_data.shape
    assert np.array_equal(obs_masks_layer_data, exp_masks_layer_data)

    # Validate the centroids (points) layer.
    exp_centers_layer_data = []
    exp_nuclei_labels = []
    for fov in EXPECTED_FIELDS_OF_VIEW:
        centers, labels = read_points_data(get_centers_file(fov))
        exp_centers_layer_data.extend([fov.get - 1, *coords] for coords in centers)
        exp_nuclei_labels.extend(labels)
    assert obs_centers_layer_name == "points"
    # The points layer should be called 'labels', have infinitesimally small points, and nuclei labels as texts
    assert obs_centers_layer_params == {
        "name": "labels",
        "size": 0,
        "text": {"string": "{nucleus}", "size": 10, "color": "black"},
        "properties": {"nucleus": [int(n) for n in exp_nuclei_labels]},
    }
    assert np.array_equal(obs_centers_layer_data, exp_centers_layer_data)


def get_example_folder() -> Path:
    return get_package_examples_folder().joinpath("images__example_1")


def get_centers_file(fov: FieldOfViewFrom1) -> Path:
    return get_centers_folder() / f"{get_fov_name(fov)}.nuclear_masks.csv"


def get_centers_folder() -> Path:
    return get_example_folder() / "_nuclear_masks_visualisation"


def get_fov_name(fov: FieldOfViewFrom1) -> str:
    return f"P{str(fov.get).zfill(4)}"


def get_images_folder() -> Path:
    return get_example_folder() / "nuc_images"


def get_image_file(fov: FieldOfViewFrom1) -> Path:
    return get_images_folder() / f"P{str(fov.get).zfill(4)}.zarr"


def get_masks_folder() -> Path:
    return get_example_folder() / "nuc_masks"


def get_masks_file(fov: FieldOfViewFrom1) -> Path:
    return get_masks_folder() / f"P{str(fov.get).zfill(4)}.zarr"


def read_image_file(fov: FieldOfViewFrom1) -> PixelArray:
    fp: Path = get_image_file(fov)
    data = read_zarr(fp)
    return data.reshape(*(d for d in data.shape if d != 1))


def read_masks_file(fov: FieldOfViewFrom1) -> PixelArray:
    fp: Path = get_masks_file(fov)
    data = read_zarr(fp)
    return data.reshape(*(d for d in data.shape if d != 1))
