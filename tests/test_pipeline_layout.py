"""Tests for reading nuclei data as published by the looptrace pipeline (B03_NUCLEI_SEGMENTATION)"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nuclei_vis_napari import get_package_examples_folder
from nuclei_vis_napari.data_bundles import NucleiDataSubfolders
from nuclei_vis_napari.reader import get_reader

LEGACY_EXAMPLE = get_package_examples_folder() / "images__example_1"
PIPELINE_FOLDER_NAME = "B03_NUCLEI_SEGMENTATION"


@pytest.fixture
def pipeline_example(tmp_path) -> Path:
    """Rearrange the bundled example the way looptrace's nuclei segmentation block publishes it."""
    root = tmp_path / PIPELINE_FOLDER_NAME
    shutil.copytree(LEGACY_EXAMPLE, root)
    (root / "_nuclear_masks_visualisation").rename(root / "nuclear_masks_visualisation")
    # The published images folder is a zarr group, so it carries a .zgroup beside the per-FOV arrays.
    (root / "nuc_images" / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    return root


def test_pipeline_layout_can_be_read(pipeline_example, wrap_path):
    read_data = get_reader(wrap_path(pipeline_example))
    assert callable(read_data), f"Expected to be able to parse data from {pipeline_example}"


def test_pipeline_layout_gives_same_layers_as_legacy_layout(pipeline_example):
    observed = get_reader(pipeline_example)(pipeline_example)  # type: ignore[misc]
    expected = get_reader(LEGACY_EXAMPLE)(LEGACY_EXAMPLE)  # type: ignore[misc]
    assert len(observed) == len(expected)
    for (obs_data, obs_params, obs_type), (exp_data, exp_params, exp_type) in zip(
        observed, expected, strict=True
    ):
        assert obs_type == exp_type
        assert obs_params == exp_params
        assert np.array_equal(obs_data, exp_data)


@pytest.mark.parametrize("subfolder", [m.value for m in NucleiDataSubfolders])
def test_pipeline_layout_missing_subfolder_means_data_cannot_be_read(pipeline_example, subfolder):
    shutil.rmtree(pipeline_example / subfolder)
    assert get_reader(pipeline_example) is None


def test_unprefixed_centers_folder_is_preferred_when_both_are_present(pipeline_example):
    shutil.copytree(
        pipeline_example / "nuclear_masks_visualisation",
        pipeline_example / "_nuclear_masks_visualisation",
    )
    assert (
        NucleiDataSubfolders.CENTERS.relpath(pipeline_example)
        == pipeline_example / "nuclear_masks_visualisation"
    )
