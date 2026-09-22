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
    """Rearrange the bundled example the way looptrace's nuclei segmentation block publishes it.

    `nuc_images/` is a plain directory of per-field-of-view stores, which is what
    the pipeline writes: checked against four published analysis folders,
    including the first produced by the code that added this output. It is NOT a
    zarr group and carries no `.zgroup`; a fixture that wrote one would be
    testing a layout the pipeline does not produce, and would keep passing if the
    reader came to depend on group metadata that real data lacks.
    """
    root = tmp_path / PIPELINE_FOLDER_NAME
    shutil.copytree(LEGACY_EXAMPLE, root)
    (root / "_nuclear_masks_visualisation").rename(root / "nuclear_masks_visualisation")
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


def test_a_stray_non_fov_entry_in_the_images_folder_is_ignored(pipeline_example, wrap_path):
    """Whatever else sits beside the per-FOV stores must not stop the read.

    Nothing guarantees the published folder holds only `<fov>.zarr` entries --
    zarr tooling writes group metadata, filesystems leave `.DS_Store`, a reader
    may drop a cache file. Discovery selects by parsing a field of view out of
    each name, so anything unparseable is skipped; this pins that, rather than
    leaving it to be rediscovered by whoever first sees a folder with a stray
    file in it.
    """
    (pipeline_example / "nuc_images" / ".zgroup").write_text(
        json.dumps({"zarr_format": 2})
    )
    (pipeline_example / "nuc_images" / ".DS_Store").write_bytes(b"\x00")
    assert callable(get_reader(wrap_path(pipeline_example)))
