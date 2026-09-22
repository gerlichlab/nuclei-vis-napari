"""Tests for reading nuclei data as published by the looptrace pipeline (B03_NUCLEI_SEGMENTATION)"""

import json
import logging
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
    """Which folder wins is only interesting if the centroids drawn differ.

    Asserting on `relpath` alone would keep passing if the read stopped
    consulting it. So the centroids are read first, with only the unprefixed
    folder present; then a legacy folder is added carrying DIFFERENT centroids,
    and the read must be unchanged. Shifting y by 500 makes a wrong choice
    impossible to miss, rather than a rounding difference.
    """

    def centroids() -> list[tuple[float, float]]:
        *_, (points, _, _) = get_reader(pipeline_example)(pipeline_example)  # type: ignore[misc]
        return [(y, x) for _, y, x in points]

    expected = centroids()
    assert expected, "No centroid was read at all; the test would prove nothing"

    legacy = pipeline_example / "_nuclear_masks_visualisation"
    shutil.copytree(pipeline_example / "nuclear_masks_visualisation", legacy)
    for csv_path in legacy.glob("*.nuclear_masks.csv"):
        header, *body = csv_path.read_text().splitlines()
        shifted = []
        for row in body:
            index, label, yc, xc, *rest = row.split(",")
            shifted.append(",".join([index, label, str(float(yc) + 500), xc, *rest]))
        csv_path.write_text("\n".join([header, *shifted]) + "\n")

    assert (
        NucleiDataSubfolders.CENTERS.relpath(pipeline_example)
        == pipeline_example / "nuclear_masks_visualisation"
    )
    assert centroids() == expected


def test_a_stray_non_fov_entry_in_the_images_folder_is_ignored(pipeline_example, wrap_path):
    """Whatever else sits beside the per-FOV stores must not stop the read.

    Nothing guarantees the published folder holds only `<fov>.zarr` entries --
    zarr tooling writes group metadata, filesystems leave `.DS_Store`, a reader
    may drop a cache file. Discovery selects by parsing a field of view out of
    each name, so anything unparsable is skipped; this pins that, rather than
    leaving it to be rediscovered by whoever first sees a folder with a stray
    file in it.
    """
    (pipeline_example / "nuc_images" / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (pipeline_example / "nuc_images" / ".DS_Store").write_bytes(b"\x00")
    assert callable(get_reader(wrap_path(pipeline_example)))


def test_two_names_for_one_field_of_view_is_declined_not_raised(
    pipeline_example, caplog, wrap_path
):
    """Declining must not THROW, whatever the folder holds.

    `get_reader` runs during napari's reader SELECTION, so an exception there is
    a crash in the GUI rather than a decline that hands the drop to the next
    plugin. Discovery raises when two filenames parse to the same field of view,
    which "P1.zarr" beside "P0001.zarr" does -- both parse to 1. Checking the
    field of view eagerly, rather than leaving it to the parse, moved that raise
    into the selection path, so it has to be caught there.
    """
    images = pipeline_example / "nuc_images"
    original = next(images.glob("P0*.zarr"))
    shutil.copytree(original, images / "P1.zarr")
    with caplog.at_level(logging.DEBUG):
        assert get_reader(wrap_path(pipeline_example)) is None
    assert "Cannot resolve fields of view" in caplog.text


def test_a_folder_predating_published_images_says_so(pipeline_example, caplog, wrap_path):
    """The commonest refusal for a while, so it should name its own cause.

    Every analysis folder produced before looptrace published `nuc_images` looks
    exactly like this: two correct subfolders and one absent. Listing all three
    expected paths made that read like a malformed folder rather than an old one,
    and said nothing about the way out -- which is to resume the run, since B03's
    cached task output republishes without recomputation.
    """
    shutil.rmtree(pipeline_example / "nuc_images")
    with caplog.at_level(logging.DEBUG):
        assert get_reader(wrap_path(pipeline_example)) is None
    assert "Not a folder: nuc_images" in caplog.text
    assert "resuming the run republishes it" in caplog.text
    assert "nuc_masks" not in caplog.text
