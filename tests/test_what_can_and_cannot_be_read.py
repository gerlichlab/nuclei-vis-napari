"""Tests for what kinds of data layout on disk will be read (or not) by this plugin"""

import itertools
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import hypothesis as hyp
import pytest
from hypothesis import strategies as st

from nuclei_vis_napari import list_package_example_folders
from nuclei_vis_napari.reader import get_reader

EXAMPLE_FOLDERS = list_package_example_folders()


@dataclass(kw_only=True)
class FileOmissionSpecification:
    folder: Path
    omissions: list[str]


@pytest.mark.parametrize("example_folder", EXAMPLE_FOLDERS)
def test_each_example_can_be_read(example_folder, wrap_path):
    read_data = get_reader(wrap_path(example_folder))
    assert callable(
        read_data
    ), f"Expected to be able to parse data from {example_folder} but couldn't"


@pytest.mark.parametrize("example_folder", EXAMPLE_FOLDERS)
def test_extra_files_and_folders_are_ignored(tmp_path, wrap_path, example_folder):
    """Extra files should be ignored -- whether above or beside the folder to read."""
    # First, copy the valid example to the current test temp folder.
    data_home = tmp_path / example_folder.name
    assert not data_home.exists()
    shutil.copytree(example_folder, data_home, dirs_exist_ok=True)
    assert data_home.is_dir(), f"Path to read isn't an extant directory: {data_home}"

    # Create other folders into which to place the example files.
    extra_parent = tmp_path / "extra_parent"
    extra_parent.mkdir()
    extra_child = data_home / "extra_child"
    extra_child.mkdir()

    # Place each example file in the dummy folders, and in the temp folder.
    for fp in data_home.iterdir():
        _omni_copy(fp, extra_parent / fp.name)
        _omni_copy(fp, extra_child / fp.name)
        _omni_copy(fp, tmp_path / fp.name)

    # Get the data reader and check that it's callable.
    read_data = get_reader(wrap_path(data_home))
    assert callable(read_data), f"Expected to be able to parse data from {data_home} but couldn't"


@pytest.mark.parametrize(
    "omit_spec",
    [
        FileOmissionSpecification(folder=folder, omissions=[p.name for p in paths])
        for folder in EXAMPLE_FOLDERS
        for k in range(1 + sum(1 for _ in folder.iterdir()))
        for paths in itertools.combinations(folder.iterdir(), k)
    ],
)
def test_any_missing_file_means_data_cannot_be_read(tmp_path, wrap_path, omit_spec):
    assert list(tmp_path.iterdir()) == [], "Temp folder isn't empty!"
    candidate_paths = list(omit_spec.folder.iterdir())
    num_omit = len(omit_spec.omissions)
    for p in candidate_paths:
        dst = p.name
        if dst in omit_spec.omissions:
            continue
        _omni_copy(p, tmp_path / dst)
    assert sum(1 for _ in tmp_path.iterdir()) == len(candidate_paths) - num_omit
    read_data = get_reader(wrap_path(tmp_path))
    if num_omit == 0:
        assert callable(read_data), "Failed to get callable reader even when omitting nothing!"
    else:
        assert (
            read_data is None
        ), f"Got non-null reader when omitting {num_omit} required element(s): {', '.join(omit_spec.omissions)}"


@pytest.mark.parametrize(
    "example_path", [p for folder in EXAMPLE_FOLDERS for p in folder.iterdir()]
)
def test_required_elements_cannot_be_read_individually(example_path, wrap_path, caplog):
    arg = wrap_path(example_path)
    with caplog.at_level(logging.DEBUG):
        read_data = get_reader(arg)
    assert read_data is None
    obs_msg = list(caplog.records)[-1].message
    # Given subfolder will have been interpreted as main folder.
    exp_msg = "At least one subpath to parse isn't a folder!"
    assert obs_msg.startswith(exp_msg)


@pytest.mark.parametrize("example_paths", [list(folder.iterdir()) for folder in EXAMPLE_FOLDERS])
def test_required_elements_cannot_be_read_as_list(example_paths):
    assert isinstance(example_paths, list)
    assert get_reader(example_paths) is None


@pytest.mark.parametrize("example_folder", EXAMPLE_FOLDERS)
def test_required_cannot_read_example_folder_if_wrapped_in_list(example_folder, wrap_path):
    assert get_reader([wrap_path(example_folder)]) is None


@hyp.given(bad_arg=st.one_of(st.none(), st.integers(), st.floats(), st.uuids()))
@hyp.settings(
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink),
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture,),
)
def test_cannot_parse_non_path_like(bad_arg, caplog):
    with caplog.at_level(logging.DEBUG):
        read_data = get_reader(bad_arg)
    assert read_data is None
    obs_msg = list(caplog.records)[-1].message
    exp_msg = f"Not a path-like: {bad_arg}, cannot read looptrace nuclei visualisation data"
    assert obs_msg == exp_msg


def test_cannot_parse_non_extant_folder(tmp_path, wrap_path, caplog):
    arg = wrap_path(tmp_path / "not_extant")
    with caplog.at_level(logging.DEBUG):
        read_data = get_reader(arg)
    assert read_data is None
    obs_msg = list(caplog.records)[-1].message
    exp_msg = f"Not an extant directory: {arg}, cannot read looptrace nuclei visualisation data"
    assert obs_msg == exp_msg


def test_cannot_parse_extant_file(tmp_path, wrap_path, caplog):
    fp = tmp_path / "P0001.zarr"
    fp.touch()
    assert fp.is_file()
    arg = wrap_path(fp)
    with caplog.at_level(logging.DEBUG):
        read_data = get_reader(arg)
    assert read_data is None
    obs_msg = list(caplog.records)[-1].message
    exp_msg = f"Not an extant directory: {arg}, cannot read looptrace nuclei visualisation data"
    assert obs_msg == exp_msg


def _omni_copy(src: Path, dst: Path) -> Path:
    cp = shutil.copytree if src.is_dir() else shutil.copy
    return cp(src, dst)
