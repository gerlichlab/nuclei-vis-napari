"""Tests for what kinds of data layout on disk will be read (or not) by this plugin"""

from pathlib import Path

import pytest

from nuclei_vis_napari import list_package_example_folders
from nuclei_vis_napari.reader import get_reader


@pytest.mark.parametrize("wrap", [str, Path])
@pytest.mark.parametrize("example_folder", list_package_example_folders())
def test_each_example_can_be_read(example_folder, wrap):
    read_data = get_reader(wrap(example_folder))
    assert callable(read_data)