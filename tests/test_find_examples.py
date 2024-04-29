"""Tests for accessing this package's examples/resources"""

from nuclei_vis_napari import list_package_example_folders


def test_example_folder_count():
    expected_folder_names = ["images__example_1"]
    observed_folder_names = [f.name for f in list_package_example_folders()]
    assert observed_folder_names == expected_folder_names
