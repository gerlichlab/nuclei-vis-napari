"""Tests that napari actually routes a dropped folder to this plugin.

The other reader tests call `get_reader` directly, which proves the function
accepts a folder -- not that napari ever offers this plugin for one. That
routing is the claim the documentation makes ("the name of the folder to drop
doesn't matter"), and it rests on npe2 treating directories specially: a reader
declaring `accepts_directories` is indexed under an empty pattern, and a
directory path yields exactly those readers, `filename_patterns` unconsulted.

Nothing in the plugin enforces that, so a manifest edit -- the likeliest change
by someone who has not read this -- could break the claim with every other test
still green.
"""

from pathlib import Path

import npe2
import pytest

from nuclei_vis_napari import get_package_examples_folder

READER_COMMAND = "nuclei-vis-napari.read_nuclei"
MANIFEST = Path(__file__).resolve().parents[1] / "nuclei_vis_napari" / "napari.yaml"


@pytest.fixture
def plugin_manager():
    """A manager with just this plugin's manifest registered."""
    pm = npe2.PluginManager()
    pm.register(npe2.PluginManifest.from_file(MANIFEST))
    return pm


def _reader_commands_for(pm, path) -> list[str]:
    return [reader.command for reader in pm.iter_compatible_readers([str(path)])]


def test_a_dropped_example_folder_reaches_this_plugin(plugin_manager):
    """The user-facing claim: drop the folder, get this reader."""
    folder = get_package_examples_folder() / "images__example_1"
    assert READER_COMMAND in _reader_commands_for(plugin_manager, folder)


def test_a_folder_whose_name_does_not_match_the_pattern_still_reaches_it(plugin_manager, tmp_path):
    """The documented promise that the folder's NAME is irrelevant.

    `filename_patterns` still says 'images*'; this is what makes that vestigial
    rather than a constraint, and what would fail first if the manifest lost
    `accepts_directories`.
    """
    folder = tmp_path / "B03_NUCLEI_SEGMENTATION"
    folder.mkdir()
    assert READER_COMMAND in _reader_commands_for(plugin_manager, folder)


def test_a_plain_file_does_not_reach_it(plugin_manager, tmp_path):
    """The reason not to widen the pattern to '*'.

    This reader declines non-directories, so being offered for arbitrary files
    would only add a useless entry to napari's reader-choice dialog.
    """
    fp = tmp_path / "some_acquisition.txt"
    fp.write_text("not nuclei data")
    assert READER_COMMAND not in _reader_commands_for(plugin_manager, fp)
