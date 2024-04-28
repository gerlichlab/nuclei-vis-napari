"""Test that the plugin will be able to be used by Napari."""

import npe2.cli
import pytest
import typer


def test_plugin_package_manifest_is_valid():
    try:
        npe2.cli.validate("nuclei-vis-napari")
    except typer.Exit as e:
        pytest.fail(f"Manifest validation failed: {e}")
