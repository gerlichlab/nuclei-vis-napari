"""A plugin for napari which allows visualisation of nuclei and nuclear regions"""

__version__ = "0.5.0"

import importlib.resources as importlib_resources
from pathlib import Path
from typing import cast

_PACKAGE_NAME = package = Path(__file__).parent.name


def get_package_examples_folder() -> Path:
    """Get the path to this package's examples folder.

    Returns
    -------
    Path

    """
    return _get_package_resources().joinpath("examples")


def _get_package_resources() -> Path:
    """Get the hook with which to access resources bundled with this package.

    Returns
    -------
    Path

    """
    # files() gives a Path for a package installed as files, as this one is.
    return cast("Path", importlib_resources.files(_PACKAGE_NAME))


def list_package_example_folders() -> list[Path]:
    """List the files bundles as examples with this package.

    Returns
    -------
    list of Path

    """
    return [path for path in get_package_examples_folder().iterdir() if path.is_dir()]
