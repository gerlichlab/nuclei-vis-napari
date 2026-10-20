"""A plugin for napari which allows visualisation of nuclei and nuclear regions"""

__version__ = "0.3.0"

import importlib.resources as importlib_resources
from pathlib import Path
from typing import cast

from numpydoc_decorator import doc

_PACKAGE_NAME = package = Path(__file__).parent.name


@doc(summary="Get the path to this package's examples folder.")
def get_package_examples_folder() -> Path:  # noqa: D103
    return _get_package_resources().joinpath("examples")


@doc(summary="Get the hook with which to access resources bundled with this package.")
def _get_package_resources() -> Path:
    # files() gives a Path for a package installed as files, as this one is.
    return cast("Path", importlib_resources.files(_PACKAGE_NAME))


@doc(summary="List the files bundles as examples with this package.")
def list_package_example_folders() -> list[Path]:  # noqa: D103
    return [path for path in get_package_examples_folder().iterdir() if path.is_dir()]
