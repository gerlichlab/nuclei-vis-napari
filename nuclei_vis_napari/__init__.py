"""A plugin for napari which allows visualisation of nuclei and nuclear regions"""

__version__ = "0.2dev"

import sys

if sys.version_info < (3, 11):
    import importlib_resources  # pragma: no cover
else:
    import importlib.resources as importlib_resources  # pragma: no cover
from pathlib import Path

from numpydoc_decorator import doc  # type: ignore[import-untyped]


_PACKAGE_NAME = package = Path(__file__).parent.name


@doc(summary="Get the path to this package's examples folder.")
def get_package_examples_folder() -> Path:  # noqa: D103
    return _get_package_resources().joinpath("examples")  # type: ignore[no-any-return]


@doc(summary="Get the hook with which to access resources bundled with this package.")
def _get_package_resources():  # type: ignore[no-untyped-def]  # noqa: ANN202
    return importlib_resources.files(_PACKAGE_NAME)


@doc(summary="List the files bundles as examples with this package.")
def list_package_example_folders() -> list[Path]:  # noqa: D103
    return [path for path in get_package_examples_folder().iterdir() if path.is_dir()]
