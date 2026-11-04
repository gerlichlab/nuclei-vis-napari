"""Tests that each Parameters section of a docstring matches the signature it documents

numpydoc_decorator checked this when it wrote the docstrings. ruff's D417 checks a
function's, but not a dataclass's, whose parameters are its fields.
"""

import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable, Iterator

import pytest

import nuclei_vis_napari

SECTION_UNDERLINE = re.compile(r"^-{3,}$")
DOCUMENTED_DEFAULT = re.compile(r"\bdefault: (.+)$")


def documented_parameters(doc: str) -> list[tuple[str, str]] | None:
    """Get each documented parameter's name and type, or None without a Parameters section."""
    lines = inspect.cleandoc(doc).splitlines()
    starts = [
        i
        for i, line in enumerate(lines[:-1])
        if line == "Parameters" and SECTION_UNDERLINE.match(lines[i + 1])
    ]
    if not starts:
        return None
    section = lines[starts[0] + 2 :]
    entries: list[tuple[str, str]] = []
    for i, line in enumerate(section):
        if i + 1 < len(section) and SECTION_UNDERLINE.match(section[i + 1]):
            break  # The heading of the next section
        # A parameter's line starts at the margin; its description is indented.
        if line and not line[0].isspace():
            names, _, kind = line.partition(":")
            entries.extend((name.strip(), kind.strip()) for name in names.split(","))
    return entries


def public_objects() -> Iterator[tuple[str, Callable[..., object]]]:
    """Get each public function and class of the plugin, and each public method of those classes."""
    for module_info in pkgutil.iter_modules(nuclei_vis_napari.__path__):
        module = importlib.import_module(f"nuclei_vis_napari.{module_info.name}")
        for name, obj in vars(module).items():
            if (
                name.startswith("_")
                or getattr(obj, "__module__", None) != module.__name__
            ):
                continue
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{name}", obj
            elif inspect.isclass(obj):
                yield f"{module.__name__}.{name}", obj
                for attr, member in vars(obj).items():
                    func = getattr(member, "__func__", member)
                    if not attr.startswith("_") and inspect.isfunction(func):
                        yield f"{module.__name__}.{name}.{attr}", func


def signature_parameters(obj: Callable[..., object]) -> list[inspect.Parameter]:
    """Get the parameters a caller passes: for a class, those of its constructor."""
    params = list(inspect.signature(obj).parameters.values())
    if not inspect.isclass(obj) and params and params[0].name in ("self", "cls"):
        params = params[1:]
    return params


DOCUMENTED = [
    (name, obj, entries)
    for name, obj in public_objects()
    if obj.__doc__ and (entries := documented_parameters(obj.__doc__)) is not None
]


def test_the_dataclasses_are_among_the_documented_objects():
    names = {name for name, _, _ in DOCUMENTED}
    assert {
        "nuclei_vis_napari.data_bundles.NucleiVisualisationData",
        "nuclei_vis_napari.napari_layer.NapariLayer",
    } <= names


@pytest.mark.parametrize(
    ("obj", "entries"),
    [(obj, entries) for _, obj, entries in DOCUMENTED],
    ids=[name for name, _, _ in DOCUMENTED],
)
def test_parameters_section_names_each_parameter_in_order(obj, entries):
    assert [name for name, _ in entries] == [
        param.name for param in signature_parameters(obj)
    ]


@pytest.mark.parametrize(
    ("obj", "entries"),
    [(obj, entries) for _, obj, entries in DOCUMENTED],
    ids=[name for name, _, _ in DOCUMENTED],
)
def test_parameters_section_gives_each_default(obj, entries):
    documented = {name: DOCUMENTED_DEFAULT.search(kind) for name, kind in entries}
    for param in signature_parameters(obj):
        match = documented.get(param.name)
        if param.default is inspect.Parameter.empty:
            assert match is None, f"{param.name} has no default"
        else:
            assert match is not None, f"{param.name} has a default, {param.default!r}"
            assert match.group(1) == repr(param.default), param.name


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ("Summary.", None),
        (
            """Summary.

            Parameters
            ----------
            a : int
                First.
            b, c : float, optional, default: 1.0
                Second and
                third.

            Returns
            -------
            int
                Sum.
            """,
            [
                ("a", "int"),
                ("b", "float, optional, default: 1.0"),
                ("c", "float, optional, default: 1.0"),
            ],
        ),
    ],
    ids=["no section", "section before another"],
)
def test_documented_parameters_reads_a_numpydoc_section(doc, expected):
    assert documented_parameters(doc) == expected
