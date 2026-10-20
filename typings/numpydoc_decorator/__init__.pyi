# Type information for numpydoc_decorator, which ships none.
#
# The decorator that `doc` returns attaches a docstring to the class or function it
# decorates, and returns that same object. Annotated as returning a bare `Callable`, it
# would hide the type of everything it decorates.

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

_T = TypeVar("_T")

class DocumentationError(Exception): ...

def doc(
    summary: str,
    deprecation: Mapping[str, str] | None = None,
    extended_summary: str | None = None,
    parameters: Mapping[str, str] | None = None,
    returns: str | tuple[str, ...] | Mapping[str, str] | None = None,
    yields: str | Mapping[str, str] | None = None,
    receives: str | Mapping[str, str] | None = None,
    other_parameters: Mapping[str, str] | None = None,
    raises: Mapping[str, str] | None = None,
    warns: Mapping[str, str] | None = None,
    warnings: str | None = None,
    see_also: str | Sequence[str] | Mapping[str, str | None] | None = None,
    notes: str | None = None,
    references: Mapping[str, str] | None = None,
    examples: str | None = None,
    include_extras: bool = False,
) -> Callable[[_T], _T]: ...
