"""Test case generation via automatic parameterisation"""

from pathlib import Path


def pytest_generate_tests(metafunc):
    """Inject parameterisation into discovered test function."""
    if "wrap_path" in metafunc.fixturenames:
        metafunc.parametrize("wrap_path", [str, Path])
