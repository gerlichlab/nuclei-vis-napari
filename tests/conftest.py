"""Test case generation via automatic parameterisation"""

from pathlib import Path

from hypothesis import HealthCheck, settings

# The first test to generate data in a process pays hypothesis' one-time
# start-up, which includes scanning the modules in play for literal constants.
# That scan happens inside the window the `too_slow` check times, so the check
# reports the warm-up as though it were the strategy, and it lands on whichever
# `@given` test runs first -- a matter of filename order, not of the test. CI
# pays it every run: `.hypothesis/` is gitignored and sits in the workspace
# root, which no cache step covers.
settings.register_profile(
    "nuclei-vis-napari", suppress_health_check=[HealthCheck.too_slow]
)
# The other health checks stay armed, filter_too_much among them, which is what a
# strategy that rejects most of its draws trips. A profile named on the command
# line still wins: pytest imports this conftest before the hypothesis plugin
# applies --hypothesis-profile, so `pytest --hypothesis-profile=default` runs with
# every health check armed.
settings.load_profile("nuclei-vis-napari")


def pytest_generate_tests(metafunc):
    """Inject parameterisation into discovered test function."""
    if "wrap_path" in metafunc.fixturenames:
        metafunc.parametrize("wrap_path", [str, Path])
