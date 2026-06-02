"""Unit tests for __version__.py."""

import wdfkit  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(wdfkit, "__version__")
    assert wdfkit.__version__ != "0.0.0"
