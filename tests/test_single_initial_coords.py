"""Tests for attrs["InitialCoordinates"] on 1-D (Single) DataArrays."""

from __future__ import annotations

import pytest
from conftest import TEST_DATA

import wdfkit

_EXPECTED_KEYS = {"x", "y", "z"}

# One representative file from each dimensionality class
_FILES_BY_KIND = [
    ("single", "SiWafer_SingleScan.wdf"),
    (
        "raster_rowmajor",
        "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf",
    ),
    ("raster_snake", "SiWafer_MapImageAcquisition_rectangleFilledSnake.wdf"),
]


@pytest.mark.parametrize("kind,fname", _FILES_BY_KIND)
def test_initial_coordinates_present(kind, fname):
    da = wdfkit.read(TEST_DATA / fname)
    assert (
        "InitialCoordinates" in da.attrs
    ), f"{kind}: attrs['InitialCoordinates'] missing"


@pytest.mark.parametrize("kind,fname", _FILES_BY_KIND)
def test_initial_coordinates_keys(kind, fname):
    da = wdfkit.read(TEST_DATA / fname)
    assert set(da.attrs["InitialCoordinates"].keys()) == _EXPECTED_KEYS, (
        f"{kind}: expected keys {_EXPECTED_KEYS}, "
        f"got {set(da.attrs['InitialCoordinates'].keys())}"
    )


def test_single_initial_coordinates_values():
    da = wdfkit.read(TEST_DATA / "SiWafer_SingleScan.wdf")
    ic = da.attrs["InitialCoordinates"]
    assert ic["x"] == 0, f"x={ic['x']!r}"
    assert ic["y"] == 0, f"y={ic['y']!r}"
    assert ic["z"] == 0, f"z={ic['z']!r}"
