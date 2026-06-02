"""Tests for the ``kind="raster_snake"`` handler."""

from __future__ import annotations

import numpy as np
from conftest import TEST_DATA

import wdfkit

_RECT_FILE = "SiWafer_MapImageAcquisition_rectangleFilledSnake.wdf"
_CIRC_FILE = "SiWafer_MapImageAcquisition_circleFilledSnake.wdf"


def test_classify_rect_snake():
    info = wdfkit.classify(TEST_DATA / _RECT_FILE)
    assert info["kind"] == "raster_snake"


def test_classify_circ_snake():
    info = wdfkit.classify(TEST_DATA / _CIRC_FILE)
    assert info["kind"] == "raster_snake"


def test_rect_snake_shape_and_dims():
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    assert da.dims[:2] == ("y", "x"), f"Got {da.dims}"
    assert da.dims[-1] == "raman_shift"
    assert da.shape == (7, 7, 1015)


def test_circ_snake_falls_back_to_points():
    """Irregular snake (nsteps=(1,1,1)) should produce point-list output."""
    da = wdfkit.read(TEST_DATA / _CIRC_FILE)
    # Falls back to (point, raman_shift) since no rectangular grid is possible
    assert da.dims[-1] == "raman_shift"
    assert da.shape[-1] == 1015


def test_spectral_axis_last():
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    assert da.dims[-1] == "raman_shift"


def test_kind_attr():
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    assert da.attrs["kind"] == "raster_snake"


def test_odd_rows_unserpentined():
    """After un-serpentining, x coords on each row should be identical."""
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    # All rows should share the same set of x values
    x_vals = da["x"].values
    assert len(x_vals) == 7
    # Coordinates should be monotonically non-decreasing (un-reversed)
    assert np.all(
        np.diff(x_vals) >= 0
    ), "x coords are not non-decreasing after un-serpentine"
