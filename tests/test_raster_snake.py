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
    assert da.dims[:2] == ("row", "column"), f"Got {da.dims}"
    assert da.dims[-1] == "spectral"
    assert da.shape == (7, 7, 1015)


def test_circ_snake_falls_back_to_points():
    """Irregular snake (nsteps=(1,1,1)) should produce point-list output."""
    da = wdfkit.read(TEST_DATA / _CIRC_FILE)
    # Falls back to (point, spectral) since no rectangular grid is possible
    assert da.dims[-1] == "spectral"
    assert da.shape[-1] == 1015


def test_spectral_axis_last():
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    assert da.dims[-1] == "spectral"


def test_kind_attr():
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    assert da.attrs["kind"] == "raster_snake"
    assert da.attrs["data_type"] == "grid"
    assert da.attrs["row_axis"] == "y"
    assert da.attrs["column_axis"] == "x"


def test_odd_rows_unserpentined():
    """After un-serpentining,
    column coords should be monotonically non-decreasing."""
    da = wdfkit.read(TEST_DATA / _RECT_FILE)
    x_vals = da["column"].values
    assert len(x_vals) == 7
    assert np.all(
        np.diff(x_vals) >= 0
    ), "column coords are not non-decreasing after un-serpentine"
