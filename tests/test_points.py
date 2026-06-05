"""Tests for the ``kind="points"`` handler."""

from __future__ import annotations

import pytest
from conftest import TEST_DATA

import wdfkit

_FILES = [
    ("SiWafer_MapImageAcquisition_7points.wdf", 7),
    ("SiWafer_MapImageAcquisition_circleFilledRaster.wdf", 21),
]


@pytest.mark.parametrize("fname,n", _FILES)
def test_classify_points(fname, n):
    info = wdfkit.classify(TEST_DATA / fname)
    assert info["kind"] == "points"


@pytest.mark.parametrize("fname,n", _FILES)
def test_points_shape_and_dims(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims == ("point", "spectral"), f"Got {da.dims}"
    assert da.shape == (n, 1015)


@pytest.mark.parametrize("fname,n", _FILES)
def test_spectral_axis_last(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims[-1] == "spectral"


@pytest.mark.parametrize("fname,n", _FILES)
def test_xy_coords_present(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert "x" in da.coords
    assert "y" in da.coords
    assert da["x"].shape == (n,)
    assert da["y"].shape == (n,)


@pytest.mark.parametrize("fname,n", _FILES)
def test_kind_attr(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.attrs["kind"] == "points"
    assert da.attrs["data_type"] == "sequence"
