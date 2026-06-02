"""Tests for the ``kind="line_xy"`` handler."""

from __future__ import annotations

import pytest
from conftest import TEST_DATA

import wdfkit

_FILES = [
    ("SiWafer_MapImageAcquisition_line.wdf", 12),
    ("SiWafer_StreamHR_ImageAcquisition_line.wdf", 13),
]


@pytest.mark.parametrize("fname,n", _FILES)
def test_classify_line_xy(fname, n):
    info = wdfkit.classify(TEST_DATA / fname)
    assert info["kind"] == "line_xy"


@pytest.mark.parametrize("fname,n", _FILES)
def test_line_xy_shape_and_dims(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims == ("point", "raman_shift"), f"Got {da.dims}"
    assert da.shape == (n, 1015)


@pytest.mark.parametrize("fname,n", _FILES)
def test_spectral_axis_last(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims[-1] == "raman_shift"


@pytest.mark.parametrize("fname,n", _FILES)
def test_xy_coords_present(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert "x" in da.coords
    assert "y" in da.coords


@pytest.mark.parametrize("fname,n", _FILES)
def test_kind_attr(fname, n):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.attrs["kind"] == "line_xy"
