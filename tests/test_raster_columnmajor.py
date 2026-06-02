"""Tests for the ``kind="raster_columnmajor"`` handler (StreamLine)."""

from __future__ import annotations

import pytest
from conftest import TEST_DATA

import wdfkit
from wdfkit.wdf.io import parse_wdf_to_parsed

_FILES = [
    ("SiWafer_StreamLineImageAcquisition_DataOptimisedExposureTime.wdf", 7, 8),
]


@pytest.mark.parametrize("fname,ny,nx", _FILES)
def test_classify_raster_columnmajor(fname, ny, nx):
    info = wdfkit.classify(TEST_DATA / fname)
    assert info["kind"] == "raster_columnmajor"


@pytest.mark.parametrize("fname,ny,nx", _FILES)
def test_shape_and_dims(fname, ny, nx):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims[:2] == ("y", "x"), f"Got {da.dims}"
    assert da.dims[-1] == "raman_shift"
    assert da.shape[:2] == (ny, nx)


@pytest.mark.parametrize("fname,ny,nx", _FILES)
def test_spectral_axis_last(fname, ny, nx):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims[-1] == "raman_shift"


def test_orgn_x_constant_within_rows():
    """After column-major reshape+transpose, SpatialX should be constant
    within each row."""
    fname = "SiWafer_StreamLineImageAcquisition_DataOptimisedExposureTime.wdf"
    parsed = parse_wdf_to_parsed(TEST_DATA / fname)
    nx = int(parsed.wmap.nsteps[0])
    ny = int(parsed.wmap.nsteps[1])
    orgn_x = parsed.orgn_by_type("SpatialX")
    assert orgn_x is not None
    x_nx_ny = orgn_x.values[: nx * ny].reshape(nx, ny)
    # Transposed: rows are Y-rows, columns are X-columns
    x_transposed = x_nx_ny.T  # shape (ny, nx)
    assert (
        x_transposed.std(axis=0).max() < 1e-6
    ), "SpatialX not constant within rows (col-major)"


def test_kind_attr():
    da = wdfkit.read(
        TEST_DATA
        / "SiWafer_StreamLineImageAcquisition_DataOptimisedExposureTime.wdf"
    )
    assert da.attrs["kind"] == "raster_columnmajor"
