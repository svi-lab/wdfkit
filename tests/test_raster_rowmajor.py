"""Tests for the ``kind="raster_rowmajor"`` handler."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TEST_DATA

import wdfkit
from wdfkit.wdf.io import parse_wdf_to_parsed

_FILES = [
    ("SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf", 9, 11),
    ("SiWafer_StreamHR_ImageAcquisition_rectangleFilled.wdf", 7, 10),
    ("test_map.wdf", 17, 25),
]


@pytest.mark.parametrize("fname,ny,nx", _FILES)
def test_classify_raster_rowmajor(fname, ny, nx):
    info = wdfkit.classify(TEST_DATA / fname)
    assert info["kind"] == "raster_rowmajor"


@pytest.mark.parametrize("fname,ny,nx", _FILES)
def test_shape_and_dims(fname, ny, nx):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims[-1] == "spectral", f"Bad spectral dim: {da.dims[-1]}"
    assert da.dims[:2] == ("row", "column"), f"Got {da.dims}"
    assert da.shape[:2] == (ny, nx)


@pytest.mark.parametrize("fname,ny,nx", _FILES)
def test_spectral_axis_last(fname, ny, nx):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.dims[-1] == "spectral"


def test_orgn_x_constant_within_rows():
    """SpatialX should be constant within each row after reshape(ny, nx)."""
    fname = "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"
    parsed = parse_wdf_to_parsed(TEST_DATA / fname)
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    orgn_x = parsed.orgn_by_type("SpatialX")
    assert orgn_x is not None
    x_2d = orgn_x.values[: ny * nx].reshape(ny, nx)
    # All rows should have the same X values
    assert x_2d.std(axis=0).max() < 1e-6, "SpatialX not constant within rows"


def test_orgn_y_constant_within_columns():
    """SpatialY should be constant within each column after reshape(ny, nx)."""
    fname = "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"
    parsed = parse_wdf_to_parsed(TEST_DATA / fname)
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    orgn_y = parsed.orgn_by_type("SpatialY")
    assert orgn_y is not None
    y_2d = orgn_y.values[: ny * nx].reshape(ny, nx)
    assert (
        y_2d.std(axis=1).max() < 1e-6
    ), "SpatialY not constant within columns"


def test_xy_coords_are_1d(
    fname="SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf",
):
    da = wdfkit.read(TEST_DATA / fname)
    assert da["column"].ndim == 1
    assert da["row"].ndim == 1
    assert da.attrs["data_type"] == "grid"
    assert da.attrs["row_axis"] == "y"
    assert da.attrs["column_axis"] == "x"


def test_kind_attr():
    da = wdfkit.read(
        TEST_DATA / "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"
    )
    assert da.attrs["kind"] == "raster_rowmajor"


def test_golden_sha256_test_map():
    """Verify test_map.wdf data bytes match the original implementation."""
    import hashlib

    da = wdfkit.read(TEST_DATA / "test_map.wdf")
    sha = hashlib.sha256(np.ascontiguousarray(da.values).tobytes()).hexdigest()
    assert (
        sha
        == "3b389ea645ecc6f712147b264d607800598bc76354cff6a28b558c08b06c25a9"
    )
