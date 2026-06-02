"""Tests for the ``kind="series"`` handler."""

from __future__ import annotations

import xarray as xr
from conftest import TEST_DATA

import wdfkit
from wdfkit.wdf.io import parse_wdf_to_parsed

_FILE = "SiWafer_DepthSeries.wdf"


def test_classify_series():
    info = wdfkit.classify(TEST_DATA / _FILE)
    assert info["kind"] == "series"


def test_series_returns_dataarray():
    da = wdfkit.read(TEST_DATA / _FILE)
    assert isinstance(da, xr.DataArray)


def test_series_shape_and_dims():
    da = wdfkit.read(TEST_DATA / _FILE)
    assert da.dims == ("SpatialZ", "raman_shift"), f"Got {da.dims}"
    assert da.shape == (10, 1015)


def test_spectral_axis_last():
    da = wdfkit.read(TEST_DATA / _FILE)
    assert da.dims[-1] == "raman_shift"


def test_primary_axis_length_matches_nspectra():
    parsed = parse_wdf_to_parsed(TEST_DATA / _FILE)
    primary = parsed.primary_orgn()
    assert primary is not None, "No primary ORGN entry found"
    assert len(primary.values) == parsed.nspectra


def test_series_z_coord_present():
    da = wdfkit.read(TEST_DATA / _FILE)
    assert "SpatialZ" in da.coords
    assert len(da["SpatialZ"]) == 10


def test_kind_attr():
    da = wdfkit.read(TEST_DATA / _FILE)
    assert da.attrs["kind"] == "series"
