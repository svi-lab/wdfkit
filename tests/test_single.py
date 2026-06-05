"""Tests for the ``kind="single"`` handler."""

from __future__ import annotations

import pytest
import xarray as xr
from conftest import TEST_DATA

import wdfkit
from wdfkit.wdf.io import parse_wdf_to_parsed

_FILES = [
    "SiWafer_SingleScan.wdf",
    "test.wdf",
]


@pytest.mark.parametrize("fname", _FILES)
def test_classify_single(fname):
    info = wdfkit.classify(TEST_DATA / fname)
    assert info["kind"] == "single"


@pytest.mark.parametrize("fname", _FILES)
def test_single_returns_dataarray(fname):
    da = wdfkit.read(TEST_DATA / fname)
    assert isinstance(da, xr.DataArray)


@pytest.mark.parametrize("fname", _FILES)
def test_single_is_1d(fname):
    da = wdfkit.read(TEST_DATA / fname)
    assert da.ndim == 1, f"Expected 1-D DataArray, got {da.dims}"


@pytest.mark.parametrize("fname", _FILES)
def test_spectral_axis_is_last_and_only(fname):
    da = wdfkit.read(TEST_DATA / fname)
    assert len(da.dims) == 1
    assert da.dims[-1] == da.dims[0]


def test_single_scan_si_shape_and_dim():
    da = wdfkit.read(TEST_DATA / "SiWafer_SingleScan.wdf")
    assert da.dims == ("spectral",)
    assert da.shape == (1015,)
    assert da.attrs["kind"] == "single"
    assert da.attrs["data_type"] == "single"


def test_single_scan_nspectra_1():
    parsed = parse_wdf_to_parsed(TEST_DATA / "SiWafer_SingleScan.wdf")
    assert parsed.nspectra == 1


def test_data_payload_size():
    """DATA payload == nspectra * xlist_length * 4 bytes."""
    import os
    import struct

    path = TEST_DATA / "SiWafer_SingleScan.wdf"
    parsed = parse_wdf_to_parsed(path)
    expected = parsed.nspectra * parsed.npoints * 4
    # Verify by re-reading block sizes
    BLOCK_HDR = struct.Struct("<4sIq")
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        offset = 0
        data_payload = None
        while offset < fsize - 1:
            f.seek(offset)
            raw = f.read(16)
            if len(raw) < 16:
                break
            tag, uid, size = BLOCK_HDR.unpack(raw)
            if tag == b"DATA":
                data_payload = size - 16  # subtract block header
                break
            offset += size
    assert data_payload is not None
    assert data_payload == expected
