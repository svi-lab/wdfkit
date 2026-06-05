"""Tests for the XLST block (§4)."""

from pathlib import Path

import pytest

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"

FIXTURES = [
    "SiWafer_SingleScan.wdf",
    "SiWafer_DepthSeries.wdf",
    "SiWafer_MapImageAcquisition_7points.wdf",
    "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf",
]


@pytest.mark.parametrize("fname", FIXTURES)
def test_xlst_present(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.xlst is not None
    assert r.xlst.values is not None
    assert len(r.xlst.values) == 1015


@pytest.mark.parametrize("fname", FIXTURES)
def test_xlst_dtype_units(fname):
    r = WDFReader(TEST_DATA / fname)
    assert isinstance(r.xlst.data_type, str)
    assert isinstance(r.xlst.units, str)
    assert len(r.xlst.units) > 0


def test_spectral_coord_attrs():
    """spectral coord must carry 'units' and 'long_name' attrs."""
    da, _ = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    sc = da["spectral"]
    assert "units" in sc.attrs, "spectral coord missing 'units'"
    assert "long_name" in sc.attrs, "spectral coord missing 'long_name'"
    assert len(sc.attrs["units"]) > 0
    assert len(sc.attrs["long_name"]) > 0


def test_xlst_sorted():
    """After sort_spectral(), spectral axis must be strictly ascending."""
    import numpy as np

    da, _ = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    vals = da["spectral"].values
    assert float(vals[0]) < float(vals[-1]), "spectral axis first > last"
    assert np.all(
        np.diff(vals) > 0
    ), "spectral axis not monotonically ascending"
