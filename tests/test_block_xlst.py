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


def test_xlst_sorted():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    vals = r.xlst.values
    assert float(vals[0]) != float(vals[-1])
