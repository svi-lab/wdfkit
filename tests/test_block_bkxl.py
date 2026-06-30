"""Tests for the BKXL block (background X list)."""

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
def test_bkxl_present(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.bkxl is not None
    assert r.bkxl.values is not None
    assert len(r.bkxl.values) == 1019


@pytest.mark.parametrize("fname", FIXTURES)
def test_bkxl_dtype_units(fname):
    r = WDFReader(TEST_DATA / fname)
    assert isinstance(r.bkxl.data_type, str)
    assert isinstance(r.bkxl.units, str)
    assert len(r.bkxl.units) > 0
