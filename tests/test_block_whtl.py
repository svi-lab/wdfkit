"""Tests for the WHTL white-light image block (§9)."""

from pathlib import Path

import pytest

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"

MAP_FIXTURES = [
    "SiWafer_MapImageAcquisition_7points.wdf",
    "SiWafer_MapImageAcquisition_line.wdf",
    "SiWafer_MapImageAcquisition_circleFilledRaster.wdf",
    "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf",
    "SiWafer_StreamLineImageAcquisition_DataOptimisedExposureTime.wdf",
    "SiWafer_StreamHR_ImageAcquisition_line.wdf",
    "SiWafer_StreamHR_ImageAcquisition_rectangleFilled.wdf",
]

NON_MAP_FIXTURES = [
    "SiWafer_SingleScan.wdf",
    "SiWafer_DepthSeries.wdf",
]


@pytest.mark.parametrize("fname", MAP_FIXTURES)
def test_whtl_present_in_maps(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.has_whitelight
    assert r.whtl_jpeg_bytes is not None
    assert r.whtl_jpeg_bytes[:3] == b"\xff\xd8\xff"  # JPEG SOI
    assert r.whtl_jpeg_bytes[-2:] == b"\xff\xd9"  # JPEG EOI


@pytest.mark.parametrize("fname", NON_MAP_FIXTURES)
def test_whtl_absent_in_non_maps(fname):
    r = WDFReader(TEST_DATA / fname)
    assert not r.has_whitelight
    assert r.whtl_jpeg_bytes is None
