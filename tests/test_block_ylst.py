"""Tests for the YLST block (§5)."""

from pathlib import Path

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"


def test_ylst_none_for_point_detector():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    assert r.ylst is None


def test_ylst_none_for_series():
    r = WDFReader(TEST_DATA / "SiWafer_DepthSeries.wdf")
    assert r.ylst is None


def test_ylst_none_for_map():
    r = WDFReader(
        TEST_DATA / "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"
    )
    assert r.ylst is None
