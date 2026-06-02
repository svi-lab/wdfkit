"""Tests for the WMAP block (§8)."""

from pathlib import Path

from wdfkit import WDFReader
from wdfkit.wdf.types import MapFlag

TEST_DATA = Path(__file__).resolve().parent / "test_data"


def test_wmap_absent_for_single():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    assert r.wmap is None


def test_wmap_absent_for_series():
    r = WDFReader(TEST_DATA / "SiWafer_DepthSeries.wdf")
    assert r.wmap is None


def test_wmap_present_for_map():
    r = WDFReader(
        TEST_DATA / "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"
    )
    assert r.wmap is not None
    assert r.wmap.flag == MapFlag.StandardRaster
    assert len(r.wmap.nsteps) == 3
    assert r.wmap.nsteps[2] == 1  # 2D map, nz == 1


def test_wmap_random_points():
    r = WDFReader(TEST_DATA / "SiWafer_MapImageAcquisition_7points.wdf")
    assert r.wmap is not None
    assert r.wmap.flag & MapFlag.RandomPoints


def test_wmap_line():
    r = WDFReader(TEST_DATA / "SiWafer_MapImageAcquisition_line.wdf")
    assert r.wmap is not None
    assert r.wmap.flag == MapFlag.XYLine


def test_wmap_column_major():
    r = WDFReader(
        TEST_DATA
        / "SiWafer_StreamLineImageAcquisition_DataOptimisedExposureTime.wdf"
    )
    assert r.wmap is not None
    assert r.wmap.flag & MapFlag.ColumnMajor
