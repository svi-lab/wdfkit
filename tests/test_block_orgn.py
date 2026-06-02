"""Tests for the ORGN block (§6)."""

from pathlib import Path

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"


def test_orgn_single_has_time():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    entry = r.orgn_by_type("Time")
    assert entry is not None
    assert len(entry.values) == r.nspectra


def test_orgn_series_has_spatial():
    r = WDFReader(TEST_DATA / "SiWafer_DepthSeries.wdf")
    spatial = r.orgn_by_type("SpatialZ")
    assert spatial is not None
    assert len(spatial.values) == r.nspectra


def test_orgn_map_has_xy():
    r = WDFReader(
        TEST_DATA / "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"
    )
    x_entry = r.orgn_by_type("SpatialX")
    y_entry = r.orgn_by_type("SpatialY")
    assert x_entry is not None
    assert y_entry is not None


def test_orgn_primary_flag():
    r = WDFReader(TEST_DATA / "SiWafer_DepthSeries.wdf")
    primaries = [e for e in r.orgn if e.is_primary]
    assert len(primaries) >= 1
