"""Tests for the PSET blocks: WXDA, WXDM, WXIS, WXCS, ZLDC (§10)."""

from pathlib import Path

import pytest

from wdfkit import WDFReader
from wdfkit.wdf.pset import PSet

TEST_DATA = Path(__file__).resolve().parent / "test_data"

ALL_FIXTURES = [
    "SiWafer_SingleScan.wdf",
    "SiWafer_DepthSeries.wdf",
    "SiWafer_MapImageAcquisition_7points.wdf",
    "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf",
]


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_acquisition_is_pset(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.acquisition is not None
    assert isinstance(r.acquisition, PSet)


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_instrument_status_is_pset(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.instrument_status is not None
    assert isinstance(r.instrument_status, PSet)


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_calibration_is_pset(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.calibration is not None
    assert isinstance(r.calibration, PSet)


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_zeldac_is_pset(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.zeldac is not None
    assert isinstance(r.zeldac, PSet)


def test_pset_get_by_label():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    # WXIS should have ND Transmission % accessible via get_by_label
    pset = r.instrument_status
    assert pset is not None
    # Just verify the method doesn't raise; value may be None in test files
    _ = pset.get_by_label("ND Transmission %")


def test_pset_walk():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    pset = r.instrument_status
    assert pset is not None
    entries = list(pset.walk())
    assert len(entries) > 0
    for path, label, tag, val in entries:
        assert isinstance(path, str)
        assert isinstance(label, str)
        assert isinstance(tag, str)
