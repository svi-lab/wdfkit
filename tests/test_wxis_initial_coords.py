"""Tests for WXIS initial_coordinates across all fixture files (§10.3)."""

from pathlib import Path

import pytest

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"

ALL_FIXTURES = [
    "SiWafer_SingleScan.wdf",
    "SiWafer_DepthSeries.wdf",
    "SiWafer_MapImageAcquisition_7points.wdf",
    "SiWafer_MapImageAcquisition_line.wdf",
    "SiWafer_MapImageAcquisition_circleFilledRaster.wdf",
    "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf",
    "SiWafer_StreamLineImageAcquisition_DataOptimisedExposureTime.wdf",
    "SiWafer_StreamHR_ImageAcquisition_line.wdf",
    "SiWafer_StreamHR_ImageAcquisition_rectangleFilled.wdf",
]


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_initial_coordinates_present(fname):
    r = WDFReader(TEST_DATA / fname)
    ic = r.initial_coordinates
    assert ic is not None, f"{fname}: initial_coordinates is None"
    for key in ("x_um", "y_um", "z_um", "x_str", "y_str", "z_str"):
        assert key in ic, f"{fname}: missing key {key!r}"


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_initial_coordinates_numeric(fname):
    r = WDFReader(TEST_DATA / fname)
    ic = r.initial_coordinates
    assert ic is not None
    for key in ("x_um", "y_um", "z_um"):
        assert isinstance(ic[key], float), f"{fname}: {key} is not float"


@pytest.mark.parametrize("fname", ALL_FIXTURES)
def test_initial_coordinates_strings_nonempty_or_parseable(fname):
    r = WDFReader(TEST_DATA / fname)
    ic = r.initial_coordinates
    assert ic is not None
    # String copies may be empty strings if WXIS had no string motor entries.
    for key in ("x_str", "y_str", "z_str"):
        assert isinstance(ic[key], str), f"{fname}: {key} is not str"
