"""Tests for the TEXT block (§7)."""

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
def test_comment_is_string(fname):
    r = WDFReader(TEST_DATA / fname)
    assert r.comment is not None
    assert isinstance(r.comment, str)
    assert len(r.comment) > 0


def test_single_comment_content():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    assert "single scan" in r.comment.lower()


def test_series_comment_content():
    r = WDFReader(TEST_DATA / "SiWafer_DepthSeries.wdf")
    assert "depth series" in r.comment.lower()
