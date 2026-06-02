"""Tests for the DATA block (§3)."""

from pathlib import Path

import numpy as np
import pytest

from wdfkit import WDFReader

TEST_DATA = Path(__file__).resolve().parent / "test_data"

FIXTURES = [
    ("SiWafer_SingleScan.wdf", 1, 1015),
    ("SiWafer_DepthSeries.wdf", 10, 1015),
    ("SiWafer_MapImageAcquisition_7points.wdf", 7, 1015),
    ("SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf", 99, 1015),
]


@pytest.mark.parametrize("fname,nspectra,npoints", FIXTURES)
def test_data_shape(fname, nspectra, npoints):
    r = WDFReader(TEST_DATA / fname)
    assert r.raw_data is not None
    assert r.raw_data.shape == (nspectra, npoints)
    assert np.issubdtype(r.raw_data.dtype, np.floating)


def test_data_nonzero():
    r = WDFReader(TEST_DATA / "SiWafer_SingleScan.wdf")
    assert r.raw_data is not None
    assert np.any(r.raw_data != 0)
