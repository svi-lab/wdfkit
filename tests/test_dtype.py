"""Tests for the ``dtype`` parameter on ``read()`` / ``WDFReader``."""

from pathlib import Path

import numpy as np
import pytest

from wdfkit import WDFReader, read

TEST_DATA = Path(__file__).resolve().parent / "test_data"

SINGLE = TEST_DATA / "SiWafer_SingleScan.wdf"
MAP = TEST_DATA / "SiWafer_MapImageAcquisition_rectangleFilledRaster.wdf"


@pytest.mark.parametrize("chunks", [False, True])
def test_default_dtype_is_float64(chunks):
    da = read(MAP, chunks=chunks)
    assert da.dtype == np.float64


@pytest.mark.parametrize("chunks", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_explicit_dtype(chunks, dtype):
    da = read(MAP, chunks=chunks, dtype=dtype)
    assert da.dtype == np.dtype(dtype)


def test_lazy_dtype_before_compute():
    r = WDFReader(MAP, chunks=True, dtype="float32")
    assert r.raw_data.dtype == np.float32  # dask array, not yet computed
    assert r.raw_data.compute().dtype == np.float32


def test_numpy_dtype_accepted():
    da = read(SINGLE, dtype=np.float32)
    assert da.dtype == np.float32


@pytest.mark.parametrize("bad", ["int32", int, "complex128"])
def test_non_float_dtype_rejected(bad):
    with pytest.raises(TypeError, match="floating-point"):
        read(SINGLE, dtype=bad)


def test_values_match_across_dtypes():
    lo = read(MAP, dtype="float32")
    hi = read(MAP, dtype="float64")
    np.testing.assert_allclose(lo.values, hi.values, rtol=1e-6, atol=0)


def test_eager_and_lazy_values_match():
    eager = read(MAP, dtype="float32")
    lazy = read(MAP, chunks=True, dtype="float32")
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(lazy))
