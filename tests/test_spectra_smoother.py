"""Tests for SpectraSmoother and the underlying smooth_1d helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wdfkit import SpectraSmoother
from wdfkit.spectra_smoother._smooth_1d import (
    auto_lam,
    savgol_smooth_1d,
    whittaker_smooth_1d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_da(shape, dims=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    data = rng.random(shape).astype(np.float64) + 5.0
    if dims is None:
        spatial = [f"d{i}" for i in range(len(shape) - 1)]
        dims = (*spatial, "raman_shift")
    coords = {d: np.arange(s) for d, s in zip(dims, shape)}
    return xr.DataArray(
        data, dims=dims, coords=coords, attrs={"treatments": {}}
    )


def _make_1d_da(n: int = 80, *, rng=None) -> xr.DataArray:
    if rng is None:
        rng = np.random.default_rng(1)
    data = rng.random(n).astype(np.float64) + 5.0
    return xr.DataArray(
        data,
        dims=("raman_shift",),
        coords={"raman_shift": np.arange(n)},
        attrs={"treatments": {}},
    )


# ---------------------------------------------------------------------------
# savgol_smooth_1d (low-level)
# ---------------------------------------------------------------------------


def test_savgol_1d_shape_preserved():
    y = np.random.default_rng(10).random(100)
    out = savgol_smooth_1d(y, window_length=11, polyorder=3)
    assert out.shape == y.shape
    assert out.dtype == np.float64


def test_savgol_reduces_noise():
    rng = np.random.default_rng(11)
    x = np.linspace(0, 4 * np.pi, 200)
    signal = np.sin(x)
    noisy = signal + rng.normal(scale=0.1, size=x.shape)
    smoothed = savgol_smooth_1d(noisy, window_length=15, polyorder=3)
    assert np.mean((smoothed - signal) ** 2) < np.mean((noisy - signal) ** 2)


def test_savgol_even_window_raises():
    with pytest.raises(ValueError, match="window_length"):
        savgol_smooth_1d(np.ones(50), window_length=10, polyorder=3)


def test_savgol_polyorder_too_large_raises():
    with pytest.raises(ValueError, match="polyorder"):
        savgol_smooth_1d(np.ones(50), window_length=5, polyorder=5)


# ---------------------------------------------------------------------------
# whittaker_smooth_1d (low-level)
# ---------------------------------------------------------------------------


def test_whittaker_1d_shape_preserved():
    y = np.random.default_rng(20).random(100)
    out = whittaker_smooth_1d(y, lam=1000.0)
    assert out.shape == y.shape


def test_whittaker_reduces_noise():
    rng = np.random.default_rng(21)
    x = np.linspace(0, 4 * np.pi, 200)
    signal = np.sin(x)
    noisy = signal + rng.normal(scale=0.1, size=x.shape)
    smoothed = whittaker_smooth_1d(noisy, lam=1000.0)
    assert np.mean((smoothed - signal) ** 2) < np.mean((noisy - signal) ** 2)


def test_whittaker_negative_lam_raises():
    with pytest.raises(ValueError, match="lam"):
        whittaker_smooth_1d(np.ones(50), lam=-1.0)


# ---------------------------------------------------------------------------
# auto_lam
# ---------------------------------------------------------------------------


def test_auto_lam_returns_positive():
    rng = np.random.default_rng(30)
    y = np.sin(np.linspace(0, 2 * np.pi, 100)) + rng.normal(
        scale=0.05, size=100
    )
    lam = auto_lam(y, d=2, lam0=100.0, max_calls=5)
    assert lam > 0


def test_auto_lam_smooths_reasonably():
    rng = np.random.default_rng(31)
    x = np.linspace(0, 4 * np.pi, 150)
    signal = np.sin(x)
    noisy = signal + rng.normal(scale=0.08, size=x.shape)
    lam = auto_lam(noisy, d=2, max_calls=5)
    smoothed = whittaker_smooth_1d(noisy, lam=lam)
    assert np.mean((smoothed - signal) ** 2) < np.mean((noisy - signal) ** 2)


# ---------------------------------------------------------------------------
# SpectraSmoother (high-level)
# ---------------------------------------------------------------------------


def test_smoother_savgol_1d_shape_preserved():
    da = _make_1d_da()
    out = SpectraSmoother().smooth(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_smoother_savgol_2d_shape_preserved():
    da = _make_da((30, 80))
    out = SpectraSmoother().smooth(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_smoother_savgol_3d_shape_preserved():
    da = _make_da((4, 5, 60))
    out = SpectraSmoother().smooth(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_smoother_whittaker_1d_shape_preserved():
    da = _make_1d_da()
    out = SpectraSmoother(method="whittaker", lam=500.0).smooth(da)
    assert out.shape == da.shape


def test_smoother_whittaker_auto_lam_runs():
    da = _make_1d_da(n=100)
    sc = SpectraSmoother(method="whittaker")
    out = sc.smooth(da)
    assert out.shape == da.shape
    meta = out.attrs["treatments"]["spectra_smoothing"]
    assert "lam_used" in meta
    assert meta["lam_auto"] is True


def test_smoother_writes_treatment():
    da = _make_1d_da()
    out = SpectraSmoother(method="savgol").smooth(da)
    assert "spectra_smoothing" in out.attrs["treatments"]
    meta = out.attrs["treatments"]["spectra_smoothing"]
    assert meta["method"] == "savgol"
    assert meta["window_length"] == 11
    assert meta["polyorder"] == 3


def test_smoother_whittaker_writes_treatment():
    da = _make_1d_da()
    out = SpectraSmoother(method="whittaker", lam=200.0).smooth(da)
    meta = out.attrs["treatments"]["spectra_smoothing"]
    assert meta["method"] == "whittaker"
    assert meta["lam"] == 200.0
    assert meta["lam_auto"] is False


def test_smoother_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        SpectraSmoother(method="wavelet")


def test_smoother_bad_window_raises():
    with pytest.raises(ValueError, match="window_length"):
        SpectraSmoother(window_length=4)


def test_smoother_wrong_type_raises():
    with pytest.raises(TypeError, match="xarray.DataArray"):
        SpectraSmoother().smooth(np.ones(50))


def test_smoother_transform_alias():
    da = _make_1d_da()
    sc = SpectraSmoother()
    np.testing.assert_array_equal(
        sc.smooth(da).values, sc.transform(da).values
    )
