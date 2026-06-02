"""Tests for spectral preprocessing (normalize, cosmic-ray removal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wdfkit import CosmicRayRemover, WDFReader, normalize

TEST_DATA = Path(__file__).resolve().parent / "test_data"


@pytest.fixture(scope="module")
def single_da():
    da, _ = WDFReader(TEST_DATA / "test.wdf")
    return da


@pytest.fixture(scope="module")
def map_da():
    da, _ = WDFReader(TEST_DATA / "test_map.wdf")
    return da


def test_normalize_default_spectral_dim_matches_reader(single_da):
    out = normalize(single_da, method="max")
    assert out.dims == single_da.dims
    assert "normalization" in out.attrs["treatments"]
    assert out.attrs["treatments"]["normalization"]["method"] == "max"
    # Max-abs row norm, then per-row min subtraction (legacy): rows touch zero.
    row_span = out.values.max(axis=-1) - out.values.min(axis=-1)
    assert np.all((out.values >= -1e-9) & (row_span > 0.95))


def test_normalize_explicit_spectral_dim_when_not_last(map_da):
    # Build a 2D array with spectral dim first (not last) using the map.
    # map_da has dims (y, x, wavelength_nm) — stack y/x and put spectral first.
    da_2d = map_da.isel(y=0)  # (x, wavelength_nm)
    da_spec_first = da_2d.transpose("wavelength_nm", "x")
    assert da_spec_first.dims[0] == "wavelength_nm"
    out = normalize(
        da_spec_first, method="min_max", spectral_dim="wavelength_nm"
    )
    assert out.dims == da_spec_first.dims
    # min_max to [0, 1] per spectrum; final min-subtract leaves [0, 1].
    assert float(np.nanmax(out.values)) <= 1.0 + 1e-6
    assert float(np.nanmin(out.values)) >= -1e-6


def test_normalize_invalid_spectral_dim_raises(single_da):
    with pytest.raises(ValueError, match="spectral_dim"):
        normalize(single_da, spectral_dim="not_a_dim")


def test_cosmic_ray_single_no_spike_unchanged():
    n = 300
    spec = np.full(n, 100.0, dtype=np.float64)
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("idx", "wavenumber"),
        coords={"idx": [0], "wavenumber": np.arange(n)},
        attrs={"treatments": {}},
    )
    cr = CosmicRayRemover(spike_threshold=5.0, spike_width=5)
    out = cr.transform(da)
    np.testing.assert_array_equal(out.values[0], spec)
    meta = out.attrs["treatments"]["Cosmic Ray Correction"]
    assert "CRs found (spectral indices)" not in meta


def test_cosmic_ray_single_removes_spike():
    n = 200
    spec = np.linspace(0, 1, n, dtype=np.float64) + 100.0
    spec[50] = 5000.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("Time", "nm"),
        coords={"Time": [0.0], "nm": np.arange(n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(spike_threshold=3.0, spike_width=5).transform(da)
    assert out.values[0, 50] < da.values[0, 50] / 10
    assert (
        "CRs found (spectral indices)"
        in out.attrs["treatments"]["Cosmic Ray Correction"]
    )


def test_cosmic_ray_single_wide_spike_width_removes_3ch_spike():
    """spike_width=7 detects a 3-channel spike that spike_width=5 misses."""
    n = 200
    spec = np.linspace(0, 1, n, dtype=np.float64) + 100.0
    # 3-channel spike: with kernel=5 the spike channels fill >half the window
    # at the peak so median stays at spike level → not detected.
    # With kernel=7 (4 non-spike channels dominate) the spike is visible.
    spike_centre = 100
    spec[spike_centre - 1 : spike_centre + 2] = 5000.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("Time", "nm"),
        coords={"Time": [0.0], "nm": np.arange(n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(spike_threshold=3.0, spike_width=7).transform(da)
    assert out.values[0, spike_centre] < 5000.0 / 5


def test_cosmic_ray_single_high_threshold_no_change():
    """Very high threshold: no channels flagged, signal unchanged."""
    n = 300
    # Constant spectrum: medfilt(constant) == constant at every channel
    # including zero-padded boundary, so residual == 0 and no spikes found.
    spec = np.full(n, 50.0, dtype=np.float64)
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("i", "cm"),
        coords={"i": [0], "cm": np.arange(n, dtype=float)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(spike_width=5, spike_threshold=50.0).transform(da)
    np.testing.assert_array_equal(out.values[0], spec)


def test_cosmic_ray_invalid_map_method_raises():
    with pytest.raises(ValueError, match="map_method"):
        CosmicRayRemover(map_method="bad_method")


def test_cosmic_ray_map_removes_spike():
    ny, nx, n = 5, 5, 64
    cube = np.random.default_rng(0).random((ny, nx, n)).astype(np.float64) * 10
    cube[2, 2, 20] = 500.0
    da = xr.DataArray(
        cube,
        dims=("Y", "X", "nm"),
        coords={
            "Y": np.arange(ny),
            "X": np.arange(nx),
            "nm": np.arange(n),
        },
        attrs={"treatments": {}},
    )
    cr = CosmicRayRemover(
        map_sensitivity=0.5, map_spike_width=8, map_disk_radius=2
    )
    out = cr.transform(da)
    assert out.values[2, 2, 20] < cube[2, 2, 20] / 5


def test_cosmic_ray_degenerate_map_uses_single_path():
    n = 100
    spec = np.ones(n, dtype=np.float64) * 50
    spec[30] = 800.0
    da = xr.DataArray(
        spec.reshape(1, 1, n),
        dims=("Y", "X", "nm"),
        coords={"Y": [0], "X": [0], "nm": np.arange(n)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(spike_threshold=2.5, spike_width=5).transform(da)
    assert out.values[0, 0, 30] < 100.0


def test_cosmic_ray_rejects_unsupported_ndim(map_da):
    # 4-D has no supported code path and must raise.
    fake_4d = map_da.expand_dims("Batch")
    with pytest.raises(ValueError, match="CosmicRayRemover"):
        CosmicRayRemover().transform(fake_4d)
