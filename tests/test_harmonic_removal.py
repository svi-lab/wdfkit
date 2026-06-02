"""Tests for laser-harmonic notch (Nd:YAG 355 nm) before cosmic-ray
removal."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wdfkit import CosmicRayRemover
from wdfkit.cosmic_ray._harmonic import (
    harmonic_correct_dataarray,
    read_laser_wavelength_nm,
    should_apply_nd_yag_harmonic_cleanup,
)


def test_laser_trigger_band():
    assert should_apply_nd_yag_harmonic_cleanup(355.0)
    assert should_apply_nd_yag_harmonic_cleanup(354.0)
    assert should_apply_nd_yag_harmonic_cleanup(356.0)
    assert not should_apply_nd_yag_harmonic_cleanup(354.0 - 0.01)
    assert not should_apply_nd_yag_harmonic_cleanup(356.0 + 0.01)
    assert not should_apply_nd_yag_harmonic_cleanup(None)
    assert read_laser_wavelength_nm({}) is None


@pytest.mark.parametrize("laser", [532.0, None])
def test_harmonic_check_skipped_wrong_laser_or_missing(laser):
    n = 200
    da = xr.DataArray(
        np.ones((1, n), dtype=float),
        dims=("i", "nm"),
        coords={"i": [0], "nm": np.linspace(400, 800, n)},
        attrs={"treatments": {}},
    )
    if laser is not None:
        da.attrs["laser_wavelength_nm"] = laser
    out = CosmicRayRemover().harmonic_check(da)
    assert out is da
    np.testing.assert_array_equal(out.values, da.values)


def test_harmonic_check_notches_532_nm_axis(capsys):
    n = 1500
    nm = np.linspace(520, 545, n, dtype=float)
    y = np.ones(n, dtype=float) * 100.0
    center = int(np.argmin(np.abs(nm - 532.0)))
    y[center] += 5000.0
    da = xr.DataArray(
        y[np.newaxis, :],
        dims=("i", "nm"),
        coords={"i": [0], "nm": nm},
        attrs={
            "laser_wavelength_nm": 355.0,
            "Filename": "synthetic.dat",
            "treatments": {},
        },
    )
    out = CosmicRayRemover().harmonic_check(da)
    captured = capsys.readouterr()
    assert "355nm laser detected" in captured.out
    assert any(s in captured.out for s in ("530", "531", "532"))
    assert "synthetic.dat" in captured.out
    assert out.values[0, center] < y[center] / 10
    assert "Laser harmonic removal" in out.attrs["treatments"]


def test_harmonic_check_wavenumber_axis(capsys):
    n = 1500
    span_lo = 1e7 / 545.0
    span_hi = 1e7 / 520.0
    wn = np.linspace(span_lo, span_hi, n, dtype=float)
    y = np.ones(n, dtype=float) * 50.0
    target_wn = 1e7 / 532.0
    center = int(np.argmin(np.abs(wn - target_wn)))
    y[center] += 5000.0
    da = xr.DataArray(
        y[np.newaxis, :],
        dims=("t", "wavenumber"),
        coords={"t": [0], "wavenumber": wn},
        attrs={
            "laser_wavelength_nm": 355.0,
            "Filename": "unknown file",
            "treatments": {},
        },
    )
    out = harmonic_correct_dataarray(da)
    captured = capsys.readouterr()
    assert "355nm laser detected" in captured.out
    assert out.values[0, center] < y[center] / 10


def test_remove_runs_harmonic_then_cosmic_ray():
    n = 200
    spec = np.linspace(0, 1, n, dtype=np.float64) + 100.0
    spec[50] = 5000.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("Time", "nm"),
        coords={"Time": [0.0], "nm": np.linspace(400, 600, n)},
        attrs={"treatments": {}, "laser_wavelength_nm": 400.0},
    )
    out = CosmicRayRemover(spike_threshold=3.0, spike_width=5).remove(da)
    assert out.values[0, 50] < da.values[0, 50] / 10


def test_remove_cosmic_rays_skips_harmonic_when_no_laser_metadata():
    n = 200
    spec = np.linspace(0, 1, n, dtype=np.float64) + 100.0
    spec[50] = 5000.0
    da = xr.DataArray(
        spec[np.newaxis, :],
        dims=("Time", "nm"),
        coords={"Time": [0.0], "nm": np.arange(n, dtype=float)},
        attrs={"treatments": {}},
    )
    out = CosmicRayRemover(
        spike_threshold=3.0, spike_width=5
    ).remove_cosmic_rays(da)
    assert out.values[0, 50] < da.values[0, 50] / 10
