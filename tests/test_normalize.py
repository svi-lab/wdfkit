"""Small-signal tests for :func:`wdfkit.preprocessing.normalize`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wdfkit._shared.normalize import _trapz_y, normalize

# All supported method names (``invalid`` triggers warn + no-op scaling).
NORMALIZE_METHODS = (
    "l1",
    "l2",
    "max",
    "min_max",
    "area",
    "wave_number",
    "robust_scale",
)


@pytest.fixture
def tiny_spectral_da():
    """Two spectra × five channels, last dim is spectral."""
    rng = np.random.default_rng(42)
    data = np.abs(rng.standard_normal((2, 5)) * 10.0) + 1.0
    x = np.asarray([400.0, 450.0, 500.0, 700.0, 900.0], dtype=np.float64)
    return xr.DataArray(
        data,
        dims=("point", "nm"),
        coords={"point": [0, 1], "nm": x},
        attrs={"treatments": {}},
    )


@pytest.fixture
def tiny_map_da():
    """2×2 map × four spectral bins (3D)."""
    rng = np.random.default_rng(7)
    spec = np.linspace(1000.0, 1500.0, 4)
    cube = np.abs(rng.standard_normal((2, 2, 4))) * 5.0 + 2.0
    return xr.DataArray(
        cube,
        dims=("Y", "X", "raman_shift"),
        coords={
            "Y": [0.0, 1.0],
            "X": [0.0, 1.0],
            "raman_shift": spec,
        },
        attrs={"treatments": {}},
    )


@pytest.mark.parametrize("method", NORMALIZE_METHODS)
def test_normalize_each_method_dataarray(tiny_spectral_da, method):
    out = normalize(tiny_spectral_da, method=method)
    assert out.dims == tiny_spectral_da.dims
    assert np.all(np.isfinite(out.values))
    assert out.attrs["treatments"]["normalization"]["method"] == method


@pytest.mark.parametrize("method", NORMALIZE_METHODS)
def test_normalize_each_method_runs_on_small_ndarray(method):
    spec = np.array(
        [[1.0, 2.0, 4.0, 1.0], [3.0, 1.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    x = np.array([0.0, 1.0, 3.0, 10.0], dtype=np.float64)
    out = normalize(spec, method=method, x_values=x)
    assert out.shape == spec.shape
    assert np.all(np.isfinite(out))


def test_normalize_area_numpy2_trapz_integral_unity_before_min_subtract():
    """``area`` uses the trapezoid rule; flat spectrum becomes all
    zeros."""
    row = np.ones((1, 4), dtype=np.float64)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    spectra = row.copy()
    denom = _trapz_y(spectra, x, axis=-1)[:, np.newaxis]
    normalized = spectra / denom
    assert normalized.shape == (1, 4)
    np.testing.assert_allclose(_trapz_y(normalized, x, axis=-1), [1.0])
    # ``normalize`` also subtracts per-spectrum min → flat spectrum becomes 0.
    out = normalize(row, method="area", x_values=x)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_normalize_area_nonuniform_axis_changes_scaling(tiny_spectral_da):
    da_u = tiny_spectral_da
    data = da_u.values.copy()
    x_log = np.exp(np.linspace(np.log(400.0), np.log(900.0), data.shape[-1]))
    da_ir = xr.DataArray(
        data,
        dims=da_u.dims,
        coords={"point": da_u["point"], "nm": x_log},
        attrs={"treatments": {}},
    )
    out_u = normalize(da_u, method="area")
    out_ir = normalize(da_ir, method="area")
    assert not np.allclose(out_u.values, out_ir.values, rtol=0, atol=1e-6)


def test_normalize_wave_number_scale_at_anchor():
    x = np.array([500.0, 600.0, 700.0], dtype=np.float64)
    row = np.array([[1.0, 3.0, 2.0]], dtype=np.float64)
    da = xr.DataArray(
        row,
        dims=("a", "wavenumber"),
        coords={"a": [0], "wavenumber": x},
    )
    out = normalize(da, method="wave_number", wave_number=600.0)
    j = int(np.argmin(np.abs(x - 600.0)))
    # After normalize: divide by value at 600, then subtract spectrum min.
    anchor = row[:, [j]]
    raw_scaled = row / anchor
    raw_scaled = raw_scaled - raw_scaled.min(axis=-1, keepdims=True)
    np.testing.assert_allclose(out.values, raw_scaled, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "sdim,coord",
    [
        ("nm", np.linspace(200.0, 800.0, 6)),
        ("shifts", np.linspace(0.0, 3500.0, 6)),
        (
            "raman_shift",
            1.0e7 * np.linspace(1.0 / 600.0, 1.0 / 400.0, 6),
        ),
    ],
)
def test_normalize_respects_spectral_dim_name_and_coord_units(sdim, coord):
    rng = np.random.default_rng(3)
    vals = np.abs(rng.standard_normal((1, len(coord)))) + 0.5
    da = xr.DataArray(
        vals,
        dims=("spot", sdim),
        coords={"spot": [0], sdim: coord.astype(np.float64)},
        attrs={"treatments": {}},
    )
    out_max = normalize(da, method="max")
    out_area = normalize(da, method="area")
    assert out_max.dims == da.dims
    assert out_max["spot"].equals(da["spot"])
    assert out_max[sdim].equals(da[sdim])
    assert out_area[sdim].equals(da[sdim])
    assert out_area.attrs["treatments"]["normalization"]["method"] == "area"


def test_normalize_spectral_dim_when_not_last_on_map(tiny_map_da):
    da = tiny_map_da.transpose("raman_shift", "Y", "X")
    assert da.dims[0] == "raman_shift"
    out = normalize(da, method="l1", spectral_dim="raman_shift")
    assert out.dims == da.dims
    assert out.attrs["treatments"]["normalization"]["method"] == "l1"
    assert np.allclose(out.values.min(axis=0), 0.0, atol=1e-10)
    assert np.all(np.isfinite(out.values))


def test_normalize_invalid_method_raises(tiny_spectral_da):
    with pytest.raises(ValueError, match="not_a_real_method"):
        normalize(tiny_spectral_da, method="not_a_real_method")
