"""Unit tests for :func:`wdfkit.cosmic_ray.remove_cosmic_rays_1d` and
:class:`wdfkit.CosmicRayRemover` with 1-D input."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wdfkit import CosmicRayRemover
from wdfkit.cosmic_ray import remove_cosmic_rays_1d


def test_validate_y_must_be_1d():
    with pytest.raises(ValueError, match="1D"):
        remove_cosmic_rays_1d(np.zeros((2, 3)), kernel_size=5)


def test_validate_kernel_must_be_odd():
    with pytest.raises(ValueError, match="spike_width"):
        remove_cosmic_rays_1d(np.zeros(10), kernel_size=4)


def test_validate_map_method_invalid():
    with pytest.raises(ValueError, match="map_method"):
        CosmicRayRemover(map_method="pca_bad")


def test_validate_threshold_positive():
    with pytest.raises(ValueError, match="threshold"):
        remove_cosmic_rays_1d(np.arange(20, dtype=float), threshold=0.0)


def test_median_constant_spectrum_no_change():
    y = np.full(50, 100.0)
    out, mask = remove_cosmic_rays_1d(y, kernel_size=5)
    np.testing.assert_array_equal(out, y)
    assert not mask.any()


def test_positive_spike_reduced():
    y = np.linspace(0, 1, 100, dtype=np.float64) + 10.0
    y[50] = 500.0
    out, mask = remove_cosmic_rays_1d(y, kernel_size=5, threshold=3.0)
    assert out[50] < y[50] / 5
    assert mask[50]


def test_remove_cosmic_rays_1d_casts_to_float64():
    y = np.arange(25, dtype=np.int32)
    out, _ = remove_cosmic_rays_1d(y, kernel_size=5)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# CosmicRayRemover with 1-D DataArray (single spectrum from WDFReader)
# ---------------------------------------------------------------------------


def _make_1d_da(n: int = 80, *, rng=None) -> xr.DataArray:
    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.random(n).astype(np.float64) + 10.0
    return xr.DataArray(
        data, dims=("raman_shift",), coords={"raman_shift": np.arange(n)}
    )


def test_remover_1d_shape_and_dims_preserved():
    da = _make_1d_da()
    out = CosmicRayRemover().remove_cosmic_rays(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_remover_1d_with_diagnostics():
    da = _make_1d_da()
    out, diag = CosmicRayRemover().remove_cosmic_rays_with_diagnostics(da)
    assert out.shape == da.shape
    assert "cosmic_mask" in diag
    assert diag["cosmic_mask"].shape == (da.shape[0],)


def test_remover_1d_spike_removed():
    rng = np.random.default_rng(7)
    da = _make_1d_da(rng=rng)
    da_spike = da.copy(data=da.values.copy())
    da_spike.values[40] = 5000.0
    out = CosmicRayRemover(spike_threshold=3.0).remove_cosmic_rays(da_spike)
    assert out.values[40] < da_spike.values[40] / 5
