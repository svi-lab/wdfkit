"""Tests for SpectraCleaner and the underlying denoise_spectra_pca."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wdfkit import SpectraCleaner, SpectraSmoother
from wdfkit.spectra_cleaner._pca import denoise_spectra_pca

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_da(shape, dims=None, rng=None):
    """Create a synthetic DataArray with given shape and optional dims."""
    if rng is None:
        rng = np.random.default_rng(0)
    data = rng.random(shape).astype(np.float64)
    if dims is None:
        spatial = [f"d{i}" for i in range(len(shape) - 1)]
        dims = (*spatial, "nm")
    coords = {d: np.arange(s) for d, s in zip(dims, shape)}
    return xr.DataArray(
        data, dims=dims, coords=coords, attrs={"treatments": {}}
    )


# ---------------------------------------------------------------------------
# denoise_spectra_pca (low-level)
# ---------------------------------------------------------------------------


def test_pca_output_shape_preserved():
    rng = np.random.default_rng(1)
    # n_samples >= n_features required for n_components="mle"
    arr = rng.random((50, 20))
    cleaned, meta = denoise_spectra_pca(arr)
    assert cleaned.shape == arr.shape


def test_pca_3d_cube_shape_preserved():
    rng = np.random.default_rng(2)
    # use explicit n_components so mle constraint is irrelevant
    arr = rng.random((4, 5, 60))
    cleaned, meta = denoise_spectra_pca(arr, n_components=5)
    assert cleaned.shape == arr.shape


def test_pca_meta_keys():
    arr = np.random.default_rng(3).random((30, 10))
    _, meta = denoise_spectra_pca(arr)
    for key in (
        "n_components_used",
        "explained_variance_ratio_total",
        "n_spectra",
        "n_spectral",
    ):
        assert key in meta, f"missing meta key: {key}"


def test_pca_return_decomposition_payload():
    arr = np.random.default_rng(4).random((12, 8))
    cleaned, meta, payload = denoise_spectra_pca(
        arr, return_decomposition=True
    )
    assert "components" in payload
    assert "coeffs" in payload
    assert "mean" in payload
    assert payload["components"].shape[1] == 8


def test_pca_single_spectrum_raises():
    arr = np.random.default_rng(5).random((1, 50))
    with pytest.raises(ValueError, match="more than one spectrum"):
        denoise_spectra_pca(arr)


def test_pca_reduces_noise():
    """A low-rank signal plus white noise should be recovered better by PCA."""
    rng = np.random.default_rng(6)
    n_spec, n_pts = 40, 100
    # Low-rank signal: 3 components
    components = rng.random((3, n_pts))
    coeffs = rng.random((n_spec, 3))
    signal = coeffs @ components
    noisy = signal + rng.normal(scale=0.05, size=signal.shape)

    # subtract_min=False + restore_min=False keeps the absolute scale intact.
    cleaned, _ = denoise_spectra_pca(noisy, n_components=3, subtract_min=False)
    mse_noisy = float(np.mean((noisy - signal) ** 2))
    mse_cleaned = float(np.mean((cleaned - signal) ** 2))
    assert mse_cleaned < mse_noisy, (
        f"PCA should reduce noise: MSE before={mse_noisy:.4f}, "
        f"after={mse_cleaned:.4f}"
    )


# ---------------------------------------------------------------------------
# SpectraCleaner (high-level)
# ---------------------------------------------------------------------------


def test_spectra_cleaner_output_shape_2d():
    # n_spectra (80) >= n_spectral (15) so "mle" is valid
    da = _make_da((80, 15))
    out = SpectraCleaner().clean(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_spectra_cleaner_output_shape_3d():
    da = _make_da((3, 4, 20))
    out = SpectraCleaner(n_components=5).clean(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_spectra_cleaner_writes_treatment():
    da = _make_da((10, 8))
    out = SpectraCleaner(n_components=3).clean(da)
    assert "spectra_cleaning" in out.attrs["treatments"]
    meta = out.attrs["treatments"]["spectra_cleaning"]
    assert meta["method"] == "pca"
    assert meta["n_components_used"] <= 3


def test_spectra_cleaner_n_components_int():
    da = _make_da((20, 10))
    out = SpectraCleaner(n_components=2).clean(da)
    meta = out.attrs["treatments"]["spectra_cleaning"]
    assert meta["n_components_used"] == 2


def test_spectra_cleaner_transform_alias():
    da = _make_da((10, 8))
    sc = SpectraCleaner(n_components=3)
    out_clean = sc.clean(da)
    out_transform = sc.transform(da)
    np.testing.assert_array_equal(out_clean.values, out_transform.values)


def test_spectra_cleaner_with_decomposition():
    da = _make_da((8, 6))
    sc = SpectraCleaner(n_components=3)
    out, payload = sc.clean_with_decomposition(da)
    assert out.shape == da.shape
    assert "components" in payload
    assert payload["components"].shape == (3, 6)


def test_spectra_cleaner_explicit_spectral_dim():
    # spectral axis first: 6 channels, 8 spectra → n_spectra >= n_spectral
    da = _make_da((6, 8), dims=("nm", "time"))
    sc = SpectraCleaner(spectral_dim="nm", n_components=3)
    out = sc.clean(da)
    assert out.dims == da.dims


def test_spectra_cleaner_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        SpectraCleaner(method="svd")


def test_spectra_cleaner_invalid_n_components_float_raises():
    with pytest.raises(ValueError, match="n_components"):
        SpectraCleaner(n_components=1.5)


def test_spectra_cleaner_invalid_n_components_int_raises():
    with pytest.raises(ValueError, match="n_components"):
        SpectraCleaner(n_components=0)


def test_spectra_cleaner_single_spectrum_raises():
    da = _make_da((1, 50))
    with pytest.raises(ValueError, match="more than one spectrum"):
        SpectraCleaner().clean(da)


def test_spectra_cleaner_wrong_type_raises():
    arr = np.random.default_rng(7).random((10, 50))
    with pytest.raises(TypeError, match="xarray.DataArray"):
        SpectraCleaner().clean(arr)


def test_spectra_cleaner_dask_warns_and_computes():
    da = _make_da((30, 10)).chunk({"nm": 5})
    sc = SpectraCleaner(n_components=3)
    with pytest.warns(UserWarning, match="Dask"):
        out = sc.clean(da)
    assert out.chunks is None  # result is NumPy-backed


# ---------------------------------------------------------------------------
# SpectraCleaner → SpectraSmoother delegation
# ---------------------------------------------------------------------------


def _make_1d_da(n: int = 80) -> xr.DataArray:
    data = np.random.default_rng(50).random(n).astype(np.float64) + 5.0
    return xr.DataArray(
        data,
        dims=("raman_shift",),
        coords={"raman_shift": np.arange(n)},
        attrs={"treatments": {}},
    )


def test_cleaner_1d_delegates_to_smoother():
    da = _make_1d_da()
    out = SpectraCleaner().clean(da)
    assert out.shape == da.shape
    assert out.dims == da.dims
    meta = out.attrs["treatments"]["spectra_cleaning"]
    assert meta["method"] == "savgol"


def test_cleaner_1d_custom_smoother():
    da = _make_1d_da()
    out = SpectraCleaner(
        smoother=SpectraSmoother(method="whittaker", lam=500.0)
    ).clean(da)
    meta = out.attrs["treatments"]["spectra_cleaning"]
    assert meta["method"] == "whittaker"


def test_cleaner_per_spectrum_2d():
    da = _make_da((20, 40))
    out = SpectraCleaner(per_spectrum=True).clean(da)
    assert out.shape == da.shape
    assert out.dims == da.dims
    meta = out.attrs["treatments"]["spectra_cleaning"]
    assert meta["method"] == "savgol"


def test_cleaner_per_spectrum_3d():
    da = _make_da((3, 4, 40))
    out = SpectraCleaner(per_spectrum=True).clean(da)
    assert out.shape == da.shape
    assert out.dims == da.dims


def test_cleaner_1d_no_decomposition():
    da = _make_1d_da()
    out, payload = SpectraCleaner().clean_with_decomposition(da)
    assert out.shape == da.shape
    assert payload == {}
