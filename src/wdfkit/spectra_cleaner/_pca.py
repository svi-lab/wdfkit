# -*- coding: utf-8 -*-
"""PCA-based spectral denoising for stacks of spectra and 3D map cubes.

PCA decomposes a *population* of spectra into orthogonal components and
reconstructs each spectrum from the leading ones. Components dominated by
uncorrelated per-channel noise are dropped, so the reconstruction is a
denoised version of the input. This requires more than one spectrum — see
:class:`wdfkit.spectra_cleaner.SpectraCleaner` for the user-facing API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import decomposition

NComponents = int | float | str | None


def _flatten_to_row_stack(
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape ``(..., n_spectral)`` to ``(n_spectra, n_spectral)``.

    Returns the row-stack and the original spatial shape (everything
    before the spectral axis), so the cleaned output can be reshaped
    back.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2:
        raise ValueError(
            "PCA denoising needs >= 2D input (last axis = spectral); "
            f"got ndim={arr.ndim}"
        )
    spatial_shape = arr.shape[:-1]
    n_spectral = arr.shape[-1]
    return arr.reshape(-1, n_spectral), spatial_shape


def _per_spectrum_min(row_stack: np.ndarray) -> np.ndarray:
    """Per-row min along the spectral axis, shape ``(n_spectra, 1)``."""
    return np.min(row_stack, axis=-1, keepdims=True)


def denoise_spectra_pca(
    values: np.ndarray,
    *,
    n_components: NComponents = "mle",
    subtract_min: bool = True,
    restore_min: bool = False,
    pca_kwargs: dict[str, Any] | None = None,
    return_decomposition: bool = False,
) -> (
    tuple[np.ndarray, dict[str, Any]]
    | tuple[np.ndarray, dict[str, Any], dict[str, Any]]
):
    """Denoise a stack / cube of spectra by PCA reconstruction.

    The input is reshaped to ``(n_spectra, n_spectral)`` for the fit, then
    reshaped back to the original spatial layout on return. PCA itself
    mean-centers internally; the optional per-spectrum min subtraction below
    only changes the *baseline offset* fed to the decomposition.

    Parameters
    ----------
    values
        Array of shape ``(..., n_spectral)``. Typical inputs:
        ``(ny, nx, n_spectral)`` map cube, or ``(n_spectra, n_spectral)``
        stack. Needs more than one spectrum (PCA on a single spectrum is
        degenerate).
    n_components
        Forwarded to :class:`sklearn.decomposition.PCA`. ``\"mle\"`` (default)
        picks the number with Minka's MLE; a ``float`` in ``(0, 1)`` keeps the
        components that explain that fraction of variance; an ``int`` fixes
        the count; ``None`` uses ``min(n_spectra, n_spectral)``.
    subtract_min
        If True (default, matches legacy ``pca_clean``), subtract the
        per-spectrum minimum *before* the fit so PCA models the spectral
        shape rather than offsets.
    restore_min
        If True, add the saved per-spectrum minimum back to the cleaned
        output. Off by default to match legacy ``pca_clean``; turn on to
        preserve absolute intensities.
    pca_kwargs
        Extra kwargs passed straight to :class:`sklearn.decomposition.PCA`
        (e.g. ``{\"svd_solver\": \"full\"}``).
    return_decomposition
        If True, also return a third dict with the components, per-spectrum
        coefficients, mean, and explained-variance arrays (large; not
        suitable for ``DataArray.attrs``).

    Returns
    -------
    cleaned
        Same shape and dtype-flavor (float) as ``values``.
    meta
        Small dict with the parameters actually used and summary stats —
        safe to attach to ``DataArray.attrs``.
    decomposition_payload
        Only when ``return_decomposition=True``. Has keys ``components``,
        ``coeffs``, ``mean``, ``explained_variance``,
        ``explained_variance_ratio``, ``noise_variance``.
    """
    row_stack, spatial_shape = _flatten_to_row_stack(values)
    n_spectra, n_spectral = row_stack.shape
    if n_spectra < 2:
        raise ValueError(
            "PCA denoising needs more than one spectrum; got "
            f"n_spectra={n_spectra}. For a single spectrum use a 1D "
            "smoother (e.g. Savitzky-Golay) instead."
        )

    if subtract_min:
        per_spec_min = _per_spectrum_min(row_stack)
        spectra_for_fit = row_stack - per_spec_min
    else:
        per_spec_min = np.zeros((n_spectra, 1), dtype=float)
        spectra_for_fit = row_stack.copy()

    pca = decomposition.PCA(n_components=n_components, **(pca_kwargs or {}))
    coeffs = pca.fit_transform(spectra_for_fit)
    cleaned_rows = pca.inverse_transform(coeffs)

    if restore_min:
        cleaned_rows = cleaned_rows + per_spec_min

    cleaned = cleaned_rows.reshape(spatial_shape + (n_spectral,))

    n_components_used = int(pca.n_components_)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    meta: dict[str, Any] = {
        "method": "pca",
        "n_components_requested": n_components,
        "n_components_used": n_components_used,
        "subtract_min": bool(subtract_min),
        "restore_min": bool(restore_min),
        "explained_variance_ratio_total": float(explained.sum()),
        "explained_variance_ratio_first_5": [float(v) for v in explained[:5]],
        "n_spectra": int(n_spectra),
        "n_spectral": int(n_spectral),
    }

    if not return_decomposition:
        return cleaned, meta

    coeffs_spatial = coeffs.reshape(spatial_shape + (n_components_used,))
    payload: dict[str, Any] = {
        "components": np.asarray(pca.components_, dtype=float).copy(),
        "coeffs": coeffs_spatial.copy(),
        "mean": np.asarray(pca.mean_, dtype=float).copy(),
        "explained_variance": np.asarray(
            pca.explained_variance_, dtype=float
        ).copy(),
        "explained_variance_ratio": explained.copy(),
        "noise_variance": float(getattr(pca, "noise_variance_", 0.0)),
        "per_spectrum_min": per_spec_min.reshape(spatial_shape + (1,)).copy(),
    }
    return cleaned, meta, payload


__all__ = ["denoise_spectra_pca", "NComponents"]
