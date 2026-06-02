# -*- coding: utf-8 -*-
"""
High-level spectral denoising: :class:`SpectraCleaner`.

PCA-based reconstruction for multi-spectrum inputs (2D/3D).  For 1-D single
spectra — or any input when ``per_spectrum=True`` — delegates automatically
to :class:`~wdfkit.spectra_smoother.SpectraSmoother` (Savitzky-Golay by
default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from .._shared._spectral import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from .._shared.clean_data import CleanData
from ..wdf._helpers.utils import ensure_in_memory
from ._pca import NComponents, denoise_spectra_pca

if TYPE_CHECKING:
    from ..spectra_smoother import SpectraSmoother

CleanMethod = Literal["pca"]

_TREATMENT_KEY = "spectra_cleaning"


@dataclass
class SpectraCleaner:
    """Denoise a population of spectra by low-rank PCA reconstruction.

    **Multi-spectrum inputs** (2D stacks ``(n_spectra, spectral)`` or 3D map
    cubes ``(y, x, spectral)``): uses PCA to separate shared signal from
    per-channel noise.

    **1-D single spectrum** ``(spectral,)`` or any input when
    ``per_spectrum=True``: delegates to
    :class:`~wdfkit.spectra_smoother.SpectraSmoother` (Savitzky-Golay by
    default).  Pass a pre-configured ``SpectraSmoother`` via the ``smoother``
    parameter to change the method or its settings.

    Parameters
    ----------
    method
        PCA denoising method. Currently only ``\"pca\"``; kept for forward
        compatibility.
    n_components
        Forwarded to :class:`sklearn.decomposition.PCA`. ``\"mle\"``
        (default), a ``float`` in ``(0, 1)`` for variance-explained,
        an ``int`` count, or ``None`` for ``min(n_spectra, n_spectral)``.
    subtract_min
        Subtract per-spectrum min before the PCA fit.
    restore_min
        Add the saved per-spectrum min back after reconstruction.
    spectral_dim
        Name of the spectral axis. Defaults to the last dimension.
    pca_kwargs
        Extra kwargs forwarded to :class:`sklearn.decomposition.PCA`.
    per_spectrum
        If ``True``, bypass PCA and apply ``smoother`` independently to
        every spectrum regardless of input dimensionality.  Useful when
        you want 1-D-style smoothing on a 2D/3D dataset.
    smoother
        A :class:`~wdfkit.spectra_smoother.SpectraSmoother` instance used
        for 1-D input and when ``per_spectrum=True``.  ``None`` (default)
        creates a ``SpectraSmoother()`` with Savitzky-Golay defaults.
    """

    method: CleanMethod = "pca"
    n_components: NComponents = "mle"
    subtract_min: bool = True
    restore_min: bool = False
    spectral_dim: str | None = None
    pca_kwargs: dict[str, Any] = field(default_factory=dict)
    per_spectrum: bool = False
    smoother: "SpectraSmoother | None" = None

    def __post_init__(self) -> None:
        allowed: tuple[str, ...] = ("pca",)
        if self.method not in allowed:
            raise ValueError(
                f"method must be one of {allowed!r}, got {self.method!r}"
            )
        if isinstance(self.n_components, float) and not (
            0.0 < self.n_components < 1.0
        ):
            raise ValueError(
                "float n_components must be in (0, 1) (variance ratio); "
                f"got {self.n_components}"
            )
        if isinstance(self.n_components, int) and self.n_components < 1:
            raise ValueError(
                f"int n_components must be >= 1, got {self.n_components}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, spectra: xr.DataArray) -> xr.DataArray:
        """Return a denoised copy of ``spectra`` (no decomposition payload)."""
        if not isinstance(spectra, xr.DataArray):
            raise TypeError(
                "SpectraCleaner.clean expects an xarray.DataArray; got "
                f"{type(spectra).__name__}"
            )
        spectra = CleanData(spectral_dim=self.spectral_dim).check(spectra)
        cleaned, meta, _ = self._clean_core(
            spectra, return_decomposition=False
        )
        return with_new_values(spectra, cleaned, _TREATMENT_KEY, meta)

    def clean_with_decomposition(
        self,
        spectra: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Like :meth:`clean`, but also returns the PCA decomposition.

        When the smoother path is taken (1-D input or ``per_spectrum=True``),
        the returned payload is ``{}`` — no decomposition is available for
        per-spectrum filtering.

        The PCA payload has keys ``components``, ``coeffs``, ``mean``,
        ``explained_variance``, ``explained_variance_ratio``,
        ``noise_variance``.
        """
        if not isinstance(spectra, xr.DataArray):
            raise TypeError(
                "SpectraCleaner.clean expects an xarray.DataArray; got "
                f"{type(spectra).__name__}"
            )
        spectra = CleanData(spectral_dim=self.spectral_dim).check(spectra)
        cleaned, meta, payload = self._clean_core(
            spectra, return_decomposition=True
        )
        out = with_new_values(spectra, cleaned, _TREATMENT_KEY, meta)
        return out, payload if payload is not None else {}

    def transform(self, spectra: xr.DataArray) -> xr.DataArray:
        """Alias of :meth:`clean`."""
        return self.clean(spectra)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_smoother(self) -> "SpectraSmoother":
        """Return the configured smoother, or a default one."""
        if self.smoother is not None:
            return self.smoother
        from ..spectra_smoother import SpectraSmoother

        return SpectraSmoother()

    def _clean_core(
        self,
        spectra: xr.DataArray,
        *,
        return_decomposition: bool,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None]:
        """Route to smoother or PCA, return ``(cleaned, meta, payload)``."""
        # 1-D single spectrum or explicit per-spectrum flag → smoother
        if spectra.ndim == 1 or self.per_spectrum:
            smoother = self._get_smoother()
            cleaned, meta = smoother._smooth_core(spectra)
            return cleaned, meta, None

        # Multi-spectrum PCA path
        sdim = resolve_spectral_dim(spectra, self.spectral_dim)
        da_w, orig_order = transpose_spectral_last(spectra, sdim)

        da_w = ensure_in_memory(
            da_w,
            caller="SpectraCleaner (PCA)",
            reason=(
                "PCA requires the full covariance matrix of all spectra "
                "and cannot be computed chunk-by-chunk."
            ),
            stacklevel=3,
        )

        spatial_shape = da_w.shape[:-1]
        n_spectra = int(np.prod(spatial_shape))
        if n_spectra < 2:
            raise ValueError(
                "SpectraCleaner needs more than one spectrum (PCA on a "
                "single spectrum is degenerate). Got input with shape "
                f"{tuple(spectra.shape)} → n_spectra={n_spectra} along "
                "non-spectral dims. Use per_spectrum=True or a "
                "SpectraSmoother for single-spectrum smoothing."
            )

        result = denoise_spectra_pca(
            da_w.values,
            n_components=self.n_components,
            subtract_min=self.subtract_min,
            restore_min=self.restore_min,
            pca_kwargs=self.pca_kwargs or None,
            return_decomposition=return_decomposition,
        )
        if return_decomposition:
            cleaned_w, meta, payload = result
        else:
            cleaned_w, meta = result
            payload = None
        meta = {**meta, "spectral_dim": sdim}

        cleaned_w_array = reshape_row_stack_to(
            cleaned_w.reshape(-1, cleaned_w.shape[-1]),
            da_w.shape,
        )
        if tuple(da_w.dims) != orig_order:
            cleaned_da = da_w.copy(data=cleaned_w_array).transpose(*orig_order)
            cleaned = cleaned_da.values
        else:
            cleaned = cleaned_w_array
        return cleaned, meta, payload


__all__ = ["SpectraCleaner", "CleanMethod"]
