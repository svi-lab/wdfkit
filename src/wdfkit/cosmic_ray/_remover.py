# -*- coding: utf-8 -*-
"""High-level cosmic-ray removal: :class:`CosmicRayRemover`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .._shared._spectral import resolve_spectral_dim, with_new_values
from .._shared.clean_data import CleanData
from ..wdf._helpers.utils import ensure_in_memory
from ._1d import remove_cosmic_rays_1d
from ._harmonic import harmonic_correct_dataarray
from ._map import (
    correct_cosmic_rays_collection,
    correct_cosmic_rays_on_map_cube,
)

# ---------------------------------------------------------------------------
# Internal tuning constants (not exposed as user parameters)
# ---------------------------------------------------------------------------

_MAP_MAD_MULTIPLIER: float = 7.0
_MAP_NOISY_RELAX_MIN: float = 0.82
_MAP_MIN_RESIDUAL_OVER_CUTOFF: float = 1.05
_MAP_REQUIRE_SPATIAL_LOCAL_MAX: bool = False

# Below this many spectra → apply 1D engine independently per spectrum;
# at or above → use the collection (global-median / PCA) engine.
_COLLECTION_THRESHOLD: int = 20


@dataclass
class CosmicRayRemover:
    """Cosmic-ray removal with automatic routing by data dimensionality.

    **1D (single spectrum)** — always uses the 1D medfilt + MAD engine
    controlled by :attr:`spike_width`, :attr:`spike_threshold`,
    :attr:`spike_passes`.

    **2D (line scan / series / point collection)**

    * fewer than 20 spectra → 1D engine applied independently to each
      spectrum (no population statistics yet).
    * 20 or more spectra → *collection engine*: global median or PCA
      reconstruction as reference; :attr:`map_method` selects which.

    **3D (spatial map)**

    * fewer than 20 spectra → same per-spectrum 1D path as above.
    * 20 or more → spatial disk-median engine (``map_method="median"``,
      default) or PCA engine (``map_method="pca"``).  The disk-median path
      additionally respects :attr:`map_sensitivity` and
      :attr:`map_disk_radius`.

    Optionally removes broad Nd:YAG harmonics before spike removal via
    :meth:`harmonic_check` / :meth:`remove`.

    Parameters
    ----------
    spike_width
        **1D engine** — odd integer ≥ 3.  Sets the ``medfilt`` window in
        spectral channels.  Raise to 9–13 when cosmic rays span 7–10
        channels; keep at 5 for narrow single-channel spikes.
    spike_threshold
        **1D engine** — positive float.  Spike cutoff = ``spike_threshold ×
        MAD_noise``.  Lower → more aggressive.  Raise to 5–6 for very
        noisy spectra to avoid false positives.
    spike_passes
        **1D engine** — integer ≥ 1.  Iterations of detect → repair.  Each
        pass works on the already-repaired signal so that large spikes no
        longer mask smaller ones.
    map_sensitivity
        **3D disk-median engine only** — scales overall aggressiveness.
        Larger → more hits (default 0.01).
    map_disk_radius
        **3D disk-median engine only** — spatial disk radius for the
        reference median filter (pixels).
    map_spike_width
        **Collection / 3D engines** — spectral dilation in channels added
        around each detected hit (integer ≥ 1).  Increase for broader
        cosmic rays (e.g. ``9``–``15`` for multi-channel spikes).  The
        repair region is capped at ``2 × map_spike_width`` channels.
    map_method
        ``"median"`` (default): global median spectrum as reference for 2D;
        spatial disk-median for 3D.
        ``"pca"``: PCA reconstruction as reference for both 2D and 3D.
    map_n_components
        **PCA path only** — number of principal components for the
        reconstruction reference.  3–5 covers most real samples; increase
        for multi-phase or compositionally diverse maps.
    spectral_dim
        Name of the spectral axis (default: last dimension).
    """

    # --- 1D engine ---
    spike_width: int = 5
    spike_threshold: float = 3.5
    spike_passes: int = 3

    # --- collection / 3D engine ---
    map_sensitivity: float = 0.01
    map_disk_radius: int = 3
    map_spike_width: int = 5
    map_method: str = "median"
    map_n_components: int = 3

    # --- shared ---
    spectral_dim: str | None = None

    def __post_init__(self) -> None:
        if self.spike_width < 3 or self.spike_width % 2 == 0:
            raise ValueError(
                f"spike_width must be odd and >= 3, got {self.spike_width}"
            )
        if self.spike_threshold <= 0 or not np.isfinite(self.spike_threshold):
            raise ValueError("spike_threshold must be positive and finite")
        if self.spike_passes < 1:
            raise ValueError("spike_passes must be >= 1")
        if self.map_sensitivity <= 0:
            raise ValueError("map_sensitivity must be > 0")
        if self.map_spike_width < 1:
            raise ValueError("map_spike_width must be >= 1")
        if self.map_disk_radius < 1:
            raise ValueError("map_disk_radius must be >= 1")
        if self.map_method not in ("median", "pca"):
            raise ValueError(
                f"map_method must be 'median' or 'pca', "
                f"got {self.map_method!r}"
            )
        if self.map_n_components < 1:
            raise ValueError("map_n_components must be >= 1")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harmonic_check(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Notch broad harmonics when ``LaserWaveLength`` is ~355 nm
        (Nd:YAG).

        If ``spectrum.attrs['LaserWaveLength']`` is outside 354–356 nm,
        returns ``spectrum`` unchanged.

        Searches 1064 / 532 / 355 / 266 nm (±2.5 nm); replaces ~1 nm
        around each found peak with linear interpolation.
        """
        return harmonic_correct_dataarray(
            spectrum,
            spectral_dim=self.spectral_dim,
        )

    def remove_cosmic_rays(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Spike removal only (no harmonic notch)."""
        out, _ = self._route(spectrum, want_diagnostics=False)
        return out

    def remove_cosmic_rays_with_diagnostics(
        self,
        spectrum: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Like :meth:`remove_cosmic_rays`, but also returns a diagnostics
        dict for visualization / QC (not written to ``DataArray.attrs``).

        Diagnostics keys depend on the engine used:

        * **1D**: ``"cosmic_mask"``, ``"corrected_1d"``
        * **loop-1D** (< 20 spectra, 2D/3D): ``"cosmic_masks"``
        * **collection** (≥ 20 spectra, 2D or 3D PCA): ``"core_mask"``,
          ``"repair_mask"``, ``"residual"``, ``"reference"``,
          ``"noise_per_channel"``, ``"cutoff"``
        * **3D disk-median**: same as current map diagnostics
          (``"core_mask"``, ``"repair_mask"``, ``"residual"``,
          ``"preprocessed"``, ``"spatial_median_reference"``, etc.)
        """
        return self._route(spectrum, want_diagnostics=True)

    def remove(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Harmonic cleanup first, then cosmic-ray removal."""
        return self.remove_cosmic_rays(self.harmonic_check(spectrum))

    def remove_with_diagnostics(
        self,
        spectrum: xr.DataArray,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Harmonics, then :meth:`remove_cosmic_rays_with_diagnostics`."""
        return self.remove_cosmic_rays_with_diagnostics(
            self.harmonic_check(spectrum)
        )

    def transform(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Alias of :meth:`remove` (harmonics then cosmic rays)."""
        return self.remove(spectrum)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Dispatch to the correct engine based on shape and spectrum count."""
        da = CleanData(spectral_dim=self.spectral_dim).check(da)
        ndim = da.ndim

        if ndim == 1:
            resolve_spectral_dim(da, self.spectral_dim)
            return self._apply_1d(da, np.asarray(da.values, dtype=float))

        if ndim == 2:
            n_spectra = da.shape[0]
            if n_spectra <= 1:
                return self._apply_1d(
                    da,
                    np.asarray(da.values, dtype=float).reshape(-1),
                    want_diagnostics=want_diagnostics,
                )
            if n_spectra < _COLLECTION_THRESHOLD:
                return self._apply_loop_1d(
                    da, want_diagnostics=want_diagnostics
                )
            return self._apply_collection(
                da, want_diagnostics=want_diagnostics
            )

        if ndim == 3:
            n_spectra = da.shape[0] * da.shape[1]
            if n_spectra < _COLLECTION_THRESHOLD:
                return self._apply_loop_1d(
                    da, want_diagnostics=want_diagnostics
                )
            da = self._maybe_compute_for_map(da)
            return self._apply_map(da, want_diagnostics=want_diagnostics)

        raise ValueError(
            "CosmicRayRemover supports 1-D (n_spectral,), "
            "2-D (n_spatial, n_spectral), or 3-D (ny, nx, n_spectral); "
            f"got ndim={da.ndim}, shape={da.shape}"
        )

    # ------------------------------------------------------------------
    # Engines
    # ------------------------------------------------------------------

    def _apply_1d(
        self,
        da: xr.DataArray,
        arr_1d: np.ndarray,
        *,
        want_diagnostics: bool = True,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        resolve_spectral_dim(da, self.spectral_dim)
        corrected, mask = remove_cosmic_rays_1d(
            arr_1d,
            kernel_size=self.spike_width,
            threshold=self.spike_threshold,
            max_passes=self.spike_passes,
        )
        meta = self._meta_1d(mask)
        out = with_new_values(
            da, corrected.reshape(da.shape), "Cosmic Ray Correction", meta
        )
        diag = (
            {"cosmic_mask": mask, "corrected_1d": corrected}
            if want_diagnostics
            else {}
        )
        return out, diag

    def _apply_loop_1d(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Apply the 1D engine independently to every spectrum."""
        arr = np.asarray(da.values, dtype=float)
        orig_shape = arr.shape
        flat = arr.reshape(-1, orig_shape[-1])
        out_flat = flat.copy()
        masks = np.zeros_like(flat, dtype=bool) if want_diagnostics else None
        n_corrected = 0
        for i, row in enumerate(flat):
            corrected, mask = remove_cosmic_rays_1d(
                row,
                kernel_size=self.spike_width,
                threshold=self.spike_threshold,
                max_passes=self.spike_passes,
            )
            out_flat[i] = corrected
            if np.any(mask):
                n_corrected += 1
            if masks is not None:
                masks[i] = mask
        meta = {
            "spike_width": self.spike_width,
            "spike_threshold": self.spike_threshold,
            "spike_passes": self.spike_passes,
            "spectra_corrected": n_corrected,
        }
        out = with_new_values(
            da, out_flat.reshape(orig_shape), "Cosmic Ray Correction", meta
        )
        if want_diagnostics and masks is not None:
            diag: dict[str, Any] = {"cosmic_masks": masks.reshape(orig_shape)}
        else:
            diag = {}
        return out, diag

    def _apply_collection(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Global-median or PCA engine for 2D with ≥ 20 spectra."""
        result = correct_cosmic_rays_collection(
            np.asarray(da.values, dtype=float),
            method=self.map_method,
            threshold=self.spike_threshold,
            spectral_dilate_channels=self.map_spike_width,
            max_repair_extent=self.map_spike_width * 2,
            n_components=self.map_n_components,
            return_diagnostics=want_diagnostics,
        )
        if want_diagnostics:
            corrected, meta, diag = result  # type: ignore[misc]
        else:
            corrected, meta = result  # type: ignore[misc]
            diag = {}
        return (
            with_new_values(da, corrected, "Cosmic Ray Correction", meta),
            diag,
        )

    def _apply_map(
        self,
        da: xr.DataArray,
        *,
        want_diagnostics: bool,
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Spatial disk-median or PCA engine for 3D maps with ≥ 20 spectra."""
        if self.map_method == "pca":
            result = correct_cosmic_rays_collection(
                np.asarray(da.values, dtype=float),
                method="pca",
                threshold=self.spike_threshold,
                spectral_dilate_channels=self.map_spike_width,
                max_repair_extent=self.map_spike_width * 2,
                n_components=self.map_n_components,
                return_diagnostics=want_diagnostics,
            )
            if want_diagnostics:
                corrected, meta, diag = result  # type: ignore[misc]
            else:
                corrected, meta = result  # type: ignore[misc]
                diag = {}
            return (
                with_new_values(da, corrected, "Cosmic Ray Correction", meta),
                diag,
            )

        # Default: spatial disk-median
        result_map = correct_cosmic_rays_on_map_cube(
            da.values,
            sensitivity=self.map_sensitivity,
            spectral_dilate_channels=self.map_spike_width,
            disk_radius=self.map_disk_radius,
            map_mad_multiplier=_MAP_MAD_MULTIPLIER,
            map_noisy_channel_relax_min=_MAP_NOISY_RELAX_MIN,
            map_max_spectral_repair_extent=self.map_spike_width * 2,
            map_min_residual_over_cutoff=_MAP_MIN_RESIDUAL_OVER_CUTOFF,
            map_require_spatial_local_max=_MAP_REQUIRE_SPATIAL_LOCAL_MAX,
            return_diagnostic_masks=want_diagnostics,
        )
        if want_diagnostics:
            corrected_m, meta_m, diag_m = result_map  # type: ignore[misc]
        else:
            corrected_m, meta_m = result_map  # type: ignore[misc]
            diag_m = {}
        return (
            with_new_values(da, corrected_m, "Cosmic Ray Correction", meta_m),
            diag_m,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _meta_1d(self, mask: np.ndarray) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "spike_width": self.spike_width,
            "spike_threshold": self.spike_threshold,
            "spike_passes": self.spike_passes,
        }
        if np.any(mask):
            meta["CRs found (spectral indices)"] = list(np.flatnonzero(mask))
        return meta

    @staticmethod
    def _da_label(da: xr.DataArray) -> str:
        parts: list[str] = []
        if da.name:
            parts.append(f"name={da.name!r}")
        for key in ("Title", "filename", "source"):
            if key in da.attrs:
                parts.append(f"{key}={da.attrs[key]!r}")
                break
        return ", ".join(parts) if parts else "unnamed DataArray"

    @staticmethod
    def _maybe_compute_for_map(da: xr.DataArray) -> xr.DataArray:
        return ensure_in_memory(
            da,
            caller="CosmicRayRemover",
            reason=(
                "The spatial disk-median algorithm requires all pixels in "
                "memory simultaneously.\n"
                "If this causes an out-of-memory error, consider splitting "
                "the map into sub-regions before CR removal."
            ),
            stacklevel=3,
        )


__all__ = ["CosmicRayRemover", "remove_cosmic_rays_1d"]
