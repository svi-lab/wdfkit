# -*- coding: utf-8 -*-
"""Backward-compatibility shim for ``wdfkit.preprocessing``.

.. deprecated::
    Import directly from the new locations:

    * ``normalize``           → :func:`wdfkit.normalize`
    * ``denoise_spectra_pca`` → :mod:`wdfkit.spectra_cleaner._pca`
    * ``cosmic_ray_*``        → :mod:`wdfkit.cosmic_ray`
    * ``smooth_1d``           → :mod:`wdfkit.spectra_smoother._smooth_1d`
"""

from .._shared.normalize import normalize
from ..spectra_cleaner._pca import denoise_spectra_pca

__all__ = ["normalize", "denoise_spectra_pca"]
