#!/usr/bin/env python
##############################################################################
#
# (c) 2026 SVI - Saint-Gobain Research Paris.
# All rights reserved.
#
# File coded by: Danila Shiryaev.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/dshirya/wdfkit/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Python package for WDF data treatment."""

# package version
from wdfkit.version import __version__  # noqa: F401

from ._shared.normalize import normalize  # noqa: F401
from .catalog import catalog  # noqa: F401
from .cosmic_ray import CosmicRayRemover, remove_cosmic_rays_1d  # noqa: F401
from .reader import WDFReader, classify, read  # noqa: F401
from .spectra_cleaner import SpectraCleaner  # noqa: F401
from .spectra_smoother import SpectraSmoother  # noqa: F401

__all__ = [
    "WDFReader",
    "read",
    "classify",
    "catalog",
    "normalize",
    "CosmicRayRemover",
    "remove_cosmic_rays_1d",
    "SpectraCleaner",
    "SpectraSmoother",
]

# End of file
