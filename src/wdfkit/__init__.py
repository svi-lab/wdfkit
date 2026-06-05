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

from .catalog import catalog  # noqa: F401
from .reader import WDFReader, classify, read  # noqa: F401

__all__ = [
    "WDFReader",
    "read",
    "classify",
    "catalog",
]

# End of file
