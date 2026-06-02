# -*- coding: utf-8 -*-
"""Pre-read RAM availability check for WDF data arrays."""

from __future__ import annotations

import warnings

import psutil

from .parse_context import ParseContext

_WARN_THRESHOLD = 0.75  # warn when array > 75 % of available free RAM


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 2**30:.2f} GB"


def check_memory(ctx: ParseContext) -> None:
    """Raise or warn based on how the expected array size compares to
    available RAM.

    Called after ``WDF1`` and ``WMAP`` have been parsed, so ``ctx.nspectra``
    and ``ctx.npoints`` are already known.

    Behaviour
    ---------
    - **chunks already enabled** (``ctx.chunks`` is not ``False``): no check
      is performed — the data will be loaded lazily so the full array never
      needs to reside in RAM at parse time.
    - **chunks disabled** (default): raise ``MemoryError`` if the full array
      exceeds available RAM; emit ``UserWarning`` if it exceeds 75 % of
      available RAM.
    """
    expected_bytes = ctx.nspectra * ctx.npoints * 4  # float32
    mem = psutil.virtual_memory()
    free_ram = mem.available

    if ctx.chunks is not False:
        return

    if expected_bytes > free_ram:
        raise MemoryError(
            f"\nThe data array would require {_fmt_gb(expected_bytes)}"
            f"but only {_fmt_gb(free_ram)} of free RAM is available.\n\n"
            "Re-open the file with chunks=True to load it lazily:\n\n"
            f'    WDFReader("{ctx.filename}", chunks=True)\n'
        )

    if expected_bytes > free_ram * _WARN_THRESHOLD:
        warnings.warn(
            f"The spectral data array ({_fmt_gb(expected_bytes)}) will use "
            f"more than {int(_WARN_THRESHOLD * 100)} % of your available "
            f"free RAM ({_fmt_gb(free_ram)}). "
            "Consider using chunks=True to avoid memory pressure:\n\n"
            f'    WDFReader("{ctx.filename}", chunks=True)\n',
            UserWarning,
            stacklevel=4,
        )
