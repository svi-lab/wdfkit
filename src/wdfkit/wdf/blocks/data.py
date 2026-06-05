# -*- coding: utf-8 -*-
"""Parse ``DATA`` spectral intensity block.

Supports both eager (NumPy) and lazy (Dask) reading.  When ``ctx.chunks``
is not ``False``, the DATA block is never fully read into memory; instead a
:class:`dask.array.Array` of shape ``(nspectra, npoints)`` is built from
one ``dask.delayed`` slice per chunk.  Each chunk aligns to a whole number
of Y-rows so that the subsequent reshape in ``assemble.py`` produces clean
``(chunk_y, nx, npoints)`` chunks with no cross-row fragment.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named

if TYPE_CHECKING:
    from .._helpers.parse_context import ParseContext

# 16-byte block header that precedes the float32 payload in every WDF block.
_BLOCK_HEADER_BYTES = 16

# Default target size per chunk when chunks=True.
_DEFAULT_TARGET_MB = 128

# Maximum number of chunks to create (keeps scheduler overhead bounded).
_MAX_CHUNKS = 20


# ---------------------------------------------------------------------------
# Lazy-reading helpers
# ---------------------------------------------------------------------------


def _read_spectra_rows(
    filename: str,
    data_offset: int,
    start_spectrum: int,
    n_spectra: int,
    npoints: int,
) -> np.ndarray:
    """Read a contiguous slice of spectra from the DATA block.

    Opens the file independently so each Dask worker can call this without
    sharing a file handle.

    Parameters
    ----------
    filename:
        Absolute path to the ``.wdf`` file.
    data_offset:
        Byte position of the first float32 value in the DATA block
        (i.e. block start + 16-byte header).
    start_spectrum:
        Index of the first spectrum in this slice.
    n_spectra:
        Number of spectra to read.
    npoints:
        Spectral channels per spectrum.

    Returns
    -------
    ``np.ndarray`` of shape ``(n_spectra, npoints)``, dtype ``float32``.
    """
    byte_offset = data_offset + start_spectrum * npoints * 4
    n_bytes = n_spectra * npoints * 4
    with open(filename, "rb") as f:
        f.seek(byte_offset)
        raw = f.read(n_bytes)
    return np.frombuffer(raw, dtype="<f4").reshape(n_spectra, npoints)


def _compute_chunk_spec(
    nspectra: int,
    npoints: int,
    target_mb: int,
    nx: int,
) -> int:
    """Return the number of spectra per chunk, aligned to whole Y-rows.

    The result satisfies two constraints simultaneously:

    1. ``chunk_bytes ≤ target_mb * 2**20``  (memory cap)
    2. ``n_chunks ≤ _MAX_CHUNKS``           (scheduler-overhead cap)

    Taking the *maximum* of the two individual chunk sizes (not the minimum)
    ensures the stricter constraint — fewer, larger chunks — wins.  Chunks
    are always a multiple of ``nx`` so they map to complete rows.

    Parameters
    ----------
    nspectra:
        Total number of spectra  (= ny * nx for maps).
    npoints:
        Spectral channels per spectrum.
    target_mb:
        Target MB per chunk.
    nx:
        Number of X pixels per row (1 for series scans).
    """
    bytes_per_row = nx * npoints * 4

    # Rows to stay within the memory target.
    rows_by_memory = max(1, int(target_mb * 2**20 / bytes_per_row))

    # Rows to stay within the chunk-count cap.
    ny = math.ceil(nspectra / nx)
    rows_by_count = max(1, math.ceil(ny / _MAX_CHUNKS))

    chunk_rows = max(rows_by_memory, rows_by_count)
    chunk_rows = min(chunk_rows, ny)  # never exceed the full height

    spectra_per_chunk = chunk_rows * nx
    return spectra_per_chunk


def _build_lazy_spectra(
    ctx: "ParseContext",
    data_offset: int,
    target_mb: int,
):
    """Build a lazy ``dask.array`` for the DATA block.

    The returned array has shape ``(nspectra, npoints)`` and dtype
    ``float32``.  Zero-padding for incomplete recordings is applied at
    compute time (the trailing zeros are represented as a zero-filled
    delayed chunk, avoiding any disk read for missing data).
    """
    import dask
    import dask.array as da

    nspectra = ctx.nspectra
    ncollected = ctx.ncollected
    npoints = ctx.npoints
    filename = ctx.filename

    is_map = ctx.params.get("MeasurementType", "").lower().startswith("map")
    if is_map and "NbSteps" in ctx.map_params:
        nx = int(ctx.map_params["NbSteps"][0])
        if nx <= 0 or nspectra % nx != 0:
            nx = 1  # fall back to per-spectrum chunking if nx is inconsistent
    else:
        nx = 1

    spectra_per_chunk = _compute_chunk_spec(nspectra, npoints, target_mb, nx)

    pieces: list = []
    for start in range(0, nspectra, spectra_per_chunk):
        end = min(start + spectra_per_chunk, nspectra)
        chunk_len = end - start

        if start < ncollected:
            # At least part of this chunk has recorded data.
            readable_end = min(end, ncollected)
            readable_len = readable_end - start

            if readable_len == chunk_len:
                # Fully recorded chunk — single delayed read.
                delayed_chunk = dask.delayed(_read_spectra_rows)(
                    filename,
                    data_offset,
                    start,
                    readable_len,
                    npoints,
                )
                piece = da.from_delayed(
                    delayed_chunk,
                    shape=(chunk_len, npoints),
                    dtype=np.float32,
                )
            else:
                # Partially recorded: read the recorded part, pad with zeros.
                delayed_chunk = dask.delayed(_read_spectra_rows)(
                    filename,
                    data_offset,
                    start,
                    readable_len,
                    npoints,
                )
                recorded = da.from_delayed(
                    delayed_chunk,
                    shape=(readable_len, npoints),
                    dtype=np.float32,
                )
                pad_len = chunk_len - readable_len
                zeros = da.zeros((pad_len, npoints), dtype=np.float32)
                piece = da.concatenate([recorded, zeros], axis=0)
        else:
            # Entirely beyond the recorded range — pure zero padding.
            piece = da.zeros((chunk_len, npoints), dtype=np.float32)

        pieces.append(piece)

    return da.concatenate(pieces, axis=0)


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------


def parse_data(ctx: "ParseContext") -> None:
    """Fill ``ctx.spectra`` with spectral intensities from the DATA block.

    When ``ctx.chunks`` is ``False`` (default) the block is read eagerly into
    a NumPy array of shape ``(nspectra, npoints)``.

    When ``ctx.chunks`` is ``True`` or a positive integer, a lazy
    ``dask.array`` of the same shape is built instead — no data is read from
    disk until ``.compute()`` is called (or an operation triggers it).
    """
    name = "DATA"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        data_offset = ctx.blocks[i].offset + _BLOCK_HEADER_BYTES

        if ctx.chunks is not False:
            target_mb = (
                _DEFAULT_TARGET_MB if ctx.chunks is True else int(ctx.chunks)
            )
            ctx.spectra = _build_lazy_spectra(ctx, data_offset, target_mb)

            if ctx.verbose:
                n_chunks = (
                    len(ctx.spectra.chunks[0]) if ctx.spectra.chunks else 1
                )
                print(
                    f'{"Lazy Dask array (chunks)":-<40s} : \t'
                    f"shape={ctx.spectra.shape}, "
                    f"n_chunks={n_chunks}, "
                    f"target={target_mb} MB"
                )
        else:
            data_points_count = ctx.ncollected * ctx.npoints
            ctx.spectra = np.zeros((ctx.nspectra, ctx.npoints))
            ctx.f.seek(data_offset)
            ctx.spectra[: ctx.ncollected] = read_from_file(
                ctx.f, "<f", count=data_points_count
            ).reshape(ctx.ncollected, ctx.npoints)

            if ctx.verbose:
                print(f'{"The number of spectra":-<40s} : \t{ctx.ncollected}')
                print(
                    f'{"The number of points in each spectra":-<40s} : \t'
                    f"{ctx.npoints}"
                )
