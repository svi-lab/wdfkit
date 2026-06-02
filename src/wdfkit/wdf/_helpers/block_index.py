# -*- coding: utf-8 -*-
"""First-pass scan: block names, sizes, and file offsets."""

from __future__ import annotations

import numpy as np

from .binary_io import BLOCK_HEADER_DTYPE


def scan_blocks(file_obj, filesize: int) -> dict:
    """Walk the file and collect each block's tag, size, and starting
    offset."""
    blocks: dict = {
        "BlockNames": [],
        "BlockSizes": [],
        "BlockOffsets": [],
    }
    offset = 0
    while offset < filesize - 1:
        file_obj.seek(offset)
        # Use ``np.fromfile`` so ``count=1`` yields a length-1 structured array
        # (consistent indexing vs scalar ``numpy.void`` from helpers).
        block_header = np.fromfile(file_obj, dtype=BLOCK_HEADER_DTYPE, count=1)
        block_size = block_header["block_size"][0]
        if block_size == 0:
            break
        blocks["BlockOffsets"].append(offset)
        blocks["BlockNames"].append(block_header["block_name"][0].decode())
        blocks["BlockSizes"].append(block_size)
        offset += block_size
    return blocks


def indices_named(blocks: dict, name: str) -> list[int]:
    """Return indices of blocks whose four-letter tag equals
    ``name``."""
    return [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
