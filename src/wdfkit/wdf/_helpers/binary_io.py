# -*- coding: utf-8 -*-
"""Low-level binary helpers for WiRE ``.wdf`` streams."""

from __future__ import annotations

import numpy as np

# Block header layout: 4-char tag, int32 id, int64 byte size (WiRE layout).
BLOCK_HEADER_DTYPE = np.dtype(
    [
        ("block_name", "|S4"),
        ("block_id", np.int32),
        ("block_size", np.int64),
    ]
)


def read_from_file(file_obj, dtype=np.uint32, count=1):
    """Read primitive(s) from an open binary file (numpy ``fromfile``
    wrapper).

    Returns a scalar when ``count == 1``, otherwise a length-``count`` array.
    """
    if count == 1:
        return np.fromfile(file_obj, dtype=dtype, count=count)[0]
    return np.fromfile(file_obj, dtype=dtype, count=count)[0:count]
