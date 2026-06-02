# -*- coding: utf-8 -*-
"""Handler for ``kind="linefocus"`` (flag & 0x08).

In a LineFocus scan each spectrum is a full CCD-column readout: the
DATA block stores ``nspectra × ylist_length × xlist_length`` float32
values in order (spectrum_idx, line_y, spectral_channel).  YLST
provides the spatial positions along the CCD column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 4-D DataArray ``(y, x, line_y, spectral_dim)``.

    If WMAP is present the spatial (y, x) dims are filled from ORGN
    SpatialX / SpatialY.  Otherwise ``x`` / ``y`` dims are integer
    indices.

    .. note::
        Not yet supported. Raises :exc:`NotImplementedError`.
    """
    raise NotImplementedError(
        "LineFocus scans (kind='linefocus', 4-D output y/x/line_y/spectral) "
        "are not supported yet."
    )
    sdim = parsed.xlst.dim_name
    assert parsed.wmap is not None, "linefocus requires a WMAP block"
    nx_map = int(parsed.wmap.nsteps[0])
    ny_map = int(parsed.wmap.nsteps[1])
    ylist_len = parsed.ylist_length
    npts = parsed.npoints
    nspectra = parsed.nspectra

    data = np.asarray(parsed.data)

    # DATA may be flat (nspectra * ylist_len * npts) or already per-spectrum.
    expected_total = nspectra * ylist_len * npts
    if data.size == expected_total:
        cube = data.reshape(nspectra, ylist_len, npts)
    else:
        # Fallback: treat as (nspectra, npts)
        cube = data.reshape(nspectra, 1, npts)
        ylist_len = 1

    # Line-focus Y coordinate (CCD rows)
    if parsed.ylst is not None:
        line_y_vals = parsed.ylst.values
    else:
        line_y_vals = np.arange(ylist_len, dtype="float32")

    # Reshape spatial dims if grid is known
    sc = spectral_coord(parsed)
    coords: dict = {
        sdim: (sdim, sc[1], sc[2]),
        "line_y": line_y_vals,
    }
    if nx_map * ny_map == nspectra:
        cube = cube.reshape(ny_map, nx_map, ylist_len, npts)
        orgn_x = parsed.orgn_by_type("SpatialX")
        orgn_y = parsed.orgn_by_type("SpatialY")
        if orgn_x is not None:
            x_2d = orgn_x.values[: ny_map * nx_map].reshape(ny_map, nx_map)
            coords["x"] = ("x", x_2d[0, :])
        if orgn_y is not None:
            y_2d = orgn_y.values[: ny_map * nx_map].reshape(ny_map, nx_map)
            coords["y"] = ("y", y_2d[:, 0])

        attrs = make_attrs(parsed, "linefocus")
        attrs["ScanShape"] = (ny_map, nx_map)
        attrs["RowCoord"] = "y"
        attrs["ColCoord"] = "x"
        return sort_spectral(
            xr.DataArray(
                cube,
                dims=("y", "x", "line_y", sdim),
                coords=coords,
                attrs=attrs,
            )
        )
    else:
        coords["point"] = np.arange(nspectra)
        attrs = make_attrs(parsed, "linefocus")
        attrs["ScanShape"] = (nspectra, 1)
        attrs["RowCoord"] = "point"
        attrs["ColCoord"] = None
        return sort_spectral(
            xr.DataArray(
                cube,
                dims=("point", "line_y", sdim),
                coords=coords,
                attrs=attrs,
            )
        )
