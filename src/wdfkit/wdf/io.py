# -*- coding: utf-8 -*-
"""Orchestrate WiRE ``.wdf`` parsing: block index → parsers → xarray
assembly."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import xarray as xr

from ._helpers.block_index import scan_blocks
from ._helpers.memory_check import check_memory
from ._helpers.parse_context import ParseContext
from ._helpers.spectral import resolve_spectral_axis
from .blocks.bkxl import parse_bkxl
from .blocks.data import parse_data
from .blocks.orgn import parse_orgn, print_coord_lengths_if_verbose
from .blocks.text import parse_text
from .blocks.wdf1 import parse_wdf1
from .blocks.whtl import parse_whtl
from .blocks.wmap import parse_wmap
from .blocks.wxcs import parse_wxcs
from .blocks.wxda import parse_wxda
from .blocks.wxdm import parse_wxdm
from .blocks.wxis import parse_wxis
from .blocks.xlst import parse_xlst
from .blocks.ylst import parse_ylst
from .blocks.zldc import parse_zldc
from .error import WDFFormatError
from .parsed import (
    BKXLInfo,
    OrgnEntry,
    ParsedWDF,
    WMAPInfo,
    XLSTInfo,
    YLSTInfo,
)


def _ctx_to_parsed(ctx: ParseContext) -> ParsedWDF:
    """Build an immutable :class:`~wdfkit._parsed.ParsedWDF` from a filled
    :class:`ParseContext`."""
    # XLST — re-derive dim name and coord units from the spectral axis spec
    xlst_units = ctx.params.get("XlistDataUnits", "Arbitrary")
    xlst_dtype = ctx.params.get("XlistDataType", "Spectral")
    spectral_spec = resolve_spectral_axis(xlst_units, ctx.spectral_dim)
    new_sdim = spectral_spec.dim_name
    # Retrieve values from coord_dict (keyed by old dim name set by xlst.py)
    old_sdim = ctx.spectral_dim_name or "spectral"
    xlst_entry = ctx.coord_dict.get(old_sdim) or ctx.coord_dict.get(new_sdim)
    xlst_values = xlst_entry[1] if xlst_entry else np.array([])
    xlst = XLSTInfo(
        values=np.asarray(xlst_values, dtype="float32"),
        data_type=xlst_dtype,
        units=xlst_units,
        dim_name=new_sdim,
        coord_units=spectral_spec.units,
    )

    # YLST
    ylst: YLSTInfo | None = None
    if ctx.ylst_values is not None:
        ylst = YLSTInfo(
            values=np.asarray(ctx.ylst_values, dtype="float32"),
            data_type=ctx.ylst_data_type,
            units=ctx.ylst_units,
        )

    # BKXL
    bkxl: BKXLInfo | None = None
    if ctx.bkxl_values is not None:
        bkxl = BKXLInfo(
            values=np.asarray(ctx.bkxl_values, dtype="float32"),
            data_type=ctx.bkxl_data_type,
            units=ctx.bkxl_units,
        )

    # ORGN entries — pull from coord_dict (stores ("points", values, attrs))
    orgn_entries: list[OrgnEntry] = []
    for idx, label in enumerate(ctx.origin_labels):
        if label in ctx.coord_dict:
            _dim, values, attrs = ctx.coord_dict[label]
            is_primary = (
                ctx.origin_is_primary[idx]
                if idx < len(ctx.origin_is_primary)
                else False
            )
            orgn_entries.append(
                OrgnEntry(
                    label=label,
                    data_type=(
                        ctx.origin_set_dtypes[idx]
                        if idx < len(ctx.origin_set_dtypes)
                        else ""
                    ),
                    units=(
                        ctx.origin_set_units[idx]
                        if idx < len(ctx.origin_set_units)
                        else ""
                    ),
                    values=np.asarray(values),
                    is_primary=is_primary,
                )
            )

    # acquisition_time / end_time — from WDF1 header (already datetime)
    acquisition_time: datetime | None = ctx.params.get("StartTime")
    end_time: datetime | None = ctx.params.get("EndTime")

    # WMAP
    wmap: WMAPInfo | None = None
    mp = ctx.map_params
    if "NbSteps" in mp:
        wmap = WMAPInfo(
            flag=ctx.wmap_flag_raw,
            start_xyz=np.asarray(
                mp.get("InitialCoordinates", [0, 0, 0]), dtype="float32"
            ),
            step_xyz=np.asarray(
                mp.get("StepSizes", [0, 0, 0]), dtype="float32"
            ),
            nsteps=np.asarray(mp["NbSteps"], dtype="uint32"),
            linefocus_size=int(mp.get("LineFocusSize", 0)),
        )

    # file_uuid from WDF1 header
    file_uuid: str = ctx.params.get("FileUUID", "")

    return ParsedWDF(
        filename=str(ctx.filename),
        filesize=ctx.filesize,
        nspectra=ctx.nspectra,
        ncollected=ctx.ncollected,
        npoints=ctx.npoints,
        ylist_length=int(ctx.params.get("YlistLength", 1)),
        measurement_type=ctx.measurement_type_raw,
        scan_type=ctx.scan_type_raw,
        app_name=ctx.params.get("ApplicationName", ""),
        app_version=ctx.params.get("ApplicationVersion", ""),
        naccum=int(ctx.params.get("AccumulationCount", 1)),
        laser_wavelength=ctx.params.get("LaserWaveLength", "Unspecified"),
        title=ctx.params.get("Title", ""),
        spectral_units_str=ctx.params.get("SpectralUnits", ""),
        params=dict(ctx.params),
        map_params=dict(ctx.map_params),
        data=ctx.spectra if ctx.spectra is not None else None,
        xlst=xlst,
        ylst=ylst,
        orgn=tuple(orgn_entries),
        wmap=wmap,
        img=ctx.img,
        exposure_time=ctx.params.get("ExposureTime"),
        laser_power=ctx.params.get("LaserPower"),
        stage_xyz=ctx.stage_xyz,
        # new fields
        comment=ctx.comment,
        acquisition=ctx.acquisition,
        instrument_status=ctx.instrument_status,
        calibration=ctx.calibration,
        zeldac=ctx.zeldac,
        bkxl=bkxl,
        whtl_jpeg_bytes=ctx.whtl_jpeg_bytes,
        initial_coordinates=ctx.initial_coordinates,
        motor_positions=ctx.motor_positions,
        acquisition_time=acquisition_time,
        end_time=end_time,
        file_uuid=file_uuid,
    )


def _validate(ctx: ParseContext) -> None:
    """Run structural self-checks; raise :exc:`WDFFormatError` on failure."""
    blocks = ctx.blocks

    def block_idx(name: str) -> int | None:
        for i, b in enumerate(blocks):
            if b.name == name:
                return i
        return None

    # WDF1 must be first, size == 512
    wdf1_i = block_idx("WDF1")
    if wdf1_i is None or blocks[wdf1_i].size != 512:
        got = blocks[wdf1_i].size if wdf1_i is not None else "absent"
        raise WDFFormatError("WDF1 block size", 512, got)

    # DATA body size
    data_i = block_idx("DATA")
    if data_i is not None:
        expected_data = ctx.nspectra * ctx.npoints * 4
        got_data = blocks[data_i].size - 16
        if got_data != expected_data:
            raise WDFFormatError("DATA body size", expected_data, got_data)

    # XLST body size
    xlst_i = block_idx("XLST")
    if xlst_i is not None:
        expected_xlst = 8 + ctx.npoints * 4
        got_xlst = blocks[xlst_i].size - 16
        if got_xlst != expected_xlst:
            raise WDFFormatError("XLST body size", expected_xlst, got_xlst)

    # YLST body size
    ylst_i = block_idx("YLST")
    ylist_length = int(ctx.params.get("YlistLength", 1))
    if ylst_i is not None:
        expected_ylst = 8 + ylist_length * 4
        got_ylst = blocks[ylst_i].size - 16
        if got_ylst != expected_ylst:
            raise WDFFormatError("YLST body size", expected_ylst, got_ylst)

    # ORGN body size
    orgn_i = block_idx("ORGN")
    origin_count = int(ctx.params.get("DataOriginCount", 0))
    if orgn_i is not None and origin_count > 0:
        expected_orgn = 4 + origin_count * (24 + ctx.nspectra * 8)
        got_orgn = blocks[orgn_i].size - 16
        if got_orgn != expected_orgn:
            raise WDFFormatError("ORGN body size", expected_orgn, got_orgn)

    # WMAP presence matches measurement_type == 3
    wmap_present = block_idx("WMAP") is not None
    is_map = ctx.measurement_type_raw == 3
    if is_map and not wmap_present:
        raise WDFFormatError("WMAP block for Map scan", "present", "absent")
    if not is_map and wmap_present:
        raise WDFFormatError(
            "WMAP block for non-Map scan", "absent", "present"
        )

    # WMAP body must be 48 bytes
    wmap_i = block_idx("WMAP")
    if wmap_i is not None and blocks[wmap_i].size - 16 != 48:
        raise WDFFormatError("WMAP body size", 48, blocks[wmap_i].size - 16)

    # Sum of all block sizes == file size
    total = sum(b.size for b in blocks)
    if total != ctx.filesize:
        raise WDFFormatError("sum of block sizes", ctx.filesize, total)


def _run_parsers(
    ctx: ParseContext,
    *,
    load_data: bool = True,
) -> None:
    """Run block parsers against *ctx*.

    When *load_data* is ``False`` the DATA, WHTL, ORGN, and PSET
    blocks are skipped (used by :func:`parse_wdf_header` for fast
    classification).
    """
    ctx.blocks = scan_blocks(ctx.f, ctx.filesize)
    parse_wdf1(ctx)
    parse_wmap(ctx)
    parse_text(ctx)
    parse_wxdm(ctx)
    parse_wxis(ctx)
    if load_data:
        check_memory(ctx)
        parse_data(ctx)
        parse_xlst(ctx)
        parse_ylst(ctx)
        parse_whtl(ctx)
        parse_orgn(ctx)
        parse_wxda(ctx)
        parse_wxcs(ctx)
        parse_zldc(ctx)
        parse_bkxl(ctx)
        _validate(ctx)
    else:
        parse_xlst(ctx)


def parse_wdf_to_parsed(
    filename: str | os.PathLike[str],
    verbose: bool = False,
    time_coord: str | None = None,
    spectral_dim: str | None = None,
    chunks: bool | int = False,
) -> ParsedWDF:
    """Parse a WDF file and return a :class:`~wdfkit._parsed.ParsedWDF`.

    This is the low-level entry point used by handlers and
    :func:`classify_wdf`.
    """
    try:
        file_obj = open(filename, "rb")
        if verbose:
            from pathlib import Path

            print(f'Reading the file: "{Path(filename).name}"\n')
    except IOError as e:
        raise IOError(f"File {filename} does not exist!") from e

    filesize = os.path.getsize(filename)
    ctx = ParseContext(
        filename=filename,
        verbose=verbose,
        time_coord=time_coord,
        spectral_dim=spectral_dim,
        filesize=filesize,
        f=file_obj,
        chunks=chunks,
    )
    try:
        _run_parsers(ctx, load_data=True)
        if verbose:
            print_coord_lengths_if_verbose(ctx)
        return _ctx_to_parsed(ctx)
    finally:
        file_obj.close()


def parse_wdf_header(
    filename: str | os.PathLike[str],
    verbose: bool = False,
    spectral_dim: str | None = None,
) -> ParsedWDF:
    """Parse only the header blocks of a WDF file (no DATA).

    Used by :func:`wdfkit.classify` for fast scan-type triage.
    """
    try:
        file_obj = open(filename, "rb")
    except IOError as e:
        raise IOError(f"File {filename} does not exist!") from e

    filesize = os.path.getsize(filename)
    ctx = ParseContext(
        filename=filename,
        verbose=verbose,
        time_coord=None,
        spectral_dim=spectral_dim,
        filesize=filesize,
        f=file_obj,
        chunks=False,
    )
    try:
        _run_parsers(ctx, load_data=False)
        return _ctx_to_parsed(ctx)
    finally:
        file_obj.close()


def read_wdf_file(
    filename: str | os.PathLike[str],
    verbose: bool,
    time_coord: str | None,
    spectral_dim: str | None = None,
    chunks: bool | int = False,
) -> tuple[xr.DataArray, object]:
    """Parse a WiRE WDF file (invoked by
    :class:`~wdfkit.reader.WDFReader`).

    Parameters
    ----------
    spectral_dim
        Passed to :func:`~wdfkit.spectral.resolve_spectral_axis` during
        ``XLST`` handling.
    chunks
        ``False`` for eager NumPy reading (default); ``True`` or an ``int``
        MB value for lazy Dask-backed reading.
    """
    from .dispatch import dispatch

    parsed = parse_wdf_to_parsed(
        filename,
        verbose=verbose,
        time_coord=time_coord,
        spectral_dim=spectral_dim,
        chunks=chunks,
    )
    return dispatch(parsed), parsed.img
