# -*- coding: utf-8 -*-
"""Parse ``WDF1`` main header block."""

from __future__ import annotations

import numpy as np

from .._helpers import constants as const
from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from .._helpers.utils import convert_time


def parse_wdf1(ctx: ParseContext) -> None:
    name = "WDF1"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        ctx.params["WdfFlag"] = const.WDF_FLAGS[
            read_from_file(ctx.f, dtype=np.uint64)
        ]
        ctx.f.seek(60)
        ctx.params["PointsPerSpectrum"] = ctx.npoints = read_from_file(ctx.f)
        ctx.params["Capacity"] = ctx.nspectra = int(
            read_from_file(ctx.f, dtype=np.uint64)
        )
        ctx.params["Count"] = ctx.ncollected = int(
            read_from_file(ctx.f, dtype=np.uint64)
        )
        ctx.params["AccumulationCount"] = read_from_file(ctx.f)
        ctx.params["YlistLength"] = read_from_file(ctx.f)
        read_from_file(ctx.f)  # XlistLength — same as PointsPerSpectrum, skip
        ctx.params["DataOriginCount"] = read_from_file(ctx.f)
        ctx.params["ApplicationName"] = read_from_file(ctx.f, "|S24").decode()
        version = read_from_file(ctx.f, np.uint16, count=4)
        ctx.params["ApplicationVersion"] = (
            ".".join([str(x) for x in version[0:-1]])
            + " build "
            + str(version[-1])
        )
        scan_type_raw = int(read_from_file(ctx.f))
        ctx.scan_type_raw = scan_type_raw
        ctx.params["ScanType"] = const.SCAN_TYPES.get(
            scan_type_raw, str(scan_type_raw)
        )
        mtype_raw = int(read_from_file(ctx.f))
        ctx.measurement_type_raw = mtype_raw
        ctx.params["MeasurementType"] = const.MEASUREMENT_TYPES.get(
            mtype_raw, str(mtype_raw)
        )
        ctx.params["StartTime"] = convert_time(
            0.1 * read_from_file(ctx.f, dtype=np.uint64)
        )
        ctx.params["EndTime"] = convert_time(
            0.1 * read_from_file(ctx.f, dtype=np.uint64)
        )
        ctx.params["SpectralUnits"] = const.DATA_UNITS[read_from_file(ctx.f)]
        laser_wavenumber = read_from_file(ctx.f, "<f")
        ctx.params["LaserWaveLength"] = (
            np.round(10e6 / laser_wavenumber, 2)
            if laser_wavenumber
            else "Unspecified"
        )
        ctx.f.seek(240)
        ctx.params["Title"] = read_from_file(ctx.f, "|S160").decode()
