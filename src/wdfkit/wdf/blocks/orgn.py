# -*- coding: utf-8 -*-
"""Parse ``ORGN`` per-spectrum origin coordinates (time, stage axes,
...)."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np

from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from .._helpers.utils import convert_time, pad_if_unfinished
from ..error import WDFFormatError
from ..types import DataType, UnitType, _enum_name

_EPOCH = datetime(year=1601, month=1, day=1, tzinfo=timezone.utc)


def parse_orgn(ctx: ParseContext) -> None:
    name = "ORGN"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks[i].offset + 16)
        nb_origin_sets = read_from_file(ctx.f)
        if nb_origin_sets != ctx.params["DataOriginCount"]:
            raise WDFFormatError(
                "ORGN origin set count",
                ctx.params["DataOriginCount"],
                nb_origin_sets,
            )
        for _set_n in range(nb_origin_sets):
            raw_type = int(read_from_file(ctx.f))
            is_primary = bool(raw_type & 0x80000000)
            data_type_flag = raw_type & 0x7FFFFFFF
            data_type = _enum_name(DataType, data_type_flag)
            ctx.origin_is_primary.append(is_primary)
            ctx.origin_set_dtypes.append(data_type)
            coord_units_flag = read_from_file(ctx.f)
            coord_units = _enum_name(UnitType, coord_units_flag).lower()
            ctx.origin_set_units.append(coord_units)
            label_raw = read_from_file(ctx.f, "<S16") + b"\0"
            ndx = label_raw.index(b"\0")
            label = label_raw[:ndx].decode("utf-8")
            ctx.origin_labels.append(label)

            if data_type_flag == 11:
                microseconds_from_epoch = 0.1 * read_from_file(
                    ctx.f, np.uint64, count=ctx.nspectra
                )
                if ctx.time_coord == "seconds_elapsed":
                    st = ctx.params["StartTime"] - _EPOCH
                    recording_time = (
                        1e-6 * microseconds_from_epoch - st.total_seconds()
                    )
                else:
                    recording_time = convert_time(microseconds_from_epoch)

                recording_time = np.atleast_1d(recording_time)

                if ctx.params["Count"] < ctx.params["Capacity"]:
                    recording_time = pad_if_unfinished(
                        recording_time,
                        count=ctx.params["Count"],
                        capacity=ctx.params["Capacity"],
                        extend=True,
                    )
                ctx.coord_dict = {
                    **ctx.coord_dict,
                    label: (
                        "points",
                        recording_time,
                        {
                            "units": coord_units,
                            "long_name": data_type,
                        },
                    ),
                }
            elif data_type_flag in (16, 17):
                # uint64 origin type (e.g. internal Renishaw flags): advance
                # the file position but do not store — interpretation unknown.
                read_from_file(ctx.f, "<Q", count=ctx.nspectra)
            elif data_type_flag == 0:
                # Arbitrary type: consume the float64 payload to keep the
                # file position correct, but do not store.
                read_from_file(ctx.f, "<d", count=ctx.nspectra)
            else:
                coord_values = np.array(
                    np.round(
                        read_from_file(ctx.f, "<d", count=ctx.nspectra),
                        2,
                    )
                )
                ctx.coord_dict = {
                    **ctx.coord_dict,
                    label: (
                        "points",
                        coord_values,
                        {
                            "units": coord_units,
                            "long_name": data_type,
                        },
                    ),
                }


def print_coord_lengths_if_verbose(ctx: ParseContext) -> None:
    if not ctx.verbose:
        return
    try:
        print(
            [
                f"{c} : {len(ctx.coord_dict[c][1])}"
                for c in ctx.coord_dict.keys()
            ]
        )
    except Exception as exc:
        warnings.warn(
            f"Could not print coordinate lengths: {exc}",
            UserWarning,
            stacklevel=2,
        )
