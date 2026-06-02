# -*- coding: utf-8 -*-
"""Parse ``XLST`` spectral / X-axis coordinate block."""

from __future__ import annotations

from .._helpers import constants as const
from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from .._helpers.spectral import resolve_spectral_axis


def parse_xlst(ctx: ParseContext) -> None:
    name = "XLST"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        xldt = read_from_file(ctx.f)
        ctx.params["XlistDataType"] = const.DATA_TYPES.get(
            xldt, f"{xldt}_unknown"
        )
        xldu = read_from_file(ctx.f)
        ctx.params["XlistDataUnits"] = const.DATA_UNITS.get(
            xldu, f"{xldu}_unknown"
        )
        x_values = read_from_file(ctx.f, "<f", count=ctx.npoints)
        spectral_spec = resolve_spectral_axis(
            ctx.params["XlistDataUnits"], ctx.spectral_dim
        )
        ctx.spectral_dim_name = spectral_spec.dim_name
        ctx.coord_dict = {
            **ctx.coord_dict,
            ctx.spectral_dim_name: (
                ctx.spectral_dim_name,
                x_values,
                {
                    "long_name": ctx.params["XlistDataUnits"],
                    "units": spectral_spec.units,
                },
            ),
        }

    if ctx.verbose and ctx.spectral_dim_name:
        xv = ctx.coord_dict[ctx.spectral_dim_name][1]
        print(f"{'The shape of the x_values is':-<40s} : \t{xv.shape} ")
        print(
            f"*These are the \"{ctx.params['XlistDataType']}"
            f"\" recordings in \"{ctx.params['XlistDataUnits']}\" units"
        )
