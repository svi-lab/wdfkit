# -*- coding: utf-8 -*-
"""Datetime formatting helpers shared across the package."""

from __future__ import annotations

from datetime import datetime


def format_datetime(dt: datetime | None) -> str | None:
    """Strip timezone and sub-second precision from a UTC-aware datetime.

    ``2026-05-05 09:08:12.686456+00:00``  →  ``"2026-05-05 09:08:12"``
    """
    if dt is None:
        return None
    return dt.replace(microsecond=0, tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
