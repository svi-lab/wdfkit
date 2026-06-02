# -*- coding: utf-8 -*-
"""WDFFormatError — raised when a structural self-check on a .wdf file fails.
"""

from __future__ import annotations


class WDFFormatError(ValueError):
    """Raised when a WDF file fails a structural integrity check.

    Parameters
    ----------
    check:
        Short name of the check that failed, e.g. ``"DATA body size"``.
    expected:
        Value that was expected.
    got:
        Value that was actually found.
    """

    def __init__(self, check: str, expected: object, got: object) -> None:
        super().__init__(f"{check}: expected {expected!r}, got {got!r}")
        self.check = check
        self.expected = expected
        self.got = got
