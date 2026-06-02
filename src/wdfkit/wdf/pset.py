# -*- coding: utf-8 -*-
"""Renishaw PSET (property-set) binary decoder.

A PSET is a tagged-record blob used inside WXDA, WXDM, WXIS, WXCS, and ZLDC
blocks.  Each record has a 4-byte header:

    [1 byte type tag] [1 byte reserved=0x00] [1 byte key_id] [1 byte flag]

followed by a typed value payload.  Key-definition records (tag ``'k'``) bind
a key_id to a human-readable label; they live in the same namespace as values,
so parsing is two-pass: collect all records first, then resolve labels by
key_id.

Nested PSETs (tag ``'p'``) have their own local key namespace.

Type tags
---------
``?`` bool(1)    ``c`` byte(1)     ``s`` int16(2)
``i`` int32(4)   ``w`` uint32(4)   ``l`` int64(8)
``r`` float32(4) ``q`` float64(8)  ``t`` filetime(8, int64)
``u`` string : [uint32 length][UTF-8 bytes]
``k`` key-def: [uint32 length][label bytes]  — binds keys[key_id] = label
``p`` sub-pset: [uint32 length][child body]
``b`` binary : [uint32 length][bytes]
``n`` null   : no payload
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# Fixed-width tag formats
# ---------------------------------------------------------------------------

_FIXED: dict[int, tuple[str, int]] = {
    ord("?"): ("<?", 1),
    ord("c"): ("<B", 1),
    ord("s"): ("<h", 2),
    ord("i"): ("<i", 4),
    ord("w"): ("<I", 4),
    ord("r"): ("<f", 4),
    ord("l"): ("<q", 8),
    ord("q"): ("<d", 8),
    ord("t"): ("<q", 8),
}
_VARLEN: frozenset[int] = frozenset(ord(c) for c in ("u", "k", "p", "b"))


# ---------------------------------------------------------------------------
# PSet dataclass
# ---------------------------------------------------------------------------


@dataclass
class PSet:
    """Parsed representation of a Renishaw PSET binary blob.

    Attributes
    ----------
    keys:
        Mapping of key_id → human-readable label, as defined by ``'k'``
        records in this PSET.
    values:
        All scalar/string/blob values as ``(key_id, type_tag, value)`` tuples,
        in parse order.  ``type_tag`` is a single character string matching the
        PSET type tags above.
    children:
        Nested PSETs as ``(key_id, PSet)`` tuples.
    """

    keys: dict[int, str] = field(default_factory=dict)
    values: list[tuple[int, str, Any]] = field(default_factory=list)
    children: list[tuple[int, "PSet"]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_by_label(self, label: str) -> Optional[Any]:
        """Return the first value whose key label exactly matches *label*.

        Searches this PSet and all nested children recursively.
        Returns ``None`` if not found.
        """
        wanted = {kid for kid, lbl in self.keys.items() if lbl == label}
        for kid, tag, val in self.values:
            if kid in wanted and tag != "p":
                return val
        for _, child in self.children:
            result = child.get_by_label(label)
            if result is not None:
                return result
        return None

    def get_path(self, path: str) -> Optional[Any]:
        """Return a value by slash-separated label path.

        Example: ``pset.get_path("Motor Positions/XYZ Stage X Motor")``
        descends into the child whose label is ``"Motor Positions"`` and
        returns the value labelled ``"XYZ Stage X Motor"`` within it.

        Single-segment paths are equivalent to :meth:`get_by_label`.
        """
        parts = path.split("/", 1)
        if len(parts) == 1:
            return self.get_by_label(parts[0])
        head, tail = parts
        for kid, child in self.children:
            label = self.keys.get(kid, "")
            if label == head:
                return child.get_path(tail)
        return None

    def walk(self, _prefix: str = "") -> Iterator[tuple[str, str, str, Any]]:
        """Yield ``(path, label, type_tag, value)`` for every non-child entry.

        Descends into nested PSETs; child entries themselves are not yielded
        but their contents are included with a ``"parent/child"`` path prefix.
        """
        for kid, tag, val in self.values:
            if tag == "p":
                continue
            label = self.keys.get(kid, f"key_{kid}")
            path = f"{_prefix}{label}" if _prefix else label
            yield path, label, tag, val
        for kid, child in self.children:
            label = self.keys.get(kid, f"key_{kid}")
            prefix = f"{_prefix}{label}/" if _prefix else f"{label}/"
            yield from child.walk(prefix)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse(body: bytes, _inherited_keys: dict[int, str] | None = None) -> PSet:
    """Parse a PSET body (bytes *after* the ``PSET`` magic + 4-byte size).

    Returns a :class:`PSet` instance with all records decoded.  Unknown type
    tags cause the parser to stop advancing at that position; remaining bytes
    are silently ignored.

    When *_inherited_keys* is provided (used internally for nested PSETs that
    carry no own key definitions), those keys are used as a fallback for label
    resolution — matching Renishaw's convention in blocks like WXDM where
    child PSETs share the top-level key table.
    """
    out = PSet()
    p = 0
    while p + 4 <= len(body):
        tag = body[p]
        if tag not in _FIXED and tag not in _VARLEN and tag != ord("n"):
            break
        key_id = body[p + 2]
        p += 4

        if tag == ord("n"):
            out.values.append((key_id, "n", None))
            continue

        if tag in _FIXED:
            fmt, sz = _FIXED[tag]
            if p + sz > len(body):
                break
            val = struct.unpack(fmt, body[p : p + sz])[0]
            p += sz
            out.values.append((key_id, chr(tag), val))
            continue

        # Variable-length: 4-byte little-endian length prefix.
        if p + 4 > len(body):
            break
        length = struct.unpack("<I", body[p : p + 4])[0]
        p += 4
        if p + length > len(body):
            break
        payload = body[p : p + length]
        p += length

        if tag == ord("u"):
            out.values.append(
                (key_id, "u", payload.decode("utf-8", errors="replace"))
            )
        elif tag == ord("k"):
            out.keys[key_id] = payload.decode("utf-8", errors="replace")
        elif tag == ord("p"):
            child = parse(payload)
            if not child.keys and _inherited_keys:
                child.keys = dict(_inherited_keys)
            out.children.append((key_id, child))
            out.values.append((key_id, "p", child))
        elif tag == ord("b"):
            out.values.append((key_id, "b", payload))

    # Propagate own keys into any keyless children (k-records may appear in
    # the parent body only; children inherit them, recursively).
    effective_keys = out.keys or (_inherited_keys or {})
    if effective_keys and not out.keys:
        out.keys = dict(effective_keys)
    if effective_keys:
        _propagate_keys(out, effective_keys)

    return out


def _propagate_keys(pset: "PSet", keys: dict[int, str]) -> None:
    """Recursively copy *keys* into every keyless descendant of *pset*."""
    for _, child in pset.children:
        if not child.keys:
            child.keys = dict(keys)
        _propagate_keys(child, child.keys if child.keys else keys)


def parse_pset_block(block_body: bytes) -> PSet:
    """Parse a full WDF block body that starts with the ``PSET`` magic.

    Handles blocks that begin with ``b"PSET"`` + 4-byte size, as well as raw
    PSET bodies that start directly with record data.
    """
    if len(block_body) >= 8 and block_body[:4] == b"PSET":
        size = struct.unpack("<I", block_body[4:8])[0]
        return parse(block_body[8 : 8 + size])
    idx = block_body.find(b"PSET")
    if idx >= 0 and idx + 8 <= len(block_body):
        size = struct.unpack("<I", block_body[idx + 4 : idx + 8])[0]
        return parse(block_body[idx + 8 : idx + 8 + size])
    return parse(block_body)


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------


def decode_pset(data: bytes) -> dict[str, Any]:
    """Decode a PSET blob and return a flat ``{label: value}`` dict.

    This is the legacy API used by :mod:`wdfkit.wdf.blocks.wxdm` and
    :mod:`wdfkit.wdf.blocks.wxis`.  New code should use
    :func:`parse_pset_block` and the :class:`PSet` helpers instead.
    """
    pset = parse_pset_block(data)
    result: dict[str, Any] = {}
    for _, label, tag, val in pset.walk():
        if tag not in ("p", "b"):
            result.setdefault(label, val)
    return result


def find_in_pset(data: bytes, key_name: str) -> Optional[Any]:
    """Return the value for *key_name* in a PSET blob, or ``None``."""
    return decode_pset(data).get(key_name)
