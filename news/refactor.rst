**Added:**

* Add ``data_type`` attribute (``"single"``, ``"sequence"``, ``"grid"``) to every produced ``xarray.DataArray``.
* Add ``row_axis`` and ``column_axis`` attributes to grid DataArrays recording which physical stage axis each dimension represents.
* Add ``time`` coordinate (seconds elapsed) to sequence DataArrays when an ORGN Time entry is present.
* Add ``time`` coordinate with shape ``(row, column)`` to grid DataArrays when an ORGN Time entry is present.

**Changed:**

* Standardize spectral dimension name to ``"spectral"`` for all scan types; physical axis type is preserved in the coordinate's ``long_name`` attribute.
* Rename spatial dimensions of grid (raster) DataArrays from ``("y", "x", …)`` to ``("row", "column", "spectral")``; ``row`` holds physical y positions (µm), ``column`` holds physical x positions (µm).
* Standardize sequence DataArrays (``points``, ``line_xy``, ``series``) to ``("point", "spectral")`` dimensions with spatial ORGN entries (``SpatialX/Y/Z``) as named coordinates.
* Rework ``series`` handler to use a 0-based integer ``point`` index dimension instead of using the primary ORGN entry type as the dimension name.
* Decode ``WdfFlag`` header field as a list of active flag names via bitmask rather than a single scalar dict lookup.
* Replace absolute file seeks in ``wdf1`` block parser with block-relative offsets.
* Replace bare ``assert`` in ``orgn`` block parser with ``WDFFormatError``.
* Eliminate legacy ``_helpers/constants.py`` dict lookup tables; all block parsers now use ``_enum_name()`` with ``IntEnum`` classes from ``types.py``.
* Consolidate spectral axis resolution to a single ``resolve_spectral_axis()`` call in ``io.py``.

**Removed:**

* Remove ``get_spectral_dim_name()`` and associated lookup dicts from ``types.py`` (superseded by ``resolve_spectral_axis``).
* Remove dead helper ``ensure_in_memory`` from ``_helpers/utils.py``.
* Remove dead shared modules ``_shared/_spectral.py``, ``_shared/normalize.py``, and ``_shared/clean_data.py``.

**Fixed:**

* Fix file-cursor corruption in ``orgn`` parser: Arbitrary-type (type 0) ORGN entries now consume their ``nspectra × 8``-byte payload, preventing misaligned reads of all subsequent ORGN entries.
* Fix ``raster_columnmajor`` module docstring incorrectly naming the flag "StreamLine" (a scan type) instead of "ColumnMajor".
* Fix ``XLSTInfo.dim_name`` field comment to reflect that it stores the physical spectral axis name, not the xarray dimension name.
