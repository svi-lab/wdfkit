=============
Release notes
=============

.. current developments

0.1.0
=====

**Added:**

* ``catalog()``: fast, header-only directory scan that builds a metadata
  table over a collection of ``.wdf`` files without loading spectra.
* ``wdfkit.read`` and header-only ``wdfkit.classify`` on the public API.
* ``wdf/`` package layout with per-scan-kind handlers (``single``,
  ``series``, ``points``, ``line_xy``, ``raster_rowmajor``,
  ``raster_columnmajor``, ``raster_snake``, ``linefocus``, ``volume``), a
  typed ``ParsedWDF``, and enum-backed block parsing.
* ``data_type`` (``"single"`` / ``"sequence"`` / ``"grid"``) and ``kind``
  attrs on every produced ``DataArray``; ``row_axis`` / ``column_axis``
  attrs on grid (map) arrays recording which physical stage axis each
  dimension represents.
* ``time`` coordinate (seconds elapsed) on sequence and grid DataArrays
  when an ORGN Time entry is present.
* Optional ``chunks`` on ``WDFReader`` for lazy, Dask-backed map reads.
* ``comment`` and per-scan acquisition settings (``exposure_time``,
  ``laser_power``) added to ``DataArray.attrs`` where available.
* Descriptive error, listing known flag bits, for unsupported/unknown
  ``MapAreaType`` flags in the WMAP block.
* Documentation rewritten to describe the current ``DataArray`` layout
  (``data_type``/``kind``/dimension conventions) and the ``catalog``
  function.

**Changed:**

* Standardized the spectral dimension name to ``"spectral"`` across all
  scan kinds; grid (map) DataArrays use ``("row", "column", "spectral")``,
  sequence/point DataArrays use ``("point", "spectral")``.
* Cleaned ``DataArray.attrs`` down to scientifically relevant, snake_case
  keys (e.g. ``n_spectra``, ``n_points``, ``spectral_units``,
  ``start_time``, ``end_time``); internal parser-only fields are no longer
  exposed.
* ``start_time`` / ``end_time`` attrs, and the matching ``catalog()``
  columns, are formatted as ``YYYY-MM-DD HH:MM:SS`` strings.

**Fixed:**

* Fixed file-cursor corruption in the ORGN parser for arbitrary-type
  entries, which could misalign all subsequent ORGN reads.
* Removed a duplicate ``XlistLength`` key written by the WDF1 block parser.


0.0.1
=====

**Changed:**

* README.rst file updated.
* Logo added to README.rst file.
