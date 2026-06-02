**Added:**

* Expose all parsed WDF block data as typed properties on ``WDFReader``: ``orgn``,
  ``xlst``, ``ylst``, ``wmap``, ``comment``, ``acquisition``, ``instrument_status``,
  ``calibration``, ``zeldac``, ``bkxl``, ``whtl_jpeg_bytes``, ``initial_coordinates``,
  ``motor_positions``, ``acquisition_time``, ``file_uuid``.
* Add ``WDFFormatError`` raised on structural file integrity failures.
* Add per-block test suite (87 new tests).

**Changed:**

* Reorganise internal helpers into ``wdf/_helpers/`` subpackage.
* Rename block parsers to match WDF block IDs (``data.py``, ``orgn.py``).
* Upgrade PSET parser to return structured ``PSet`` object with ``get_by_label()``,
  ``get_path()``, and ``walk()`` methods.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
