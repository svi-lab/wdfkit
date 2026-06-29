**Changed:**

* Cleaned ``data.attrs``: removed internal parser fields, renamed ``Count`` → ``Nspectra``, ``ScanShape`` → ``Shape``, ``NbSteps`` → ``NSteps``; replaced raw ``XlistDataUnits`` with ``SpectralUnits`` (X-axis units).
* Formatted ``StartTime`` / ``EndTime`` attrs as ``YYYY-MM-DD HH:MM:SS`` strings (no timezone, no microseconds).
* Applied same time formatting to catalog ``start_time`` / ``end_time`` columns.

**Fixed:**

* Removed duplicate ``XlistLength`` key (identical to ``PointsPerSpectrum``) written by the WDF1 block parser.

**Added:**

* Added ``Comment`` to ``data.attrs``.
* Added shared ``format_datetime()`` helper in ``wdfkit._shared.time_utils``.
