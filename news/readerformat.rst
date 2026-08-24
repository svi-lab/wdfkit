**Added:**

* Add a ``dtype`` parameter to ``read()`` and ``WDFReader`` for choosing
  float32 or float64 spectral data.

**Fixed:**

* Fix inconsistent spectral dtype between eager (float64) and chunked
  (float32) reads; both paths now honor the same default.
* Fix the pre-read memory estimate to account for the actual in-memory
  dtype instead of assuming float32.
