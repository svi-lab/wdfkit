=============
Release notes
=============

.. current developments

0.1.1
=====

**Added:**

* Added ``SpectraSmoother`` support for 1-D single spectra.
* Added "PCA" CosmicRayRemover for 3D spectra
* Add ``CleanData`` class to detect oversaturated spectra (10+ consecutive zero channels) and automatically remove them from 2D/3D arrays; integrated as the first step in ``CosmicRayRemover`` and ``SpectraCleaner``.
* Add zero-saturation detection to ``CosmicRayRemover`` via ``_zero_saturation_mask``, flagging ADC-clipped channels before positive-spike removal.
* Add reading of InitialCoordinates for 2D files.
* Added ``CosmicRayRemover`` support for 1-D single spectra.
* Add initial coordinate for 1D WDF files.

**Changed:**

* Reorganize package internals into dedicated sub-packages; the public API is unchanged.
* Restrict top-level exports to ``WDFReader``, ``CosmicRayRemover``, ``SpectraCleaner``, and ``normalize``.
* CHANGELOG.rst file updated.
* Changed parameters set for CosmicRayRemover

**Deprecated:**

* ``wdfkit.preprocessing`` module; import ``normalize`` directly from ``wdfkit`` instead.

**Fixed:**

* Fixed a bug in the spectra smoother where the spectral dimension was not being preserved.
* Fix ``CosmicRayRemover`` collection-engine repair to interpolate from the original spectrum's clean channels instead of the PCA reference, eliminating residual negative spikes.
* Fix ``CosmicRayRemover`` collection engine to run a second detection pass on a reference rebuilt from clean data, improving sensitivity on heterogeneous maps.
* Lower default ``spike_threshold`` in ``CosmicRayRemover`` from ``5.0`` to ``3.5`` to improve cosmic-ray detection on typical spectra.


0.1.0
=====

**Added:**

* Add optional ``chunks`` on ``WDFReader`` for lazy, Dask-backed map reads with Y-row-aligned targets and a RAM guard before the ``DATA`` block.
* Add ``wdfkit.read`` and header-only ``wdfkit.classify`` on the public API.
* Add a ``wdf/`` layout with per-scan-kind handlers, ``ParsedWDF``, and typed enums/constants for parsing and dispatch.
* Add ``CosmicRayRemover`` support for 2-D inputs and an iterative ``max_passes`` (default ``3``).
* Expose ``ExposureTime`` and ``LaserPower`` from ``WXDM`` / ``WXIS`` as ``DataArray`` attributes where applicable.
* Ship ``py.typed`` and extend type annotations on the read/assembly surface.

**Changed:**

* Make ``normalize``, ``CosmicRayRemover``, and ``SpectraCleaner`` Dask-aware with lazy paths or warnings when full materialisation is needed.
* Unify ``WDFReader``, ``wdfkit.read``, and ``read_wdf_file`` on one handler-based code path; fold the former top-level ``internal/``, ``spectral/`` package tree, and related modules into ``wdf/`` (public ``from wdfkit.spectral import SpectralAxisSpec`` unchanged).
* Return map dimensions as ``x`` / ``y`` (was ``X`` / ``Y``), single scans as 1-D spectral arrays, and sort the spectral coordinate ascending for all kinds.
* Tidy ``attrs`` (CamelCase only), use linear interpolation and slight mask dilation for 1-D cosmic-ray repair, and refresh docs and branding.

**Fixed:**

* Fix chunked-map reads that hit Dask ``nd`` fancy-index failures during ``sortby`` assembly.
* Fix the installed ``wdfkit`` CLI entry point (``ModuleNotFoundError`` under the previous target).
* Fix a file-pointer bug in ``origin.py`` for certain dtypes, preserve exception chaining on missing files in ``io.py``, and confine truncated-image PIL settings to parsing so import has no global side effect.
* Gate noisy YLST debug output behind verbose mode.
* Raise ``ValueError`` for unknown ``normalize`` methods instead of returning silently; replace coordinate ``assert`` checks with explicit errors.

**Removed:**

* Remove the old ``wdf/assemble.py``-centric layout, duplicate package roots absorbed into ``wdf/``, and a few obsolete test helpers.


0.0.1
=====

**Changed:**

* README.rst file updated.
* Logo added to README.rst file.
