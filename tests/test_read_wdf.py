"""Parity tests for :class:`wdfkit.WDFReader` against golden spectral
arrays.

The golden SHA256 hashes pin the raw spectral values (byte-for-byte).
Shape changes from the refactor do not affect these hashes because
C-contiguous float32 bytes are the same regardless of whether the array
is 1-D or n-D.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from wdfkit import WDFReader, read

TEST_DATA = Path(__file__).resolve().parent / "test_data"

# SHA256 of contiguous spectral ``values`` bytes — pins full arrays.
_VALUES_SHA256 = {
    "test.wdf": (
        "1121d8054198265a1c476ebf662816edb7714a8c163f774b0e3457ba3e46ec65"
    ),
    "test_map.wdf": (
        "3b389ea645ecc6f712147b264d607800598bc76354cff6a28b558c08b06c25a9"
    ),
}


def _sha256_values(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_read_wdf_single_scan_matches_golden():
    path = TEST_DATA / "test.wdf"
    da, img = WDFReader(path)

    # New shape: single → 1-D ("spectral",)
    assert dict(da.sizes) == {"spectral": 9341}
    assert da.attrs["measurement_type"] == "Single"
    assert da.attrs["n_points"] == 9341
    assert da.attrs["n_spectra"] == 1
    assert da.attrs["shape"] == (1, 1)
    assert da.attrs["file_size"] == "357.5kB"
    assert da.attrs["kind"] == "single"
    assert da.attrs["data_type"] == "single"
    assert np.isclose(da.attrs["laser_wavelength_nm"], 354.74)
    assert img is None

    # Spectral values (sorted ascending by nm)
    np.testing.assert_array_equal(da["spectral"].shape, (9341,))
    np.testing.assert_allclose(
        da.values[:3], [959.03619385, 961.67810059, 973.14605713]
    )
    np.testing.assert_allclose(
        da.values[-3:], [17035.36328125, 17592.48242188, 17531.3671875]
    )

    assert _sha256_values(da.values) == _VALUES_SHA256["test.wdf"]


def test_read_wdf_map_matches_golden():
    path = TEST_DATA / "test_map.wdf"
    da, img = WDFReader(path)

    # New shape: raster_rowmajor → ("row", "column", "spectral")
    assert dict(da.sizes) == {"row": 17, "column": 25, "spectral": 9341}
    assert da.attrs["measurement_type"] == "Map"
    assert da.attrs["n_spectra"] == 425
    assert da.attrs["n_points"] == 9341
    assert da.attrs["shape"] == (17, 25)
    assert da.attrs["file_size"] == "16.2MB"
    assert da.attrs["kind"] == "raster_rowmajor"
    assert img is not None
    assert getattr(img, "mode") == "RGB"

    corner = da.values[0, 0, :3]
    np.testing.assert_allclose(
        corner, [668.75933838, 673.73608398, 695.56280518]
    )
    np.testing.assert_allclose(
        da.values[0, 0, -3:],
        [12142.55175781, 12009.16113281, 12445.69824219],
    )

    assert _sha256_values(da.values) == _VALUES_SHA256["test_map.wdf"]


def test_read_wdf_missing_file_raises():
    with pytest.raises(IOError, match="does not exist"):
        WDFReader(TEST_DATA / "nonexistent_file.wdf")


def test_wdf_reader_idempotent_same_file():
    """Two reads of the same path yield identical spectral cubes."""
    path = TEST_DATA / "test.wdf"
    da_a, _ = WDFReader(path)
    da_b, _ = WDFReader(path)
    np.testing.assert_array_equal(da_a.values, da_b.values)


def test_data_type_attrs():
    """Every DataArray has a data_type attr describing its format."""
    path_single = TEST_DATA / "test.wdf"
    path_map = TEST_DATA / "test_map.wdf"
    da_s, _ = WDFReader(path_single)
    da_m, _ = WDFReader(path_map)
    assert da_s.attrs["data_type"] == "single"
    assert da_m.attrs["data_type"] == "grid"


def test_module_level_read_returns_dataarray():
    path = TEST_DATA / "test.wdf"
    da = read(path)
    assert dict(da.sizes) == {"spectral": 9341}


def test_exposure_time_and_laser_power_single_scan():
    """ExposureTime and LaserPower are read from WXDM/WXIS blocks."""
    path = TEST_DATA / "test.wdf"
    da, _ = WDFReader(path)

    assert "exposure_time" in da.attrs
    assert np.isclose(da.attrs["exposure_time"], 10.0)

    assert "laser_power" in da.attrs
    assert np.isclose(da.attrs["laser_power"], 10.0)


def test_exposure_time_and_laser_power_map():
    """exposure_time and laser_power are read correctly for 2-D map data."""
    path = TEST_DATA / "test_map.wdf"
    da, _ = WDFReader(path)

    assert "exposure_time" in da.attrs
    assert np.isclose(da.attrs["exposure_time"], 10.0)

    assert "laser_power" in da.attrs
    assert np.isclose(da.attrs["laser_power"], 5.0)
