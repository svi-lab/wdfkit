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

    # New shape: single → 1-D (spectral_dim,)
    assert dict(da.sizes) == {"wavelength_nm": 9341}
    assert da.attrs["WdfFlag"] == "WdfXYXY"
    assert da.attrs["MeasurementType"] == "Single"
    assert da.attrs["PointsPerSpectrum"] == 9341
    assert da.attrs["Capacity"] == 1
    assert da.attrs["Count"] == 1
    assert da.attrs["ScanShape"] == (1, 1)
    assert da.attrs["ColCoord"] is None
    assert da.attrs["RowCoord"] is None
    assert da.attrs["Filename"] == "test.wdf"
    assert da.attrs["Folder name"] == str(path.parent)
    assert da.attrs["FileSize"] == "357.5kB"
    assert da.attrs["SpectralUnits"] == "Counts"
    assert np.isclose(da.attrs["LaserWaveLength"], 354.74)
    assert img is None

    # Spectral values (sorted ascending by nm)
    np.testing.assert_array_equal(da["wavelength_nm"].shape, (9341,))
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

    # New shape: raster_rowmajor → (y, x, spectral_dim)
    assert dict(da.sizes) == {"y": 17, "x": 25, "wavelength_nm": 9341}
    assert da.attrs["WdfFlag"] == "16: UnknownFlag (LiveTrack?)"
    assert da.attrs["MeasurementType"] == "Map"
    assert da.attrs["Capacity"] == 425
    assert da.attrs["Count"] == 425
    assert da.attrs["PointsPerSpectrum"] == 9341
    assert da.attrs["ScanShape"] == (17, 25)
    assert da.attrs["ColCoord"] == "x"
    assert da.attrs["RowCoord"] == "y"
    assert da.attrs["Filename"] == "test_map.wdf"
    assert da.attrs["Folder name"] == str(path.parent)
    assert da.attrs["FileSize"] == "16.2MB"
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


def test_spectral_dim_override_restores_legacy_name():
    """Force spectral coordinate dimension name (e.g. ``shifts``)."""
    path = TEST_DATA / "test.wdf"
    da, _ = WDFReader(path, spectral_dim="shifts")
    assert dict(da.sizes) == {"shifts": 9341}
    assert da["shifts"].attrs.get("units") == "nm"


def test_module_level_read_returns_dataarray():
    path = TEST_DATA / "test.wdf"
    da = read(path)
    assert dict(da.sizes) == {"wavelength_nm": 9341}


def test_exposure_time_and_laser_power_single_scan():
    """ExposureTime and LaserPower are read from WXDM/WXIS blocks."""
    path = TEST_DATA / "test.wdf"
    da, _ = WDFReader(path)

    assert "ExposureTime" in da.attrs
    assert np.isclose(da.attrs["ExposureTime"], 10.0)

    assert "LaserPower" in da.attrs
    assert np.isclose(da.attrs["LaserPower"], 10.0)


def test_exposure_time_and_laser_power_map():
    """ExposureTime and LaserPower are read correctly for 2-D map data."""
    path = TEST_DATA / "test_map.wdf"
    da, _ = WDFReader(path)

    assert "ExposureTime" in da.attrs
    assert np.isclose(da.attrs["ExposureTime"], 10.0)

    assert "LaserPower" in da.attrs
    assert np.isclose(da.attrs["LaserPower"], 5.0)
