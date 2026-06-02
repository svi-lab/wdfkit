"""Tests for wdfkit.catalog."""

from pathlib import Path

import pandas as pd
import pytest

from wdfkit import WDFReader, catalog
from wdfkit.catalog import Catalog

TEST_DATA = Path(__file__).resolve().parent / "test_data"

EXPECTED_COLUMNS = [
    "filename",
    "scan_type",
    "measurement_type",
    "nspectra",
    "laser_wavelength",
    "laser_power",
    "exposure_time",
    "xlist_units",
    "comment",
    "start_time",
    "end_time",
]


def test_catalog_returns_catalog_instance():
    cat = catalog(TEST_DATA)
    assert isinstance(cat, Catalog)


def test_catalog_row_count():
    cat = catalog(TEST_DATA)
    assert len(cat) == len(list(TEST_DATA.glob("*.wdf")))


def test_catalog_has_all_columns():
    cat = catalog(TEST_DATA)
    for col in EXPECTED_COLUMNS:
        assert col in cat.df.columns, f"missing column: {col!r}"


def test_catalog_df_is_dataframe():
    cat = catalog(TEST_DATA)
    assert isinstance(cat.df, pd.DataFrame)


def test_catalog_filenames_are_wdf():
    cat = catalog(TEST_DATA)
    for name in cat.df["filename"]:
        assert name.endswith(".wdf")


def test_catalog_nspectra_positive():
    cat = catalog(TEST_DATA)
    assert (cat.df["nspectra"] > 0).all()


def test_summary_returns_dataframe():
    cat = catalog(TEST_DATA)
    s = cat.summary()
    assert isinstance(s, pd.DataFrame)
    assert "scan_type" in s.columns
    assert "count" in s.columns


def test_summary_counts_sum_to_total():
    cat = catalog(TEST_DATA)
    assert cat.summary()["count"].sum() == len(cat)


def test_to_csv(tmp_path):
    cat = catalog(TEST_DATA)
    out = tmp_path / "catalog.csv"
    cat.to_csv(out)
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == len(cat)
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


def test_load_returns_wdfreader():
    cat = catalog(TEST_DATA)
    r = cat.load(1)
    assert isinstance(r, WDFReader)


def test_load_data_accessible():
    cat = catalog(TEST_DATA)
    r = cat.load(1)
    assert r.data is not None
    assert r.data.ndim == 2


@pytest.mark.parametrize("idx", range(1, 15))
def test_load_all_fixtures(idx):
    cat = catalog(TEST_DATA)
    r = cat.load(idx)
    assert r.data is not None
