"""Shared pytest fixtures and helpers."""

from pathlib import Path

import pytest

TEST_DATA = Path(__file__).resolve().parent / "test_data"


@pytest.fixture
def data_dir():
    return TEST_DATA
