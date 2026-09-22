from pathlib import Path

import pandas as pd
import pytest

from airline.data.loading import validate_file_path, load_data


def test_validate_existing_csv(tmp_path: Path) -> None:
    """Verify that a valid CSV path passes validation."""

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("A,B\n1,2")

    result = validate_file_path(csv_file)

    assert result == csv_file


def test_validate_missing_file(tmp_path: Path) -> None:
    """Verify that a missing file raises FileNotFoundError."""

    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        validate_file_path(missing_file)


def test_validate_non_csv_file(tmp_path: Path) -> None:
    """Verify that a non-CSV file raises ValueError."""

    text_file = tmp_path / "sample.txt"
    text_file.write_text("test")

    with pytest.raises(ValueError):
        validate_file_path(text_file)


def test_load_valid_csv(tmp_path: Path) -> None:
    """Verify that a valid CSV is loaded as a DataFrame."""

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("A,B\n1,2\n3,4")

    df = load_data(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)