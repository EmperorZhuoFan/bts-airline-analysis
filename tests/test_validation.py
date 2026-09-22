import pandas as pd
import pytest

from airline.data.validation import (
    validate_required_columns,
    validate_data_types,
    validate_duplicates,
    validate_domain_rules,
)


def test_required_columns_pass() -> None:
    """Verify that validation passes when all required columns exist."""

    df = pd.DataFrame({
        "YEAR": [2020],
        "MONTH": [1],
    })

    validate_required_columns(df, ["YEAR", "MONTH"])


def test_required_columns_fail() -> None:
    """Verify that missing required columns raise ValueError."""

    df = pd.DataFrame({
        "YEAR": [2020],
    })

    with pytest.raises(ValueError):
        validate_required_columns(df, ["YEAR", "MONTH"])


def test_data_types_pass() -> None:
    """Verify that correct data types pass validation."""

    df = pd.DataFrame({
        "YEAR": pd.Series([2020], dtype="int64"),
    })

    validate_data_types(df, {"YEAR": "int64"})


def test_duplicates_are_detected() -> None:
    """Verify that duplicate rows are counted correctly."""

    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [2, 2, 3],
    })

    result = validate_duplicates(df)

    assert result == 1


def test_domain_rules_pass() -> None:
    """Verify that valid domain values produce no errors."""

    df = pd.DataFrame({
        "TAXI_OUT": [10],
        "TAXI_IN": [5],
        "AIR_TIME": [100],
        "DISTANCE": [500],
        "ACTUAL_ELAPSED_TIME": [120],
        "MONTH": [6],
        "QUARTER": [2],
        "DAY_OF_WEEK": [3],
        "CANCELLED": [0],
        "DIVERTED": [0],
    })

    result = validate_domain_rules(df)

    assert result == []