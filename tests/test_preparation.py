import pandas as pd

from airline.data.preparation import (
    handle_missing_values,
    convert_date_columns,
    clean_categorical_columns,
    remove_duplicates,
)


def test_handle_missing_values() -> None:
    """Verify that delay causes are filled and cancellation reason is created."""

    df = pd.DataFrame({
        "CARRIER_DELAY": [None],
        "NAS_DELAY": [None],
        "SECURITY_DELAY": [None],
        "LATE_AIRCRAFT_DELAY": [None],
        "CANCELLATION_CODE": ["A"],
    })

    result = handle_missing_values(df)

    assert result["CARRIER_DELAY"].iloc[0] == 0
    assert result["CANCELLATION_REASON"].iloc[0] == "Carrier"


def test_convert_date_columns() -> None:
    """Verify that flight dates are converted and time features are created."""

    df = pd.DataFrame({
        "FL_DATE": ["01/15/2020 12:00:00 AM"],
    })

    result = convert_date_columns(df)

    assert pd.api.types.is_datetime64_any_dtype(result["FL_DATE"])
    assert "MONTH_NAME" in result.columns
    assert "DAY_NAME" in result.columns
    assert "DAY_OF_MONTH" in result.columns
    assert "WEEK_OF_YEAR" in result.columns
    assert "IS_WEEKEND" in result.columns


def test_clean_categorical_columns() -> None:
    """Verify that leading and trailing whitespace is removed."""

    df = pd.DataFrame({
        "OP_UNIQUE_CARRIER": [" AA "],
        "TAIL_NUM": [" N123AA "],
        "ORIGIN": [" JFK "],
        "ORIGIN_CITY_NAME": [" New York, NY "],
        "ORIGIN_STATE_ABR": [" NY "],
        "DEST": [" LAX "],
        "DEST_CITY_NAME": [" Los Angeles, CA "],
        "DEST_STATE_ABR": [" CA "],
    })

    result = clean_categorical_columns(df)

    assert result["OP_UNIQUE_CARRIER"].iloc[0] == "AA"
    assert result["TAIL_NUM"].iloc[0] == "N123AA"
    assert result["ORIGIN"].iloc[0] == "JFK"
    assert result["ORIGIN_CITY_NAME"].iloc[0] == "New York, NY"
    assert result["ORIGIN_STATE_ABR"].iloc[0] == "NY"
    assert result["DEST"].iloc[0] == "LAX"
    assert result["DEST_CITY_NAME"].iloc[0] == "Los Angeles, CA"
    assert result["DEST_STATE_ABR"].iloc[0] == "CA"


def test_remove_duplicates() -> None:
    """Verify that duplicate rows are removed."""

    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [2, 2, 3],
    })

    result = remove_duplicates(df)

    assert len(result) == 2