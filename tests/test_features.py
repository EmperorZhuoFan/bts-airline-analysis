import pandas as pd

from airline.features.engineering import (
    create_time_features,
    create_delay_features,
    create_route_features,
    create_operational_features,
)


def test_create_time_features() -> None:
    """Verify that departure hour and time-of-day features are created."""

    df = pd.DataFrame({
        "CRS_DEP_TIME": [730],
    })

    result = create_time_features(df)

    assert result["DEP_HOUR"].iloc[0] == 7
    assert result["TIME_OF_DAY"].iloc[0] == "Morning"


def test_create_delay_features() -> None:
    """Verify that delay status and severity features are created."""

    df = pd.DataFrame({
        "DEP_DELAY": [45],
    })

    result = create_delay_features(df)

    assert result["IS_DEP_DELAYED"].iloc[0] == 1
    assert result["DEP_DELAY_STATUS"].iloc[0] == "Delayed"
    assert result["DEP_DELAY_SEVERITY"].iloc[0] == "Moderate Delay"


def test_create_route_features() -> None:
    """Verify that the route feature combines origin and destination."""

    df = pd.DataFrame({
        "ORIGIN": ["JFK"],
        "DEST": ["LAX"],
    })

    result = create_route_features(df)

    assert result["ROUTE"].iloc[0] == "JFK --> LAX"


def test_create_operational_features() -> None:
    """Verify that taxi time and distance category features are created."""

    df = pd.DataFrame({
        "TAXI_OUT": [20],
        "TAXI_IN": [10],
        "DISTANCE": [800],
    })

    result = create_operational_features(df)

    assert result["TOTAL_TAXI_TIME"].iloc[0] == 30
    assert result["DISTANCE_CATEGORY"].iloc[0] == "Medium"