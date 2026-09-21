import pandas as pd 


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features that represent when each flight operates."""

    df = df.copy()

    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100)

    df["TIME_OF_DAY"] = pd.cut(df["DEP_HOUR"], bins=[-1, 5, 11, 17, 23], 
                                                labels=["Night", "Morning", "Afternoon", "Evening"])

    return df



def create_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features that represent the severity and behavior of flight delays."""

    df = df.copy()

    df["IS_DEP_DELAYED"] = (df["DEP_DELAY"] > 0)

    df["IS_DEP_DELAYED"] = df["IS_DEP_DELAYED"].map({False: 0, True: 1})

    df["DEP_DELAY_STATUS"] = df["IS_DEP_DELAYED"].map({0: "Not Delayed", 1: "Delayed"})

    df["DEP_DELAY_SEVERITY"] = pd.cut(df["DEP_DELAY"], bins=[-float("inf") , 0 , 30, 120, float("inf")],
                                                        labels=["Early/On-Time", "Minor Delay", "Moderate Delay", "Severe Delay"])

    return df



def create_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features that represent the relationship between flight origin and destination."""

    df = df.copy()

    df["ROUTE"] = df["ORIGIN"] + " --> " + df["DEST"]

    return df



def create_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features that represent the operational characteristics of each flight."""

    df = df.copy()

    df["TOTAL_TAXI_TIME"] = df["TAXI_OUT"] + df["TAXI_IN"]

    df["DISTANCE_CATEGORY"] = pd.cut(df["DISTANCE"], bins=[0, 500, 1000, float("inf")], 
                                                        labels=["Short", "Medium", "Long"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to the prepared DataFrame."""

    df = df.copy()

    df = create_time_features(df)
    df = create_delay_features(df)
    df = create_route_features(df)
    df = create_operational_features(df)

    return df
