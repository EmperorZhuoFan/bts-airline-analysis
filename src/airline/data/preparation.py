import pandas as pd

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in a way that respects the meaning of each column."""

    df = df.copy()

    # Missing delay-cause values represent no recorded delay contribution.
    delay_reason_columns = [
        "CARRIER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]

    for column in delay_reason_columns:
        df[column] = df[column].fillna(0)

    # Create a meaningful cancellation reason from the cancellation code.
    df["CANCELLATION_REASON"] = df["CANCELLATION_CODE"].map({
        "A": "Carrier", 
        "B": "Weather", 
        "C": "NAS", 
        "D": "Security"
    }).fillna("Not Cancelled")

    print("=" * 70)
    print("Missing value handling: Completed")
    print("CANCELLATION_REASON created from CANCELLATION_CODE")
    print("=" * 70)

    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime format."""

    df = df.copy()

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"],  format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

    invalid_dates = df["FL_DATE"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(f"Date conversion produced: {invalid_dates} -- invalid values.")

    df["MONTH_NAME"] = df["FL_DATE"].dt.month_name()
    df["DAY_NAME"] = df["FL_DATE"].dt.day_name()
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    df["WEEK_OF_YEAR"] = df["FL_DATE"].dt.isocalendar().week.astype("int8")
    df["IS_WEEKEND"] = df["FL_DATE"].dt.dayofweek.ge(5)

    print("=" * 70)
    print("Date conversion: completed")
    print(f"Date range {df['FL_DATE'].min()} - {df['FL_DATE'].max()}")
    print(f"Invalid dates: {invalid_dates}")
    print("=" * 70)

    return df


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize categorical columns."""

    df = df.copy()

    categorical_columns = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR",
                            "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR"]

    # Remove leading and trailing whitespace from categorical values
    for column in categorical_columns:
        df[column] = df[column].str.strip()

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""

    df = df.copy()

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    print("=" * 70)
    print(f"Duplicated rows found: {duplicate_count}")
    print(f"Removed rows: {duplicate_count}")
    print("=" * 70)

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the DataFrame for analysis and machine learning."""

    df = handle_missing_values(df)
    df = convert_date_columns(df)
    df = clean_categorical_columns(df)
    df = remove_duplicates(df)

    return df