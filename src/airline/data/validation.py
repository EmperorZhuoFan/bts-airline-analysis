import pandas as pd 


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that all required columns are present in the DataFrame."""

    missing_columns = []

    for column in required_columns:

        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Some columns are missing: {', '.join(missing_columns)}")
    
    print("All required columns are present.")


def validate_data_types(df: pd.DataFrame, expected_types: dict) -> None:
    """Validate that the expected data types are correct in the DataFrame."""

    for column , expected_dtype in expected_types.items(): 
        actual_type = df[column].dtype

        if str(actual_type) != expected_dtype:
            raise ValueError(f"Incorrect data type for {column}: expected {expected_dtype} got {actual_type}")

    print("All required data types are correct.")
   
  
def validate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Validate missing values in the DataFrame."""
                                                                                                
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        missing_values_percentage = missing_values.div(len(df)).mul(100).round(2).astype(str) + "%"

        missing_report = pd.DataFrame({
            "Columns": missing_values.index,
            "Missing": missing_values.values, 
            "Missing %": missing_values_percentage.values 
        })

    else:
        missing_report = pd.DataFrame(columns=["Columns", "Missing", "Missing %"])

    return missing_report


def validate_duplicates(df: pd.DataFrame) -> int:
    """Validate the DataFrame for duplicate rows and return the duplicate count."""

    count_duplicates = df.duplicated().sum()

    print(f"Duplicated rows count: {count_duplicates}")

    return count_duplicates


def validate_domain_rules(df: pd.DataFrame) -> list[str]:
    """Validate the DataFrame against domain-specific rules and return columns with invalid values."""

    nonnegative_values = df[["TAXI_OUT", "TAXI_IN", "AIR_TIME", "DISTANCE", "ACTUAL_ELAPSED_TIME"]]
    err_values = []

    for column in nonnegative_values:
        if not (df[column] >= 0).all():
            err_values.append(column)
           
    if not (df["MONTH"]).between(1, 12).all():
        err_values.append("MONTH")
        
    if not (df["QUARTER"]).between(1, 4).all():
        err_values.append("QUARTER")

    if not (df["DAY_OF_WEEK"]).between(1, 7).all():
        err_values.append("DAY_OF_WEEK")

    if not (df["CANCELLED"]).isin([0, 1]).all():
        err_values.append("CANCELLED")

    if not (df["DIVERTED"]).isin([0, 1]).all():
        err_values.append("DIVERTED")

    return err_values