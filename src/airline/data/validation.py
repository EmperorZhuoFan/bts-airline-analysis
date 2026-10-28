import pandas as pd 
import great_expectations as gx

required_columns = [
    "YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK",
    "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM",
    "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR",
    "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY",
    "DEP_DELAY_NEW", "TAXI_OUT", "TAXI_IN",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY", "ARR_DELAY_NEW",
    "ARR_DEL15", "CANCELLED", "CANCELLATION_CODE",
    "DIVERTED", "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "DISTANCE", "CARRIER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
]

DEFAULT_EXPECTED_DTYPES = {
    "YEAR": "int64",
    "QUARTER": "int64",
    "MONTH": "int64",
    "DAY_OF_WEEK": "int64",
    "FL_DATE": "object",
    "OP_UNIQUE_CARRIER": "object",
    "TAIL_NUM": "object",
    "ORIGIN": "object",
    "ORIGIN_CITY_NAME": "object",
    "ORIGIN_STATE_ABR": "object",
    "DEST": "object",
    "DEST_CITY_NAME": "object",
    "DEST_STATE_ABR": "object",
    "CRS_DEP_TIME": "int64",
    "DEP_TIME": "float64",
    "DEP_DELAY": "float64",
    "DEP_DELAY_NEW": "float64",
    "TAXI_OUT": "float64",
    "TAXI_IN": "float64",
    "CRS_ARR_TIME": "int64",
    "ARR_TIME": "float64",
    "ARR_DELAY": "float64",
    "ARR_DELAY_NEW": "float64",
    "ARR_DEL15": "float64",
    "CANCELLED": "float64",
    "CANCELLATION_CODE": "object",
    "DIVERTED": "float64",
    "ACTUAL_ELAPSED_TIME": "float64",
    "AIR_TIME": "float64",
    "DISTANCE": "float64",
    "CARRIER_DELAY": "float64",
    "NAS_DELAY": "float64",
    "SECURITY_DELAY": "float64",
    "LATE_AIRCRAFT_DELAY": "float64",
}
expected_dtypes = DEFAULT_EXPECTED_DTYPES


def validate_required_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> None:
    """Validate that all required columns exist in the DataFrame."""

    columns_to_validate = required_columns if columns is None else columns
    if not isinstance(columns_to_validate, list) or not all(
        isinstance(column, str) and column for column in columns_to_validate
    ):
        raise ValueError("columns must be a list of non-empty column names.")

    missing_columns = set(columns_to_validate) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="airline_data_source")
    data_asset = data_source.add_dataframe_asset(name="airline_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="airline_batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="airline_required_columns_suite")

    for column in columns_to_validate:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))

    results = batch.validate(expect=suite)

    if not results.success:
        raise ValueError("Great Expectations validation failed for required columns.")


def validate_data_types(
    df: pd.DataFrame,
    expected_dtypes: dict[str, str] | None = None,
) -> None:
    """Validate that the expected data types are correct in the DataFrame."""

    if expected_dtypes is None:
        expected_dtypes = DEFAULT_EXPECTED_DTYPES

    if not isinstance(expected_dtypes, dict) or not expected_dtypes:
        raise ValueError("expected_dtypes must be a non-empty dictionary.")

    if not all(isinstance(column, str) and column for column in expected_dtypes):
        raise ValueError("All column names in expected_dtypes must be non-empty strings.")

    missing_columns = set(expected_dtypes) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for data type validation: {sorted(missing_columns)}")

    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="airline_data_source")
    data_asset = data_source.add_dataframe_asset(name="airline_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="airline_batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="airline_validate_data_types")

    for column, expected_dtype in expected_dtypes.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_=expected_dtype))

    results = batch.validate(expect=suite)

    if not results.success:
        failed_expectations = []

        for result in results.results:
            if not result.success:
                failed_expectations.append({
                    "column": result.expectation_config.kwargs.get("column"),
                    "expected_dtype": result.expectation_config.kwargs.get("type_"),
                    "details": result.result
                })

        raise ValueError(f"Great Expectations failed for data types: {failed_expectations}")


def validate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a report of missing values and their percentages for each column."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot validate missing values in an empty DataFrame.")

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        missing_values_percentage = missing_values.div(len(df)).mul(100).round(2)

        missing_report = pd.DataFrame({
            "Columns": missing_values.index,
            "Missing": missing_values.values,
            "Missing %": missing_values_percentage.values
        })

        missing_report = missing_report.sort_values(
            by="Missing",
            ascending=False
        ).reset_index(drop=True)

        missing_report["Missing %"] = missing_report["Missing %"].astype(str) + "%"

    else:
        missing_report = pd.DataFrame(columns=["Columns", "Missing", "Missing %"])

    return missing_report


def validate_duplicates(df: pd.DataFrame) -> int:
    """Count duplicate rows in the DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    count_duplicates = int(df.duplicated().sum())

    return count_duplicates


def validate_domain_rules(df: pd.DataFrame) -> list[str]:
    """Validate domain-specific rules and return columns containing invalid values."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot validate domain rules on an empty DataFrame.")

    domain_rules = {
        "TAXI_OUT": lambda s: s >= 0,
        "TAXI_IN": lambda s: s >= 0,
        "AIR_TIME": lambda s: s >= 0,
        "DISTANCE": lambda s: s >= 0,
        "ACTUAL_ELAPSED_TIME": lambda s: s >= 0,
        "MONTH": lambda s: s.between(1, 12),
        "QUARTER": lambda s: s.between(1, 4),
        "DAY_OF_WEEK": lambda s: s.between(1, 7),
        "CANCELLED": lambda s: s.isin([0, 1]),
        "DIVERTED": lambda s: s.isin([0, 1]),
    }

    missing_columns = set(domain_rules) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for domain validation: {sorted(missing_columns)}")

    invalid_columns = []

    for column, rule in domain_rules.items():
        valid_values = rule(df[column])

        if not valid_values.fillna(False).all():
            invalid_columns.append(column)

    return invalid_columns