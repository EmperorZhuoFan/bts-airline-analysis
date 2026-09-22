import pandas as pd
from pathlib import Path


def validate_file_path(file_path: str | Path) -> Path:
    """Validate that the provided path exists, is a file, and is a CSV file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"The file not found: {path}")

    if not path.is_file():
        raise ValueError(f"The path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file , received: {path.suffix}")

    return path


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the BTS CSV dataset into a pandas DataFrame."""

    validated_path = validate_file_path(file_path)

    try:
        df = pd.read_csv(validated_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The dataset is empty") from error
    except pd.errors.ParserError as error: 
        raise ValueError("The CSV file could not be parsed") from error 

    if df.empty:
        raise ValueError("The DataFrame has no rows")
  
    print("=" * 70)
    print(f"Dataset loaded: {validated_path.name}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print("=" * 70)

    return df

def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the processed DataFrame as a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print("=" * 70)
    print(f"Processed dataset saved: {path}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print("=" * 70)

    return path