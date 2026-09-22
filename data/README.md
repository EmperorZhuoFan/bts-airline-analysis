# Data

This directory contains the datasets used throughout the BTS Airline Operations & Machine Learning System.

## Structure

```text
data/
├── raw/
└── processed/
```

### `raw/`

Contains the original BTS dataset exactly as obtained from the source.

Raw data is preserved separately so that the original source remains unchanged and reproducible.

The raw BTS dataset is **not committed to GitHub** because of its large file size.

### `processed/`

Contains datasets produced after validation, preparation, and transformation.

Processed data may be generated during execution and is kept separate from the original source data.

## Data Flow

```text
BTS Source Data
      ↓
data/raw/
      ↓
Loading
      ↓
Validation
      ↓
Preparation
      ↓
Feature Engineering
      ↓
data/processed/
```

## Data Source

The project uses flight-level operational data from the **U.S. Bureau of Transportation Statistics (BTS)**.

For variable definitions and transformations, see:

* [`../docs/data_dictionary/data_dictionary.md`](../docs/data_dictionary/data_dictionary.md)
* [`../docs/methodology/methodology.md`](../docs/methodology/methodology.md)
