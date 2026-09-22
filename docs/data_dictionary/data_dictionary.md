# BTS Airline Analysis & Machine Learning System

## Data Dictionary

## 1. Overview

This document defines the variables used throughout the BTS Airline Analysis & Machine Learning System.

The project uses flight-level operational data from the **U.S. Bureau of Transportation Statistics (BTS)** and transforms the original dataset through validation, preparation, analysis, and feature engineering before downstream machine learning and reporting.

The data dictionary distinguishes between:

* **Original BTS fields** — fields provided by the source dataset.
* **Prepared fields** — fields created during data preparation.
* **Engineered features** — fields created during feature engineering.

---

# 2. Dataset Overview

| Property            | Description                                          |
| ------------------- | ---------------------------------------------------- |
| Source              | U.S. Bureau of Transportation Statistics (BTS)       |
| Unit of observation | Individual flight                                    |
| Original columns    | 35                                                   |
| Prepared fields     | 6                                                    |
| Engineered features | 8                                                    |
| Original schema     | Preserved separately from project-generated features |

---

# 3. Original BTS Fields

## 3.1 Flight & Calendar Information

| Column        | Description                                                                        | Data Type                  | Role     |
| ------------- | ---------------------------------------------------------------------------------- | -------------------------- | -------- |
| `YEAR`        | Year in which the flight operated.                                                 | Integer                    | Temporal |
| `QUARTER`     | Calendar quarter in which the flight operated, from 1 to 4.                        | Integer                    | Temporal |
| `MONTH`       | Month in which the flight operated, from 1 to 12.                                  | Integer                    | Temporal |
| `DAY_OF_WEEK` | Day of the week on which the flight operated, represented numerically from 1 to 7. | Integer                    | Temporal |
| `FL_DATE`     | Date associated with the flight operation.                                         | Datetime after preparation | Temporal |

---

## 3.2 Airline & Aircraft Information

| Column              | Description                                                        | Data Type   | Role     |
| ------------------- | ------------------------------------------------------------------ | ----------- | -------- |
| `OP_UNIQUE_CARRIER` | Unique code identifying the operating air carrier.                 | Categorical | Airline  |
| `TAIL_NUM`          | Aircraft tail number identifying the aircraft used for the flight. | Categorical | Aircraft |

---

## 3.3 Route & Geographic Information

| Column             | Description                                                 | Data Type   | Role      |
| ------------------ | ----------------------------------------------------------- | ----------- | --------- |
| `ORIGIN`           | Code identifying the origin airport.                        | Categorical | Route     |
| `ORIGIN_CITY_NAME` | City associated with the origin airport.                    | Categorical | Geography |
| `ORIGIN_STATE_ABR` | State abbreviation associated with the origin airport.      | Categorical | Geography |
| `DEST`             | Code identifying the destination airport.                   | Categorical | Route     |
| `DEST_CITY_NAME`   | City associated with the destination airport.               | Categorical | Geography |
| `DEST_STATE_ABR`   | State abbreviation associated with the destination airport. | Categorical | Geography |

---

## 3.4 Scheduled & Actual Flight Times

| Column          | Description                                                                  | Data Type | Role       |
| --------------- | ---------------------------------------------------------------------------- | --------- | ---------- |
| `CRS_DEP_TIME`  | Scheduled departure time.                                                    | Integer   | Schedule   |
| `DEP_TIME`      | Actual departure time.                                                       | Numeric   | Operations |
| `DEP_DELAY`     | Difference between scheduled and actual departure time, measured in minutes. | Numeric   | Delay      |
| `DEP_DELAY_NEW` | Non-negative representation of departure delay.                              | Numeric   | Delay      |
| `CRS_ARR_TIME`  | Scheduled arrival time.                                                      | Integer   | Schedule   |
| `ARR_TIME`      | Actual arrival time.                                                         | Numeric   | Operations |
| `ARR_DELAY`     | Difference between scheduled and actual arrival time, measured in minutes.   | Numeric   | Delay      |
| `ARR_DELAY_NEW` | Non-negative representation of arrival delay.                                | Numeric   | Delay      |
| `ARR_DEL15`     | Indicator identifying whether the flight arrived at least 15 minutes late.   | Binary    | Delay      |

---

## 3.5 Operational Measurements

| Column                | Description                                                                  | Data Type | Role             |
| --------------------- | ---------------------------------------------------------------------------- | --------- | ---------------- |
| `TAXI_OUT`            | Time between leaving the departure gate and takeoff, measured in minutes.    | Numeric   | Operations       |
| `TAXI_IN`             | Time between landing and reaching the destination gate, measured in minutes. | Numeric   | Operations       |
| `ACTUAL_ELAPSED_TIME` | Total actual elapsed flight time, including taxi time.                       | Numeric   | Operations       |
| `AIR_TIME`            | Time spent in the air, measured in minutes.                                  | Numeric   | Operations       |
| `DISTANCE`            | Distance between the origin and destination airports, measured in miles.     | Numeric   | Route/Operations |

---

## 3.6 Cancellations & Diversions

| Column              | Description                                                                            | Data Type   | Role       |
| ------------------- | -------------------------------------------------------------------------------------- | ----------- | ---------- |
| `CANCELLED`         | Binary indicator showing whether the flight was cancelled.                             | Binary      | Disruption |
| `CANCELLATION_CODE` | Code representing the recorded reason for cancellation.                                | Categorical | Disruption |
| `DIVERTED`          | Binary indicator showing whether the flight was diverted from its planned destination. | Binary      | Disruption |

---

## 3.7 Delay Causes

| Column                | Description                                                                     | Data Type | Role        |
| --------------------- | ------------------------------------------------------------------------------- | --------- | ----------- |
| `CARRIER_DELAY`       | Minutes of delay attributed to the operating carrier.                           | Numeric   | Delay Cause |
| `NAS_DELAY`           | Minutes of delay attributed to the National Airspace System.                    | Numeric   | Delay Cause |
| `SECURITY_DELAY`      | Minutes of delay attributed to security-related issues.                         | Numeric   | Delay Cause |
| `LATE_AIRCRAFT_DELAY` | Minutes of delay attributed to a late-arriving aircraft from a previous flight. | Numeric   | Delay Cause |

---

# 4. Data Preparation Fields

These fields are created by:

`src/airline/data/preparation.py`

They are **not part of the original BTS schema**.

| Column                | Description                                                              | Data Type   | Source / Transformation   |
| --------------------- | ------------------------------------------------------------------------ | ----------- | ------------------------- |
| `CANCELLATION_REASON` | Human-readable cancellation category derived from `CANCELLATION_CODE`.   | Categorical | Cancellation-code mapping |
| `MONTH_NAME`          | Full month name derived from `FL_DATE`.                                  | Categorical | `FL_DATE`                 |
| `DAY_NAME`            | Full day name derived from `FL_DATE`.                                    | Categorical | `FL_DATE`                 |
| `DAY_OF_MONTH`        | Day of the month extracted from `FL_DATE`.                               | Integer     | `FL_DATE`                 |
| `WEEK_OF_YEAR`        | ISO week number extracted from `FL_DATE`.                                | Integer     | `FL_DATE`                 |
| `IS_WEEKEND`          | Indicator identifying whether the flight occurred on Saturday or Sunday. | Boolean     | `FL_DATE`                 |

## 4.1 Cancellation Reason Mapping

The project converts cancellation codes into readable categories:

| BTS Code                        | Project Category |
| ------------------------------- | ---------------- |
| `A`                             | Carrier          |
| `B`                             | Weather          |
| `C`                             | NAS              |
| `D`                             | Security         |
| No applicable cancellation code | Not Cancelled    |

The transformation improves interpretability during exploratory and business analysis.

---

# 5. Engineered Features

These features are created by:

`src/airline/features/engineering.py`

They are generated after data preparation and are intended to provide more useful representations of the original operational variables.

---

## 5.1 Time Features

| Column        | Description                                                                       | Data Type   | Source         |
| ------------- | --------------------------------------------------------------------------------- | ----------- | -------------- |
| `DEP_HOUR`    | Scheduled departure hour extracted from `CRS_DEP_TIME`.                           | Integer     | `CRS_DEP_TIME` |
| `TIME_OF_DAY` | Categorization of scheduled departure into Night, Morning, Afternoon, or Evening. | Categorical | `DEP_HOUR`     |

### Time-of-Day Bins

| Departure Hour | Category  |
| -------------- | --------- |
| 00–05          | Night     |
| 06–11          | Morning   |
| 12–17          | Afternoon |
| 18–23          | Evening   |

---

## 5.2 Delay Features

| Column               | Description                                                                        | Data Type   | Source           |
| -------------------- | ---------------------------------------------------------------------------------- | ----------- | ---------------- |
| `IS_DEP_DELAYED`     | Binary indicator identifying whether departure delay is greater than zero minutes. | Binary      | `DEP_DELAY`      |
| `DEP_DELAY_STATUS`   | Categorical representation of whether the flight was delayed.                      | Categorical | `IS_DEP_DELAYED` |
| `DEP_DELAY_SEVERITY` | Categorization of departure delay according to its severity.                       | Categorical | `DEP_DELAY`      |

### Departure Delay Status

| Value | Meaning     |
| ----- | ----------- |
| `0`   | Not Delayed |
| `1`   | Delayed     |

### Departure Delay Severity

| `DEP_DELAY`            | Category       |
| ---------------------- | -------------- |
| ≤ 0 minutes            | Early/On-Time  |
| > 0 and ≤ 30 minutes   | Minor Delay    |
| > 30 and ≤ 120 minutes | Moderate Delay |
| > 120 minutes          | Severe Delay   |

This representation allows the project to analyze both the occurrence and severity of departure delays.

---

## 5.3 Route Features

| Column  | Description                                                                  | Data Type   | Source           |
| ------- | ---------------------------------------------------------------------------- | ----------- | ---------------- |
| `ROUTE` | Combined origin and destination airport codes representing the flight route. | Categorical | `ORIGIN`, `DEST` |

The route is constructed using:

```text
ORIGIN --> DEST
```

This provides a single categorical representation of the complete origin-destination relationship.

---

## 5.4 Operational Features

| Column              | Description                                                    | Data Type   | Source                |
| ------------------- | -------------------------------------------------------------- | ----------- | --------------------- |
| `TOTAL_TAXI_TIME`   | Combined taxi-out and taxi-in time for the flight.             | Numeric     | `TAXI_OUT`, `TAXI_IN` |
| `DISTANCE_CATEGORY` | Categorization of flight distance into Short, Medium, or Long. | Categorical | `DISTANCE`            |

### Distance Categories

The project uses the following intervals:

| Distance                | Category |
| ----------------------- | -------- |
| > 0 and ≤ 500 miles     | Short    |
| > 500 and ≤ 1,000 miles | Medium   |
| > 1,000 miles           | Long     |

---

# 6. Variable Classification

The project's variables can be grouped into the following analytical categories.

### Temporal Variables

`YEAR`, `QUARTER`, `MONTH`, `DAY_OF_WEEK`, `FL_DATE`, `MONTH_NAME`, `DAY_NAME`, `DAY_OF_MONTH`, `WEEK_OF_YEAR`, `IS_WEEKEND`, `DEP_HOUR`, `TIME_OF_DAY`

### Airline & Aircraft Variables

`OP_UNIQUE_CARRIER`, `TAIL_NUM`

### Route & Geographic Variables

`ORIGIN`, `ORIGIN_CITY_NAME`, `ORIGIN_STATE_ABR`, `DEST`, `DEST_CITY_NAME`, `DEST_STATE_ABR`, `ROUTE`, `DISTANCE`, `DISTANCE_CATEGORY`

### Schedule & Timing Variables

`CRS_DEP_TIME`, `DEP_TIME`, `CRS_ARR_TIME`, `ARR_TIME`

### Delay Variables

`DEP_DELAY`, `DEP_DELAY_NEW`, `ARR_DELAY`, `ARR_DELAY_NEW`, `ARR_DEL15`, `IS_DEP_DELAYED`, `DEP_DELAY_STATUS`, `DEP_DELAY_SEVERITY`

### Operational Variables

`TAXI_OUT`, `TAXI_IN`, `TOTAL_TAXI_TIME`, `ACTUAL_ELAPSED_TIME`, `AIR_TIME`

### Disruption Variables

`CANCELLED`, `CANCELLATION_CODE`, `CANCELLATION_REASON`, `DIVERTED`

### Delay-Cause Variables

`CARRIER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`

---

# 7. Machine Learning Considerations

The presence of a variable in the dataset does not automatically mean that it should be used as a machine-learning feature.

Feature selection depends on:

* Prediction objective
* Data availability
* Data quality
* Feature relevance
* Feature representation
* Model requirements
* Potential information leakage

Categorical variables may require encoding before model training.

Numerical variables may require scaling depending on the algorithm.

Target variables must be separated from predictive features before model training.

The project therefore follows this conceptual transformation:

```text
Original BTS Variables
        ↓
Prepared Variables
        ↓
Engineered Features
        ↓
Selected Model Inputs
        ↓
Machine Learning
```

This separation makes the data lineage explicit and helps reduce the risk of accidentally using inappropriate or leakage-prone variables.

---

# 8. Data Lineage

The project's data transformation follows:

```text
BTS Raw Dataset
        ↓
Loading
        ↓
Validation
        ↓
Data Preparation
        ↓
Prepared Dataset
        ↓
Feature Engineering
        ↓
Engineered Dataset
        ↓
Analysis / Machine Learning
```

The raw BTS dataset remains the source layer, while preparation and feature engineering create progressively more useful representations for downstream analysis and machine learning.

---

# 9. Source vs Project-Generated Variables

The project maintains a clear distinction between source data and transformations.

### Original BTS Fields

These variables are supplied by the BTS source dataset.

Examples:

`YEAR`, `ORIGIN`, `DEST`, `DEP_DELAY`, `ARR_DELAY`, `CANCELLED`, `DISTANCE`

### Prepared Fields

These variables are created during data cleaning and transformation.

Examples:

`CANCELLATION_REASON`, `MONTH_NAME`, `DAY_NAME`, `DAY_OF_MONTH`, `WEEK_OF_YEAR`, `IS_WEEKEND`

### Engineered Features

These variables are intentionally created during feature engineering.

Examples:

`DEP_HOUR`, `TIME_OF_DAY`, `IS_DEP_DELAYED`, `DEP_DELAY_STATUS`, `DEP_DELAY_SEVERITY`, `ROUTE`, `TOTAL_TAXI_TIME`, `DISTANCE_CATEGORY`

---

# 10. Documentation and Code Alignment

This data dictionary should remain synchronized with the project's implementation.

Primary implementation files:

* `src/airline/data/loading.py`
* `src/airline/data/validation.py`
* `src/airline/data/preparation.py`
* `src/airline/features/engineering.py`
* `src/airline/models/supervised.py`
* `src/airline/models/unsupervised.py`
* `src/airline/models/evaluation.py`

If the source schema, preparation logic, or feature-engineering logic changes, this document should be updated accordingly.

The data dictionary therefore serves as the documented reference for the project's data schema and transformation lineage.
