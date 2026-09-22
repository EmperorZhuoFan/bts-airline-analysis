# BTS Airline Operations & Machine Learning System

## 1. Executive Summary

This project develops an end-to-end airline operations analysis and machine learning system using data from the U.S. Bureau of Transportation Statistics (BTS).

The system analyzes flight operations, delays, cancellations, airport activity, and operational characteristics before applying supervised and unsupervised machine learning techniques.

The project combines:

* Data ingestion and validation
* Data preparation
* Exploratory data analysis
* Statistical analysis
* Business-focused analysis
* Feature engineering
* Supervised machine learning
* Unsupervised machine learning
* Model evaluation
* Reporting and visualization

The supervised learning workflow evaluates multiple classification models for flight-delay prediction, while the unsupervised workflow groups airports according to their operational characteristics.

The project is designed as a reproducible data science and machine learning system rather than a collection of isolated analyses.

---

## 2. Problem Definition

Airline operations generate large amounts of operational data containing information about flight schedules, delays, cancellations, airports, routes, and operational performance.

The objective of this project is to transform raw BTS flight data into actionable operational analysis and machine learning outputs.

The project addresses several questions:

* How does flight volume change across time?
* When and how frequently do delays occur?
* What are the major cancellation patterns and causes?
* Which airports experience different levels of operational exposure?
* Are observed operational differences statistically meaningful?
* Which features can be engineered to better represent airline operations?
* How effectively can machine learning models classify flight-delay outcomes?
* Can airports be grouped according to similar operational characteristics?

The goal is not to force machine learning techniques onto the dataset, but to apply them where they provide meaningful analytical value.

---

## 3. Data & Scope

### Data Source

The dataset is derived from the U.S. Bureau of Transportation Statistics (BTS) airline transportation data.

The primary unit of observation is an individual flight record.

The raw dataset contains 35 original variables covering:

* Flight dates and scheduling
* Airlines and aircraft
* Origin and destination airports
* Departure and arrival information
* Departure and arrival delays
* Cancellation information
* Diversions
* Flight duration
* Taxi times
* Distance
* Delay causes

The original dataset is preserved separately from the transformed data to maintain a clear distinction between source data and project-generated data.

### Scope

The project focuses on operational characteristics of flights and airports.

The analysis does not attempt to estimate financial revenue, passenger impact, fuel costs, or direct monetary losses because those variables are not contained in the available dataset.

---

## 4. Methodology

The project follows a structured end-to-end workflow:

```text
Raw BTS Data
    ↓
Data Loading
    ↓
Data Validation
    ↓
Data Preparation
    ↓
Exploratory Analysis
    ↓
Statistical Analysis
    ↓
Business Analysis
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Evaluation
    ↓
Reporting & Visualization
```

### Data Loading

The loading layer validates the supplied file path and confirms that the source is a valid CSV file before loading it into a pandas DataFrame.

### Data Validation

The validation layer checks:

* Required columns
* Expected data types
* Missing values
* Duplicate records
* Domain-specific rules

Domain validation includes checks for valid month, quarter, and day-of-week ranges, binary cancellation and diversion indicators, and non-negative operational measurements.

### Data Preparation

The preparation stage transforms validated data while preserving its operational meaning.

Key operations include:

* Handling delay-cause missing values
* Creating cancellation reasons
* Converting flight dates to datetime
* Creating date-related fields
* Standardizing categorical values
* Removing duplicate records

### Exploratory Analysis

Exploratory analysis investigates:

* Flight volume
* Departure delays
* Delay severity
* Cancellation frequency
* Cancellation reasons
* Day-of-week patterns
* Airport activity

### Statistical Analysis

Statistical analysis extends the exploratory findings using:

* Welch's independent-samples t-test
* One-way ANOVA
* Pearson correlation

These methods are used to evaluate differences and relationships observed in the operational data.

### Business Analysis

Business analysis combines operational variables to evaluate:

* Operational reliability
* Delay propagation
* Operational exposure
* Cancellation exposure

The objective is to translate descriptive findings into airline-operational context without introducing unsupported financial assumptions.

---

## 5. Exploratory Findings

The exploratory analysis provides an initial understanding of airline operational behavior.

The analysis examines flight volume across time, departure-delay frequency and severity, cancellation behavior, cancellation causes, and airport activity.

Particular attention is given to the distinction between:

* Normal and delayed operations
* Minor, moderate, and severe delays
* Cancelled and non-cancelled flights
* Different cancellation causes
* Differences across days and airports

These findings establish the operational patterns investigated further through statistical and business analysis.

Detailed exploratory visualizations are produced through the project notebooks and analysis modules.

---

## 6. Statistical Findings

Statistical analysis is used to determine whether observed operational differences and relationships warrant further consideration beyond descriptive summaries.

### Delay Difference Testing

A Welch independent-samples t-test is used when comparing mean delay behavior between groups.

Welch's test is used because it does not require equal population variances between the compared groups.

### Day-of-Week Analysis

A one-way ANOVA is used to examine whether average departure delay differs across days of the week.

### Correlation Analysis

Pearson correlation is used to investigate linear relationships between selected numerical operational variables.

These statistical methods complement the exploratory analysis by providing formal tests of selected relationships.

Statistical significance is interpreted using the defined significance level rather than treating correlation or group differences as evidence of causation.

---

## 7. Business Analysis

The business analysis layer translates operational data into airline-relevant performance measures.

### Operational Reliability

Operational reliability is evaluated using measures such as:

* Delayed flights
* Cancelled flights
* Diverted flights
* On-time flights
* Corresponding operational rates

These measures provide an overall view of operational performance.

### Delay Propagation

The analysis examines whether flights experiencing departure delays remain delayed through subsequent operations or recover before arrival.

This helps distinguish initial operational disruption from delays that continue through the flight.

### Operational Exposure

Airport-level operational exposure is examined using:

* Flight volume
* Delayed flights
* Cancelled flights
* Delay rate
* Cancellation rate

This allows airports to be compared using both operational scale and disruption rates.

### Cancellation Exposure

Cancellation exposure is examined by combining airport activity with cancellation causes.

The analysis identifies how cancellation activity is distributed across operational locations and cancellation categories.

The business analysis deliberately avoids assigning monetary costs because the required financial variables are not available in the BTS dataset.

---

## 8. Feature Engineering

Feature engineering transforms the prepared dataset into a richer representation of airline operations.

The project creates eight engineered features.

### Time Features

**DEP_HOUR**

Represents the scheduled departure hour derived from `CRS_DEP_TIME`.

**TIME_OF_DAY**

Groups scheduled departure times into:

* Night
* Morning
* Afternoon
* Evening

### Delay Features

**IS_DEP_DELAYED**

Binary indicator representing whether departure delay is greater than zero.

**DEP_DELAY_STATUS**

Categorizes flights as:

* Not Delayed
* Delayed

**DEP_DELAY_SEVERITY**

Classifies departure delay into:

* Early/On-Time
* Minor Delay
* Moderate Delay
* Severe Delay

### Route Features

**ROUTE**

Represents the origin-to-destination airport relationship.

### Operational Features

**TOTAL_TAXI_TIME**

Combines taxi-out and taxi-in duration.

**DISTANCE_CATEGORY**

Groups flights into:

* Short
* Medium
* Long

These engineered variables provide additional operational representations for downstream analysis and machine learning.

---

## 9. Supervised Machine Learning

The supervised learning task is formulated as a classification problem.

The workflow evaluates six classification models:

1. Logistic Regression
2. Decision Tree
3. Bagging
4. Random Forest
5. AdaBoost
6. Gradient Boosting

The dataset is divided into training and held-out test data using:

* 80% training data
* 20% test data
* Random state: 42
* Stratification of the target variable

Model preprocessing and modeling steps are organized into machine learning pipelines.

Hyperparameter tuning is performed using `RandomizedSearchCV` with cross-validation on a stratified development sample.

This approach limits the computational cost of hyperparameter search while maintaining a completely held-out test set.

---

## 10. Unsupervised Machine Learning

The unsupervised learning workflow groups airports according to operational characteristics.

Instead of treating each flight as an independent clustering observation, flight records are aggregated at the airport level.

The clustering features include:

* Flight volume
* Delayed flights
* Cancelled flights
* Diverted flights
* Average departure delay
* Average taxi-out time
* Average flight distance
* Departure-delay rate
* Cancellation rate
* Diversion rate

The features are standardized using `StandardScaler`.

K-Means clustering is evaluated across cluster counts from 2 through 8.

Two evaluation measures are considered:

* Inertia
* Silhouette score

The final cluster count is selected using the highest silhouette score.

The resulting analysis identified **2 airport clusters**, with a silhouette score of approximately **0.564**.

The clusters are then profiled according to their operational characteristics to understand how the airport groups differ.

---

## 11. Model Evaluation

The supervised models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

F1-score is used as the primary comparison metric because it balances precision and recall for the classification task.

### Observed Model Results

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.6320 |    0.5423 | 0.4363 | 0.4836 |  0.6463 |
| Decision Tree       |   0.6021 |    0.4959 | 0.4539 | 0.4740 |  0.5763 |
| Bagging             |   0.6468 |    0.5764 | 0.3993 | 0.4718 |  0.6589 |
| Random Forest       |   0.6418 |    0.5700 | 0.3786 | 0.4550 |  0.6518 |
| AdaBoost            |   0.6445 |    0.6657 | 0.2007 | 0.3084 |  0.6551 |
| Gradient Boosting   |   0.6582 |    0.6318 | 0.3225 | 0.4270 |  0.6824 |

The models demonstrate different trade-offs between precision, recall, F1-score, and ROC-AUC.

The reported test-set results should be interpreted as evaluation measurements for this specific dataset and experimental configuration rather than as guarantees of future operational performance.

---

## 12. Operational Insights

The combined analytical workflow provides several categories of operational insight.

### Flight Operations

Flight volume varies across time, providing a basis for understanding periods of higher and lower operational activity.

### Delay Behavior

Departure delays exhibit different levels of severity, allowing operational disruptions to be separated into more manageable and more severe categories.

### Cancellation Behavior

Cancellation patterns can be examined by cause, time, and airport, providing a clearer picture of where operational disruptions occur.

### Airport Differences

Airport-level aggregation reveals that airports do not share identical operational characteristics.

The clustering analysis identifies distinct airport groups based on operational scale, delays, cancellations, diversions, taxi times, and distance characteristics.

### Machine Learning

The supervised models demonstrate that flight operational variables contain predictive information for the selected classification task, although model performance varies considerably across algorithms and metrics.

The results therefore provide a baseline for further model development rather than representing a production-ready prediction system.

---

## 13. Limitations

Several limitations should be considered when interpreting the results.

### Dataset Limitations

The available BTS data does not contain all variables required to estimate complete financial or passenger-level business impact.

Examples include:

* Ticket revenue
* Passenger counts
* Operating costs
* Fuel costs
* Compensation costs

Therefore, the project focuses on operational exposure rather than direct financial impact.

### Observational Data

The dataset represents observed airline operations.

Statistical relationships and correlations should not automatically be interpreted as causal relationships.

### Machine Learning Limitations

The supervised models are evaluated using a held-out test set, but hyperparameter tuning is performed on a development sample to reduce computational cost.

A stricter final modeling workflow could retrain the selected final model on the complete training set after model selection and before performing the final test evaluation.

### Model Selection Consideration

Model performance varies depending on the selected evaluation metric.

Accuracy, precision, recall, F1-score, and ROC-AUC can provide different perspectives on model behavior.

Therefore, model evaluation should consider the operational objective rather than relying on a single metric in isolation.

### Clustering Limitations

K-Means assumes that observations can be meaningfully represented using the selected numerical features and that cluster structure is appropriate for the data.

The resulting clusters should therefore be interpreted as operational groupings based on the selected variables rather than fixed real-world categories.

---

## 14. Reproducibility

The project is organized so that the workflow can be reproduced from the original BTS data.

The main execution path follows:

```text
data/raw/
    ↓
Loading
    ↓
Validation
    ↓
Preparation
    ↓
Analysis
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Evaluation
```

The project uses a structured source-code layout under `src/airline/`.

Supporting documentation is provided under `docs/`.

The notebooks provide an interactive environment for investigating and demonstrating the main analytical and machine learning stages.

The raw BTS dataset is intentionally excluded from the Git repository because of its large file size.

To reproduce the project:

1. Obtain the required BTS dataset from the official source.
2. Place the dataset in `data/raw/`.
3. Install the project dependencies.
4. Run the project entry point.
5. Use the notebooks for exploratory investigation and model analysis.
6. Run the test suite using `pytest`.

The project also includes `pyproject.toml` to define package metadata and support a structured Python project layout.

---

## 15. Final Outputs

The completed project provides several complementary outputs.

### Source Code

A modular Python system covering:

* Data loading
* Validation
* Preparation
* Analysis
* Feature engineering
* Supervised learning
* Unsupervised learning
* Evaluation

### Documentation

Project documentation includes:

* System architecture
* Data pipeline
* Data dictionary
* Methodology
* Final report

### Notebooks

The project includes focused notebooks for:

* Data understanding
* Exploratory data analysis
* Feature engineering
* Model analysis

### Machine Learning

The machine learning layer provides:

* Multiple supervised classification models
* Hyperparameter tuning
* Cross-validation
* Airport-level K-Means clustering
* Cluster profiling
* Model evaluation

### Visualization & Reporting

Selected figures and analytical outputs are organized under the reports directory, while the Power BI dashboard provides an additional interactive visualization layer.

### Overall Project Outcome

The final system demonstrates an end-to-end workflow that transforms raw airline transportation data into structured analysis, engineered features, machine learning experiments, operational insights, and documented portfolio outputs.

The project is designed to demonstrate not only individual machine learning techniques, but also the ability to organize a complete data science workflow around a real-world operational dataset.
