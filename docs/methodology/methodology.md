# Methodology

## 1. Project Overview

This project develops an end-to-end airline operations analysis and machine learning system using flight-level data from the U.S. Bureau of Transportation Statistics (BTS).

The methodology is designed to move from raw operational data to validated analysis, engineered features, machine learning models, and business-oriented findings.

The project follows this sequence:

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
Supervised + Unsupervised Learning
    ↓
Model Evaluation
    ↓
Business & Portfolio Outputs
```

The project deliberately combines traditional data analysis with machine learning. Machine learning is applied where it provides a meaningful analytical purpose rather than being included solely to demonstrate algorithms.

---

# 2. Data Source

The project uses airline operational data provided by the U.S. Bureau of Transportation Statistics (BTS).

The dataset contains flight-level records describing scheduled and actual flight operations, delays, cancellations, diversions, routes, aircraft identifiers, and delay causes.

Each row represents an individual flight record.

The original dataset contains **35 BTS variables**, which are documented in the project data dictionary.

The raw dataset is treated as the source layer and is not modified directly.

---

# 3. Data Ingestion

Data ingestion is handled through the project's loading layer.

The loading process performs the following checks before reading the dataset:

1. Convert the supplied path into a `Path` object.
2. Verify that the path exists.
3. Verify that the path refers to a file.
4. Verify that the file has a CSV extension.
5. Load the dataset using pandas.
6. Handle empty files and CSV parsing errors.
7. Verify that the resulting DataFrame contains rows.

This separates file-system and ingestion concerns from the subsequent data-quality and analytical stages.

The raw BTS dataset is intentionally excluded from version control because of its size and is expected to remain in the local raw-data layer.

---

# 4. Data Validation

Validation is performed before data preparation to distinguish data-quality problems from transformation logic.

The validation layer checks five areas:

### 4.1 Required Columns

The dataset is checked against the required BTS fields used by the project.

Missing required columns result in a validation error rather than allowing downstream functions to fail unexpectedly.

### 4.2 Data Types

Expected data types are compared with the actual DataFrame types for required variables.

This helps ensure that numerical, categorical, and temporal operations are performed on compatible data.

### 4.3 Missing Values

Missing values are identified and summarized by:

* Column
* Missing-value count
* Missing-value percentage

Missingness is reported rather than automatically treating every missing value as an error because missingness can have different meanings depending on the variable.

### 4.4 Duplicate Records

The dataset is checked for completely duplicated rows.

The number of duplicated records is reported before duplicate removal during preparation.

### 4.5 Domain Rules

Domain-specific validation checks whether values fall within meaningful operational ranges.

Examples include:

* `MONTH` must be between 1 and 12.
* `QUARTER` must be between 1 and 4.
* `DAY_OF_WEEK` must be between 1 and 7.
* `CANCELLED` must contain valid binary values.
* `DIVERTED` must contain valid binary values.
* Operational duration and distance variables must not contain negative values.

This layer establishes whether the dataset is structurally and logically suitable for further processing.

---

# 5. Data Preparation

The preparation layer transforms validated raw data into a cleaner analytical dataset.

The preparation workflow consists of five operations.

## 5.1 Missing-Value Handling

Delay-cause variables are filled with zero when no delay contribution is recorded:

* `CARRIER_DELAY`
* `NAS_DELAY`
* `SECURITY_DELAY`
* `LATE_AIRCRAFT_DELAY`

A cancellation reason variable is also created from the BTS cancellation code.

The mapping is:

| Code               | Meaning       |
| ------------------ | ------------- |
| A                  | Carrier       |
| B                  | Weather       |
| C                  | NAS           |
| D                  | Security      |
| No applicable code | Not Cancelled |

This transformation preserves the operational meaning of the original cancellation field while providing a more interpretable categorical variable.

## 5.2 Date Conversion

`FL_DATE` is converted into a pandas datetime representation.

Additional calendar variables are derived:

* `MONTH_NAME`
* `DAY_NAME`
* `DAY_OF_MONTH`
* `WEEK_OF_YEAR`
* `IS_WEEKEND`

These variables allow temporal analysis without repeatedly transforming the original date.

## 5.3 Categorical Cleaning

Selected categorical variables are standardized by removing leading and trailing whitespace.

This prevents visually identical categories from being treated as different values because of formatting inconsistencies.

## 5.4 Duplicate Removal

Completely duplicated records are removed from the analytical dataset.

The number of duplicates detected and removed is reported.

## 5.5 Preparation Principle

The preparation layer focuses on **cleaning and structural transformation**.

It does not perform model-specific preprocessing such as scaling or categorical encoding. Those operations belong to the machine learning pipeline where they can be applied appropriately to training and testing data.

---

# 6. Exploratory Data Analysis

Exploratory analysis is used to understand the operational behavior of the airline dataset before statistical testing and machine learning.

The analysis focuses on four areas.

## 6.1 Flight Volume

Flight volume is examined across:

* Year
* Month
* Day of week

This identifies temporal patterns in airline activity.

## 6.2 Delay Behavior

Delay analysis examines:

* Delay frequency
* Delay patterns by day
* Delay severity

The analysis distinguishes between flights that experienced delays and the magnitude of those delays.

## 6.3 Cancellation Behavior

Cancellation analysis examines:

* Overall cancellation frequency
* Cancellation reasons
* Cancellation patterns by day of week

The analysis uses the operational cancellation status together with the BTS cancellation categories.

## 6.4 Airport Activity

Airport activity is examined using origin and destination information.

The analysis identifies airports with the highest observed flight volumes and provides geographic context for later operational analysis.

---

# 7. Statistical Analysis

Statistical analysis extends the exploratory findings by testing whether observed differences or relationships have statistical evidence behind them.

The statistical layer uses methods appropriate to the analytical question.

## 7.1 Delay Difference Testing

A Welch independent-samples t-test is used to compare mean delay between two groups.

Welch's version is used because it does not require the two groups to have equal variances.

## 7.2 Delay by Day of Week

A one-way ANOVA is used to test whether mean departure delay differs across days of the week.

The test evaluates whether the observed group means provide evidence of a difference rather than relying only on visual comparison.

## 7.3 Correlation Analysis

Pearson correlation is used to measure linear relationships between selected numerical operational variables.

Correlation is interpreted as an association and is not treated as evidence of causation.

An alpha significance level is retained as part of the statistical testing workflow.

---

# 8. Business Analysis

Business analysis translates operational findings into airline-relevant measures.

It is intentionally separated from exploratory analysis.

The purpose is not simply to restate EDA results using business terminology, but to combine operational variables into measures that describe exposure and reliability.

The business-analysis layer contains four areas.

## 8.1 Operational Reliability

Operational reliability summarizes:

* Delayed flights
* Cancelled flights
* Diverted flights
* On-time flights

Both counts and percentages are considered to provide an operational view of flight reliability.

## 8.2 Delay Propagation

Delay propagation examines delayed departures and determines how often those delays remained present versus cases where the operation recovered.

This provides a more operationally meaningful view than simply counting delayed flights.

## 8.3 Operational Exposure

Operational exposure is analyzed at the airport level.

For each origin airport, the analysis considers:

* Flight volume
* Delayed flights
* Cancelled flights
* Delay rate
* Cancellation rate

This helps identify where operational disruption is concentrated.

## 8.4 Cancellation Exposure

Cancellation exposure combines airport information with cancellation reasons.

This allows cancellations to be examined not only by frequency but also by the type of operational cause associated with them.

---

# 9. Feature Engineering

Feature engineering converts raw and prepared variables into representations that are more useful for analysis and machine learning.

The project creates eight engineered features.

## 9.1 Time Features

`DEP_HOUR` is derived from scheduled departure time.

`TIME_OF_DAY` categorizes scheduled departures into:

| Period    | Hours |
| --------- | ----- |
| Night     | 00–05 |
| Morning   | 06–11 |
| Afternoon | 12–17 |
| Evening   | 18–23 |

## 9.2 Delay Features

`IS_DEP_DELAYED` identifies whether departure delay is greater than zero.

`DEP_DELAY_STATUS` provides a readable categorical representation:

* Not Delayed
* Delayed

`DEP_DELAY_SEVERITY` categorizes departure delay as:

* Early/On-Time
* Minor Delay
* Moderate Delay
* Severe Delay

The severity categories are based on the project's defined delay thresholds.

## 9.3 Route Feature

`ROUTE` combines origin and destination airports into a single route representation.

The resulting feature represents the directional relationship between the two airports.

## 9.4 Operational Features

`TOTAL_TAXI_TIME` combines taxi-out and taxi-in duration.

`DISTANCE_CATEGORY` groups flights into:

* Short
* Medium
* Long

These engineered variables provide more interpretable representations of temporal, operational, route, and delay characteristics.

---

# 10. Supervised Machine Learning

Supervised learning is used to model flight-level operational outcomes using labeled data.

The project treats supervised learning as a predictive component rather than replacing the preceding analytical stages.

The supervised workflow includes:

```text
Prepared / Engineered Data
        ↓
Feature Selection
        ↓
Train / Test Split
        ↓
Preprocessing
        ↓
Model Training
        ↓
Hyperparameter Search
        ↓
Cross-Validation
        ↓
Test Evaluation
```

## 10.1 Train-Test Separation

The data is separated into training and testing sets using:

* `test_size = 0.20`
* `random_state = 42`
* Stratification of the target

The test set is held out from model training and hyperparameter tuning.

This provides an independent dataset for final performance measurement.

## 10.2 Preprocessing

Model preprocessing includes numerical scaling and categorical encoding where required.

Categorical variables use one-hot encoding with:

```text
handle_unknown = "ignore"
```

This prevents previously unseen categories in evaluation data from causing encoding failures.

Preprocessing is incorporated into the modeling workflow so that transformations are applied consistently.

## 10.3 Model Families

The supervised workflow evaluates multiple model families:

* Logistic Regression
* Decision Tree
* Bagging
* Random Forest
* AdaBoost
* Gradient Boosting

The Decision Tree is included as an interpretable nonlinear baseline, while ensemble methods provide alternative approaches for capturing more complex relationships.

## 10.4 Hyperparameter Optimization

Hyperparameter optimization is performed using `RandomizedSearchCV`.

The tuning workflow uses a stratified development sample from the training data rather than repeatedly tuning on the full training dataset.

Cross-validation is performed within this development sample.

The final held-out test set is not used during hyperparameter tuning.

This design reduces computational cost while preserving the separation between development and final evaluation data.

---

# 11. Unsupervised Machine Learning

Unsupervised learning is used to identify groups of airports with similar operational characteristics without requiring a predefined target.

The project uses K-Means clustering.

## 11.1 Airport-Level Aggregation

Flight-level records are aggregated by origin airport.

The clustering dataset contains operational characteristics including:

* Flight volume
* Delayed flights
* Cancelled flights
* Diverted flights
* Average departure delay
* Average taxi-out time
* Average flight distance
* Departure delay rate
* Cancellation rate
* Diversion rate

This changes the unit of analysis from individual flights to airports.

## 11.2 Feature Scaling

The clustering variables are standardized using `StandardScaler`.

Scaling is important because the variables have different numerical magnitudes and units.

Without scaling, high-magnitude variables could disproportionately influence distance calculations.

## 11.3 Cluster Count Selection

Candidate values from **2 through 8 clusters** are evaluated.

For each candidate value, the workflow calculates:

* K-Means inertia
* Silhouette score

The selected cluster count is based on the highest silhouette score.

## 11.4 Cluster Profiling

After training, the airports are assigned to their respective clusters.

Cluster profiles summarize the operational characteristics of each group.

This makes the clustering results interpretable in terms of airport volume, delays, cancellations, diversions, taxi time, and flight distance.

---

# 12. Model Evaluation

Evaluation is used to measure predictive and clustering performance using metrics appropriate to each learning task.

## 12.1 Classification Metrics

The supervised models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

These metrics provide different views of classification performance.

Accuracy measures overall correctness, while precision and recall provide more detailed information about positive-class predictions.

F1-score balances precision and recall.

ROC-AUC measures ranking performance across classification thresholds.

## 12.2 Model Comparison

The models are evaluated using the same held-out test set after model development.

Performance metrics are collected into a structured evaluation result.

The project uses F1-score as the primary comparison metric because it balances precision and recall for the classification task.

Other metrics are retained to provide a broader performance picture.

## 12.3 Clustering Evaluation

K-Means clustering is evaluated using silhouette score.

The silhouette score measures how well observations fit within their assigned clusters relative to neighboring clusters.

The resulting score is interpreted together with the cluster profiles rather than as a standalone business conclusion.

---

# 13. Reproducibility

Reproducibility is supported through consistent project configuration and controlled randomness.

The project uses explicit random seeds where stochastic machine learning procedures are involved.

Examples include:

* Train/test splitting
* K-Means initialization
* Model tuning procedures

The project also defines its Python package configuration through `pyproject.toml` and records project dependencies through the repository's dependency configuration.

The raw BTS dataset is intentionally excluded from Git version control because of its file size.

---

# 14. Data and Methodological Limitations

Several limitations affect interpretation of the results.

## 14.1 Operational Dataset Scope

The dataset describes flight operations and does not contain all variables required to calculate complete airline financial impact.

Therefore, the project does not claim specific financial losses from delays or cancellations.

Variables such as ticket revenue, passenger counts, operating costs, fuel costs, and compensation costs are outside the available dataset scope.

## 14.2 Observational Data

The dataset is observational.

Statistical associations and machine learning relationships should therefore not automatically be interpreted as causal relationships.

## 14.3 Missing Values

Missing values can represent different operational circumstances.

The project therefore applies domain-aware treatment to selected variables instead of assuming that all missing values have the same meaning.

## 14.4 Model Development Efficiency

Hyperparameter tuning is performed on a development sample of the training data to make the multi-model search computationally practical.

This introduces a trade-off between computational efficiency and using the maximum available training data during tuning.

After model and hyperparameter selection, a strict production-oriented workflow could refit the selected final model on the complete training set before one final evaluation on the untouched test set.

---

# 15. Methodological Principles

The project follows several principles throughout the workflow:

### Separation of Responsibilities

Each project layer has a distinct responsibility:

```text
Loading
→ Access the data

Validation
→ Verify data quality

Preparation
→ Clean and transform the data

Analysis
→ Understand and test the data

Feature Engineering
→ Create useful representations

Machine Learning
→ Learn predictive or structural patterns

Evaluation
→ Measure model performance
```

### Domain-Aware Processing

Transformations are based on the operational meaning of BTS variables rather than applying generic transformations blindly.

### Reproducibility

Random states and project configuration are explicitly controlled where appropriate.

### Leakage Awareness

The test dataset is treated as held-out evaluation data during model development.

### Interpretability

Machine learning outputs are considered alongside descriptive, statistical, and business analysis rather than being treated as isolated model scores.

### No Forced Machine Learning

Machine learning methods are included where they answer meaningful predictive or structural questions.

---

# 16. End-to-End Methodological Summary

The completed methodology can be summarized as:

```text
BTS Flight Records
        ↓
Reliable Data Ingestion
        ↓
Structural & Domain Validation
        ↓
Domain-Aware Data Preparation
        ↓
Exploratory Analysis
        ↓
Statistical Testing
        ↓
Operational Business Analysis
        ↓
Feature Engineering
        ↓
────────────────────────────
↓                          ↓
Supervised Learning       Unsupervised Learning
↓                          ↓
Classification             Airport Clustering
↓                          ↓
Model Evaluation           Cluster Evaluation
────────────────────────────
        ↓
Integrated Operational Insights
        ↓
Portfolio Outputs
```

The resulting system provides a structured path from raw airline operational records to analytical findings and machine learning outputs while maintaining clear separation between data preparation, analysis, modeling, and evaluation.
