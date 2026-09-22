# BTS Airline Operations & Machine Learning System

> An end-to-end data science and machine learning system for analyzing airline operational performance, identifying disruption patterns, and uncovering airport-level operational profiles using U.S. Bureau of Transportation Statistics (BTS) flight data.

---

## Overview

This project develops a modular, end-to-end workflow that transforms raw airline operational records into validated data, analytical insights, engineered features, machine learning outputs, and business-oriented findings.

The system combines traditional data analysis with supervised and unsupervised machine learning while maintaining a clear separation between data processing, analysis, modeling, and evaluation.

### Core Areas

* **Data Engineering** — ingestion, validation, preparation, and feature engineering
* **Exploratory Analysis** — flight volume, delays, cancellations, and airport activity
* **Statistical Analysis** — hypothesis testing and correlation analysis
* **Business Analysis** — operational reliability, delay propagation, and operational exposure
* **Supervised Machine Learning** — flight-level classification using multiple model families
* **Unsupervised Machine Learning** — airport-level operational clustering
* **Model Evaluation** — classification and clustering performance assessment
* **Business Intelligence** — downstream reporting and visualization

---

# Project Objective

The project is structured around three analytical questions.

### 1. What is happening?

Understand airline operational behavior through:

* Flight volume
* Departure delays
* Cancellations
* Diversions
* Airport activity
* Temporal patterns
* Delay severity

### 2. Is it statistically meaningful?

Use statistical methods to investigate:

* Differences in delay behavior
* Delay variation across days of the week
* Relationships between operational variables

### 3. What can machine learning reveal?

Use machine learning to:

* Model flight-level operational outcomes
* Compare multiple supervised learning approaches
* Identify airport groups with similar operational characteristics

The objective is to build a complete analytical system rather than a collection of disconnected notebooks or models.

---

# System Architecture

The project follows a layered architecture in which each stage has a defined responsibility.

```text
                    BTS RAW DATA
                         │
                         ▼
                  DATA INGESTION
                         │
                         ▼
                  DATA VALIDATION
                         │
                         ▼
                  DATA PREPARATION
                         │
                         ▼
              EXPLORATION & ANALYSIS
                         │
                         ▼
                FEATURE ENGINEERING
                         │
                         ▼
                 MACHINE LEARNING
                  ┌──────┴──────┐
                  ▼             ▼
             SUPERVISED    UNSUPERVISED
                  │             │
                  └──────┬──────┘
                         ▼
                     EVALUATION
                         │
                         ▼
                  FINAL OUTPUTS
```

Detailed architecture documentation is available in:

* [`docs/architecture/system_architecture.png`](docs/architecture/system_architecture.png)
* [`docs/architecture/data_pipeline.png`](docs/architecture/data_pipeline.png)

---

# Data Pipeline

The data pipeline transforms raw BTS records into analysis- and machine-learning-ready data.

```text
BTS RAW DATA
     ↓
DATA LOADING
     ↓
DATA VALIDATION
     ↓
DATA PREPARATION
     ↓
ANALYTICAL DATASET
     ↓
FEATURE ENGINEERING
     ↓
ML-READY DATA
```

### Data Loading

Validates the dataset path and loads the BTS CSV into a pandas DataFrame.

### Data Validation

Checks:

* Required columns
* Data types
* Missing values
* Duplicate records
* Domain rules

### Data Preparation

Performs domain-aware transformations including:

* Missing-value handling
* Date conversion
* Categorical cleaning
* Duplicate removal
* Cancellation-reason creation

### Feature Engineering

Creates operational features representing:

* Departure hour
* Time of day
* Delay status
* Delay severity
* Flight route
* Total taxi time
* Distance category

---

# Dataset

## Source

**U.S. Bureau of Transportation Statistics (BTS)**

The dataset contains flight-level operational information including scheduling, actual flight performance, delays, cancellations, diversions, routes, aircraft identifiers, and delay causes.

### Unit of Observation

Each row represents an individual flight record.

### Original Variables

The raw dataset contains **35 original BTS variables**.

The complete variable definitions and project-generated features are documented in:

[`docs/data_dictionary/data_dictionary.md`](docs/data_dictionary/data_dictionary.md)

---

# Analytical Methodology

The project uses a layered analytical approach.

## Exploratory Analysis

Exploratory analysis investigates:

* Flight volume over time
* Delay frequency
* Delay severity
* Cancellation frequency
* Cancellation reasons
* Day-of-week patterns
* Airport activity

The purpose is to understand the structure and behavior of the operational data before applying statistical or machine learning methods.

## Statistical Analysis

Statistical methods are used to test relationships and differences identified during exploration.

Methods include:

* Welch's independent-samples t-test
* One-way ANOVA
* Pearson correlation

Statistical results are interpreted as evidence of association or group differences rather than automatically as causal relationships.

## Business Analysis

Business analysis translates operational findings into airline-relevant measures.

The analysis focuses on:

* Operational reliability
* Delay propagation
* Operational exposure
* Cancellation exposure

The project deliberately avoids inventing financial impact because the BTS dataset does not provide sufficient information about passenger revenue, operating costs, fuel costs, or other financial variables.

---

# Machine Learning

Machine learning is used as an extension of the analytical workflow rather than as an isolated demonstration.

## Supervised Learning

The supervised learning workflow evaluates multiple classification model families:

* Logistic Regression
* Decision Tree
* Bagging
* Random Forest
* AdaBoost
* Gradient Boosting

The workflow includes:

```text
Training Data
     ↓
Feature Preparation
     ↓
Train / Test Split
     ↓
Model Training
     ↓
Hyperparameter Search
     ↓
Cross-Validation
     ↓
Held-Out Test Evaluation
```

### Train/Test Strategy

The dataset is divided into training and test sets using:

* `test_size = 0.20`
* `random_state = 42`
* Stratification

The test set is kept outside the hyperparameter-development process.

### Hyperparameter Optimization

`RandomizedSearchCV` is used to explore model configurations.

To keep computational requirements practical, tuning is performed using a stratified development sample from the training data with cross-validation.

---

# Unsupervised Learning

The unsupervised component uses **K-Means clustering** to identify groups of airports with similar operational characteristics.

## Airport-Level Representation

Flight-level records are aggregated by origin airport.

The clustering features include:

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

## Clustering Workflow

```text
Flight-Level Data
       ↓
Airport Aggregation
       ↓
Feature Selection
       ↓
Standardization
       ↓
K-Means Evaluation
       ↓
Cluster Selection
       ↓
Final Clustering
       ↓
Cluster Profiling
```

Candidate cluster counts from **2 to 8** are evaluated using:

* Inertia
* Silhouette score

The final clustering configuration produced **2 operational airport clusters**, with a silhouette score of approximately **0.564**.

Cluster profiling is then used to interpret the operational characteristics of each group.

---

# Model Evaluation

Different learning tasks require different evaluation strategies.

## Classification

Supervised models are evaluated using:

| Metric    | Purpose                               |
| --------- | ------------------------------------- |
| Accuracy  | Overall classification correctness    |
| Precision | Reliability of positive predictions   |
| Recall    | Ability to identify positive cases    |
| F1-score  | Balance between precision and recall  |
| ROC-AUC   | Ranking performance across thresholds |

F1-score is used as the primary comparison metric because it balances precision and recall.

## Clustering

K-Means clustering is evaluated using:

* Silhouette score
* Cluster profiles
* Operational characteristics of the resulting groups

The numerical score is interpreted together with the actual characteristics of each cluster.

---
## Power BI Dashboard

The project includes an interactive Power BI dashboard designed to present
operational performance, delay patterns, cancellation behavior, and machine
learning results.

### Dashboard Pages

1. **Executive Overview**
   - Flight volume
   - Delay, cancellation, and diversion rates
   - Operational performance by day

2. **Delays & Cancellations**
   - Delay severity
   - Cancellation patterns
   - Delay rate by time of day
   - Airport operational exposure

3. **Machine Learning & Airport Segmentation**
   - Supervised model performance
   - Model comparison
   - Airport cluster distribution
   - Airport cluster profiles

### Power BI File

The Power BI dashboard file is available locally as:

`reports/airline_bts_dashboard.pbix`
---

# Project Structure

```text
airline-bts-ml-system/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── architecture/
│   │   ├── system_architecture.png
│   │   └── data_pipeline.png
│   │
│   ├── data_dictionary/
│   │   └── data_dictionary.md
│   │
│   └── methodology/
│       └── methodology.md
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_analysis.ipynb
│
├── src/
│   └── airline/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── data/
│       │   ├── loading.py
│       │   ├── validation.py
│       │   └── preparation.py
│       │
│       ├── analysis/
│       │   ├── exploratory_analysis.py
│       │   ├── statistical_analysis.py
│       │   └── business_analysis.py
│       │
│       ├── features/
│       │   └── engineering.py
│       │
│       └── models/
│           ├── supervised.py
│           ├── unsupervised.py
│           └── evaluation.py
│
├── tests/
│   ├── test_loading.py
│   ├── test_validation.py
│   ├── test_preparation.py
│   ├── test_features.py
│   └── test_models.py
│
├── models/
│   └── README.md
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
└── .github/
    └── workflows/
        └── tests.yml
```

---

# Source Code Organization

The source code is organized according to responsibility.

| Module                    | Responsibility                                  |
| ------------------------- | ----------------------------------------------- |
| `loading.py`              | Load and validate the dataset path              |
| `validation.py`           | Verify data quality and domain constraints      |
| `preparation.py`          | Clean and transform raw data                    |
| `exploratory_analysis.py` | Explore operational patterns                    |
| `statistical_analysis.py` | Perform statistical tests                       |
| `business_analysis.py`    | Translate findings into operational measures    |
| `engineering.py`          | Create machine-learning and analytical features |
| `supervised.py`           | Train supervised models                         |
| `unsupervised.py`         | Perform airport clustering                      |
| `evaluation.py`           | Evaluate learning outputs                       |
| `main.py`                 | Orchestrate the complete workflow               |

This separation keeps individual components testable, reusable, and easier to maintain.

---

# Technology Stack

### Programming

* Python
* pandas
* NumPy

### Data Analysis

* SciPy
* Matplotlib
* Seaborn

### Machine Learning

* scikit-learn

### Documentation & Development

* Git
* GitHub
* Markdown
* Jupyter Notebooks

### Business Intelligence

* Microsoft Power BI

---

# Reproducibility

The project uses explicit configuration and controlled randomness where stochastic procedures are involved.

Examples include:

```text
random_state = 42
```

The project also uses:

* `pyproject.toml` for package configuration
* Dependency configuration for reproducible environments
* Modular Python source files
* Separate raw and processed data layers
* Dedicated documentation
* Automated testing structure

The large BTS raw dataset is intentionally excluded from Git version control.

---

# Documentation

Detailed project documentation is maintained separately from the source code.

### Architecture

[`docs/architecture/`](docs/architecture/)

Contains the system architecture and data-pipeline diagrams.

### Data Dictionary

[`docs/data_dictionary/data_dictionary.md`](docs/data_dictionary/data_dictionary.md)

Documents the original BTS variables and all project-generated variables.

### Methodology

[`docs/methodology/methodology.md`](docs/methodology/methodology.md)

Documents the complete analytical and machine-learning methodology.

---

# Notebooks

The notebooks provide an investigative and analytical workspace while the reusable project logic remains inside `src/airline/`.

Planned notebooks include:

| Notebook                       | Purpose                                      |
| ------------------------------ | -------------------------------------------- |
| `01_data_understanding.ipynb`  | Understand the dataset structure and quality |
| `02_eda.ipynb`                 | Explore operational patterns                 |
| `03_feature_engineering.ipynb` | Investigate engineered features              |
| `04_model_analysis.ipynb`      | Analyze machine-learning results             |

The notebooks are intended to document experimentation and reasoning, while production-oriented logic remains in the source package.

---

# Testing

The project includes a dedicated `tests/` structure for validating core components.

Testing areas include:

* Data loading
* Data validation
* Data preparation
* Feature engineering
* Machine learning components

The testing structure is designed to support future continuous integration through GitHub Actions.

---

# Key Methodological Considerations

The project intentionally recognizes several limitations.

### Computational Efficiency

Hyperparameter tuning uses a development sample rather than repeatedly processing the complete training dataset.

This reduces computational cost while maintaining a separate held-out test set.

### Test-Set Discipline

The test dataset is kept outside the hyperparameter-tuning process.

For a stricter production deployment workflow, the final selected model could be refit on the complete training dataset before performing one final evaluation against the untouched test set.

### Financial Interpretation

The BTS operational dataset does not contain sufficient financial information to directly calculate airline revenue loss or operational cost.

Therefore, business analysis focuses on measurable operational exposure rather than fabricated financial estimates.

### Observational Data

The dataset is observational. Statistical relationships and model associations should not automatically be interpreted as causal effects.

---

# Project Outputs

The project is designed to produce several forms of output:

```text
Raw Data
   ↓
Validated Dataset
   ↓
Prepared Dataset
   ↓
Analytical Findings
   ↓
Engineered Features
   ↓
ML Models
   ↓
Evaluation Results
   ↓
Operational Insights
   ↓
Reports / BI / Portfolio
```

Potential presentation outputs include:

* Analytical figures
* Statistical results
* Business-analysis summaries
* Classification evaluation results
* Airport cluster profiles
* Power BI dashboards
* Final project report

---

# Design Philosophy

This project follows several principles:

### Modular

Each stage has a clearly defined responsibility.

### Reproducible

The workflow uses controlled randomness, documented methodology, and explicit project configuration.

### Domain-Aware

Data transformations are based on the operational meaning of airline variables.

### ML With Purpose

Machine learning is used where it provides a meaningful extension to the operational analysis.

### Leakage-Aware

Training, development, and evaluation data are separated to reduce methodological contamination.

### Interpretable

Machine-learning outputs are analyzed alongside statistical and business findings rather than treated as isolated scores.

### Portfolio-Oriented

The project is structured as a complete software and data-science system rather than a single exploratory notebook.

---

# Status

**Project:** Completed core implementation and end-to-end workflow

**Primary focus:** Airline operations, data science, and machine learning

**Dataset:** U.S. Bureau of Transportation Statistics (BTS)

**Architecture:** Modular Python package

**Machine Learning:** Supervised + Unsupervised

**Documentation:** Architecture + Data Dictionary + Methodology

**Business Intelligence:** Power BI

---

# Author

**Omar Mostafa**

Computer Science Student | Aspiring Machine Learning Engineer

---

## License

This project is provided for educational and portfolio purposes.

The underlying airline data is sourced from the U.S. Bureau of Transportation Statistics and remains subject to its applicable terms and conditions.
