# U.S. Airline Performance Analysis (BTS Data)

An exploratory data analysis of U.S. airline flights based on the
**Reporting Carrier On-Time Performance** dataset published by the
[U.S. Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov).

The project investigates flight delays, cancellations, and route-level
performance, and includes an interactive Power BI dashboard.

## What the analysis covers

- **Data understanding** — shape, dtypes, summary statistics, missing-value
  report, and duplicate detection
- **Data cleaning** — datetime parsing, conversion of HHMM times to hours,
  missing-value investigation (including the link between missing delay
  fields and cancellations), and IQR-based outlier detection
- **Feature engineering** — date parts (year, quarter, month, day of week),
  scheduled departure/arrival hours, delay severity categories
  (Early/On Time, Minor, Moderate, Severe), taxi and flight duration
  metrics, and origin-destination route labels
- **Exploratory data analysis** — delay baselines and distributions using
  pandas, matplotlib, and seaborn
- **Modeling toolkit** — scikit-learn classifiers and regressors (decision
  trees, random forests, gradient boosting, AdaBoost, bagging) prepared for
  delay prediction experiments

## Project structure

```
.
├── airline_bts.py        # Main analysis script (runs end to end)
├── BTS_Dashboard.pbix    # Power BI dashboard
├── data/
│   └── AIRLINE_BTS.csv   # Flight dataset (not included, see below)
├── requirements.txt      # Python dependencies
└── README.md
```

## Getting the dataset

The dataset is too large to store in this repository (~126 MB, above
GitHub's per-file limit). To run the analysis:

1. Go to [BTS TranStats](https://www.transtats.bts.gov) and open the
   **Reporting Carrier On-Time Performance (1987 – present)** table.
2. Download the desired time period as a CSV.
3. Save it as `data/AIRLINE_BTS.csv` in this project.

## Setup

```bash
pip install -r requirements.txt
python airline_bts.py
```

## Power BI dashboard

Open `BTS_Dashboard.pbix` in
[Power BI Desktop](https://powerbi.microsoft.com/desktop/). If the dashboard
prompts for a data source, point it to your local copy of `AIRLINE_BTS.csv`.

## Data source

U.S. Bureau of Transportation Statistics — Reporting Carrier On-Time
Performance data, containing per-flight records for departure/arrival times,
delays and their causes (carrier, NAS, security, late aircraft), and
cancellations.
