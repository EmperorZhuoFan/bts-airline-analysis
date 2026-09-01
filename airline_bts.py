# ================================================================
# STEP 1: IMPORTING DATA
# ================================================================

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from pathlib import Path

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score , f1_score
from sklearn.linear_model import LinearRegression , LogisticRegression # logistic regression is for classifiers 

DATA_PATH = Path(__file__).resolve().parent / "data" / "AIRLINE_BTS.csv"
df = pd.read_csv(DATA_PATH)
df_copy = df.copy()

# ================================================================
# STEP 2: DATA UNDERSTANDING
# ================================================================

print(" --- the first 5 rows ---")
print(df.head())
print("=" * 70)

print(" --- Rows and Columns ---")
print(f"Rows are {df.shape[0]} and Columns are {df.shape[1]}")
print("=" * 70)

print(" --- What columns in Dataset ---")
print(df.columns.tolist())
print("=" * 70)

print(" --- Dataset types  ---")
print(df.dtypes.to_string())
print("=" * 70)

print(" --- Summary Statistics  ---")
print(df.describe())
print("=" * 70)

print(" --- Missing Data  ---")
print(df.isnull().sum())
print("=" * 70)

print(" --- Missing Data Percentage  ---")
print(f"{((df.isnull().sum() / len(df)) * 100).round(2)}")
print("=" * 70)

missing = df.isnull().sum()
missing_report = pd.DataFrame({
    "Missing Values" : missing , 
    "Missing %" : ((missing / df.shape[0]) * 100).round(2)
})

print(missing_report)

print(" --- Duplicated Rows ---")
print(df.duplicated().sum())
print("=" * 70) # Zero is good, means no dups rows in the dataset

# ================================================================
# STEP 3: DATA CLEANING
# ================================================================

# ------------------------------------------------
# 3.1 DATE TIME DATA 
# ------------------------------------------------
date_time_data = df[["FL_DATE", "DEP_TIME", "DEP_DELAY", "ARR_TIME", "ARR_DELAY", "AIR_TIME"]]
print(date_time_data.head(10))

df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
print(df["FL_DATE"].head())
print(df["FL_DATE"].info())
print("=" * 70)

df["DEP_TIME_HRS"] = (df["DEP_TIME"] / 60).round(2)
df["ARR_TIME_HRS"] = (df["ARR_TIME"] / 60).round(2)
print(df[["DEP_TIME", "DEP_TIME_HRS", "ARR_TIME", "ARR_TIME_HRS"]].head())
print("=" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
print(" --- Numeric Columns ---")   
print(numeric_columns)
print("=" * 70)

# ------------------------------------------------
# 3.2 MISSING VALUES 
# ------------------------------------------------
missing = df.isnull().sum()
missing_report = pd.DataFrame({
    "Missing Values" : missing , 
    "Missing %" : (((missing / df.shape[0]) * 100)).round(1)})
print(missing_report)
print("=" * 70)


print(" --- Calculate Missing values ---")
low_missing = missing_report[
    (missing_report["Missing %"] > 0) &
    (missing_report["Missing %"] < 5)
]

print(low_missing)
print("=" * 70)

print(" --- Index of Columns with Less than 5% Missing Values ---")
index_less_5 = low_missing.index
print(index_less_5)
print("=" * 70)

print(df[index_less_5].head())
print("=" * 70 )

print(" --- Claryfing the Relation Between Missing less than 5 and Cancelation")
index_less_5_check = df.groupby("CANCELLED")[index_less_5].apply(lambda x: x.isnull().mean().round(3))
index_less_5_check.index = index_less_5_check.index.map({
    0.0: "Not Linked to cancellation",
    1.0: "Linked to cancellation"
})
print(index_less_5_check)
print("=" * 70)

# ------------------------------------------------
# 3.3 OUTLIERS DETECTION
# ------------------------------------------------
def detect_outliers(column):

    data = df[column].dropna()

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"\n--- {column} ---")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Outliers:", len(outliers))

    return outliers
delay_outliers = detect_outliers("DISTANCE")
print("="  * 70 )

# ================================================================
# STEP 4: FEATURE ENGINEERING
# ================================================================

# ------------------------------------------------
# 4.1 Date Time Features
# ------------------------------------------------

print(" --- Date Time Features ---")

df["YEAR"] = df["FL_DATE"].dt.year
df["QUARTER"] = df["FL_DATE"].dt.quarter
df["MONTH"] = df["FL_DATE"].dt.month
df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek

df["MONTH_NAME"] = df["FL_DATE"].dt.month_name()
df["DAY_NAME"] = df["FL_DATE"].dt.day_name()

datetime = df[["YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK", "DAY_NAME", "MONTH_NAME"]]
print(datetime.info())
print("=" * 70)

print(datetime.head(10))
print("=" * 70)

# ------------------------------------------------
# 4.2 Scheduled Time Features
# ------------------------------------------------

print(" --- Scheduled Time Features ---")

df["DEP_SCHEDULED_HOUR"] = df["CRS_DEP_TIME"] // 100
df["ARR_SCHEDULED_HOUR"] = df["CRS_ARR_TIME"] // 100

print(df[["CRS_DEP_TIME", "DEP_SCHEDULED_HOUR", "CRS_ARR_TIME","ARR_SCHEDULED_HOUR"]].head(10))
print("=" * 70)

# ------------------------------------------------
# 4.3 Delay Features
# ------------------------------------------------

print(" --- Delay Features ---")

delay_values = df[["ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "CARRIER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]]
print(delay_values.head(10))
print("=" * 70)

df["DELAY_CATEGORY"] = pd.cut(df["ARR_DELAY"], bins=[-float("inf"), 0, 15, 60, float("inf")], labels=["Early/On Time", "Minor", "Moderate", "Severe"])
print(df[["ARR_DELAY", "DELAY_CATEGORY"]].head(10))
print("=" * 70)

print(df.groupby("DELAY_CATEGORY")["ARR_DELAY"].count())
print("=" * 70)

# ------------------------------------------------
# 4.3 Flight Duration
# ------------------------------------------------
print(" --- Flight Duration ---")

df["TAXI_TOTAL"] = df["TAXI_IN"] + df["TAXI_OUT"]

print(df[["TAXI_IN", "TAXI_OUT", "TAXI_TOTAL", "AIR_TIME" , "ACTUAL_ELAPSED_TIME"]].head(10))
print("=" * 70)

print(df.groupby("CANCELLED")[["TAXI_TOTAL", "AIR_TIME", "ACTUAL_ELAPSED_TIME"]].mean())
print("=" * 70)

# ------------------------------------------------
# 4.4 Route Features
# ------------------------------------------------
print(" --- Route Features ---")

df["ROUTE"] =  df["ORIGIN"] + " ---> " + df["DEST"]

print(df[["ORIGIN", "DEST", "ROUTE"]].head(10))
print("=" * 70)

# ================================================================
# PHASE 5 — EXPLORATORY DATA ANALYSIS
# ================================================================

# ------------------------------------------------
# 5.1 Establish the Delay Baseline
# ------------------------------------------------
print(" --- Establish the Delay Baseline ---")

print(df[["ARR_DELAY", "ARR_DEL15"]].head())
print("=" * 70)
print(df[ "ARR_DEL15"].value_counts())