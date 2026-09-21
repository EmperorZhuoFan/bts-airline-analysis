import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from airline.data.preparation import *

plt.style.use("Solarize_Light2")


def analyze_flight_volume(df: pd.DataFrame) -> None:
    """Analyze flight volume and identify patterns over time."""

    total_flights = len(df)
    flight_volume_by_year = df["YEAR"].value_counts().sort_index()
    flight_volume_by_month = df["MONTH"].value_counts().sort_index()
    flight_volume_by_day = df["DAY_OF_WEEK"].value_counts().sort_index()

    print("=" * 70)
    print(f"Total flight volume: {total_flights:,}")
    print("=" * 70)

    print("Flight volume by year:")
    print(flight_volume_by_year)
    print("=" * 70)

    print("Flight volume by month:")
    print(flight_volume_by_month)
    print("=" * 70)

    print("Flight volume by day of week:")
    print(flight_volume_by_day)
    print("=" * 70)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=flight_volume_by_year.index, y=flight_volume_by_year.values)
    plt.title("Flight Volume by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=flight_volume_by_month.index, y=flight_volume_by_month.values)
    plt.title("Flight Volume by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=flight_volume_by_day.index, y=flight_volume_by_day.values)
    plt.title("Flight Volume by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    return None


def analyze_delays(df: pd.DataFrame) -> None:
    """Analyze flight delays and identify their frequency, severity, and patterns."""

    delay_frequency = df["DEP_DELAY"].gt(0).value_counts()
    delay_frequency.index = delay_frequency.index.map({False: "Not Delayed", True: "Delayed"})

    delay_pattern = (df.groupby("DAY_OF_WEEK")["DEP_DELAY"].apply(lambda x: (x > 0).mean() * 100).round(0))

    delay_severity = pd.cut(df["DEP_DELAY"],bins=[-float("inf"), 0, 15, 30, 60, 120, float("inf")],
                                    labels=["Early/On time", "1-15 min", "16-30 min", "31-60 min", "61-120 min", "+120 min"])

    delay_severity_count = delay_severity.value_counts().sort_index()

    print("=" * 70)
    print("DELAY ANALYSIS")
    print("=" * 70)

    print(f"Flights with arrival delays: {(df['ARR_DELAY'] > 0).sum():,}")
    print("=" * 70)

    print(f"Flights with departure delays: {(df['DEP_DELAY'] > 0).sum():,}")
    print("=" * 70)

    print("Departure delay frequency:")
    print(delay_frequency)
    print("=" * 70)

    print("Departure delay pattern by day of week:")
    print(delay_pattern.astype(str) + "%")
    print("=" * 70)

    print("Departure delay severity:")
    print(delay_severity_count)
    print("=" * 70)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=delay_frequency.index, y=delay_frequency.values)
    plt.title("How often do departure delays happen?")    
    plt.xlabel("Delay Status")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=delay_severity_count.index, y=delay_severity_count.values)
    plt.title("How severe are departure delays?")    
    plt.xlabel("Delay Duration")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=delay_pattern.index, y=delay_pattern.values)
    plt.title("How do departure delays vary by day of week?")    
    plt.xlabel("Day of Week")
    plt.ylabel("Delayed Flights (%)")
    plt.tight_layout()
    plt.show()

    return None


def analyze_cancellations(df: pd.DataFrame) -> None:
    """Analyze flight cancellations and identify their frequency, causes, and patterns."""

    cancellation_frequency = df["CANCELLED"].value_counts()
    cancellation_frequency.index = cancellation_frequency.index.map({0.0: "Not Cancelled", 1.0: "Cancelled"})

    cancellation_reasons = df.loc[df["CANCELLATION_REASON"] != "Not Cancelled", "CANCELLATION_REASON"].value_counts()
    cancellation_status = df["CANCELLED"].map({0.0: "Not Cancelled", 1.0: "Cancelled"})

    cancellation_pattern = df.groupby("DAY_OF_WEEK")["CANCELLED"].value_counts().unstack(fill_value=0)
    cancellation_pattern.columns = cancellation_pattern.columns.map({0.0: "Not Cancelled", 1.0: "Cancelled"})

    print("=" * 70)
    print("CANCELLATION ANALYSIS")
    print("=" * 70)

    print("Cancellation frequency:")
    print(cancellation_frequency)
    print("=" * 70)

    print("Cancellation reasons:")
    print(cancellation_reasons)
    print("=" * 70)

    print("Cancellation patterns by day of week:")
    print(cancellation_pattern)
    print("=" * 70)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=cancellation_frequency.index, y=cancellation_frequency.values)
    plt.title("How Many Flights Were Cancelled?")
    plt.xlabel("Cancellation Status")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cancellation_reasons.index, y=cancellation_reasons.values)
    plt.title("Why Were Flights Cancelled?")
    plt.xlabel("Cancellation Reason")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    cancellation_pattern.plot(kind="bar")
    plt.title("When Are Flights Cancelled?")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Flights")
    plt.tight_layout()
    plt.show()

    return None


def analyze_airplorts(df: pd.DataFrame) -> None:
    """Analyze airport flight volume and identify the busiest airports."""

    origin_summary = df["ORIGIN"].value_counts().head(10)

    destination_summary = df["DEST"].value_counts().head(10)


    print("=" * 70)
    print("Airports Analysis")
    print("=" * 70)

    print(f"Top 10 departure airports: \n{origin_summary}")
    print("=" * 70)

    print(f"Top 10 arrival airports: \n{destination_summary}")
    print("=" * 70)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=origin_summary.values, y=origin_summary.index)
    plt.title("Which Airports Handle the Most Departing Flights?")
    plt.xlabel("Number of Flights")
    plt.ylabel("Origin Airport")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=destination_summary.values, y=destination_summary.index)
    plt.title("Which Airports Handle the Most Arriving Flights?")
    plt.xlabel("Number of Flights")
    plt.ylabel("Destination Airport")
    plt.tight_layout()
    plt.show()

    return None