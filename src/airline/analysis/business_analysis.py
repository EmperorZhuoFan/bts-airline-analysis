import pandas as pd 
from airline.data.preparation import handle_missing_values 


def analyze_operational_reliability(df: pd.DataFrame) -> None:
    """Assess overall operational reliability using key flight performance indicators."""

    delayed_flights = df[df["DEP_DELAY"] > 0]
    cancelled_flights = df[df["CANCELLED"] == 1]
    diverted_flights = df[df["DIVERTED"] == 1]
    on_time_flights = df[df["DEP_DELAY"] <= 0]

    print("=" * 70 )
    print("Operational Reliability Analysis:")
    print("=" * 70 )
    print(f"Delayed flights:   {len(delayed_flights)} ({round(len(delayed_flights) / len(df) * 100, 2)}%)")
    print(f"Cancelled flights: {len(cancelled_flights)}  ({round(len(cancelled_flights) / len(df) * 100, 2)}%)")
    print(f"Diverted flights:  {len(diverted_flights)}   ({round(len(diverted_flights) / len(df) * 100, 2)}%)")
    print(f"On-Time flights:   {len(on_time_flights)} ({round(len(on_time_flights) / len(df) * 100, 2)}%)")
    print("=" * 70 )


def analyze_delay_propagation(df: pd.DataFrame) -> None:
    """Analyze how departure delays carry over into arrival delays."""

    departure_delays = df[df["DEP_DELAY"] > 0]

    remained_delayed = (departure_delays["ARR_DELAY"] > 0).sum()
    recovered = (departure_delays["ARR_DELAY"] <= 0).sum()

    volume = len(departure_delays)

    print("=" * 70)
    print("Delay Propagation Analysis:")
    print("=" * 70)
    print(f"Total delayed: {len(departure_delays)} ({round((len(departure_delays) / len(df) * 100), 2)}%)")
    print(f"Remained delayed: {remained_delayed} ({round((remained_delayed/ volume * 100), 2)}%)")
    print(f"Recovered: {recovered} ({round((recovered/ volume * 100), 2)}%)")
    print("=" * 70)


def analyze_operational_exposure(df: pd.DataFrame) -> None:
    """Identify areas with the greatest operational exposure based on flight volume and reliability problems."""

    operational_exposure = df.groupby("ORIGIN").agg(
                                                flight_volume= ("ORIGIN", "size"),
                                                delayed_flights= ("DEP_DELAY", lambda x: (x > 0).sum()), 
                                                cancelled_flights= ("CANCELLED", lambda x: (x == 1).sum()))

    operational_exposure["delay_rate"] = (operational_exposure["delayed_flights"] /
                                           operational_exposure["flight_volume"] * 100).round(2).astype(str) + "%"

    operational_exposure["cancelled_rate"] = (operational_exposure["cancelled_flights"] /
                                               operational_exposure["flight_volume"] * 100).round(2).astype(str) + "%"
    
    operational_exposure = operational_exposure.sort_values("flight_volume", ascending=False)

    print("-=-" * 70)
    print("Operational Exposure Analysis:")
    print("=" * 70)
    print(operational_exposure)
    print("=" * 70)


def analyze_cancellation_exposure(df: pd.DataFrame) -> None:
    """Identify areas with the greatest cancellation exposure based on cancellation volume and rates."""

    df = handle_missing_values(df)

    cancellation_exposure = df.groupby("ORIGIN").agg(flight_volume = ("ORIGIN", "size"), 
                                                                            cancelled_flights = ("CANCELLED", lambda x: (x == 1).sum()))
    
    cancellation_exposure["cancellation_rate"] = (cancellation_exposure["cancelled_flights"] /  cancellation_exposure["flight_volume"] * 100).round(2).astype(str) + "%"
    cancellation_exposure = cancellation_exposure.sort_values("cancelled_flights", ascending=False)

    cancellation_reason = df[df["CANCELLED"] == 1].groupby(["ORIGIN", "CANCELLATION_REASON"]).size()

    print("=" * 70)
    print("Cancellation Exposure Analysis:")
    print("=" * 70)
    print(f"Cancellation exposure: \n{cancellation_exposure}")
    print("=" * 70)
    print(f"Cancellation reason: \n{cancellation_reason}")
    print("=" * 70)