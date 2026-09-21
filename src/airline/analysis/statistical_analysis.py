import pandas as pd 
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import pearsonr


def summarize_delay_statistical(df: pd.DataFrame) -> None:
    """Summarize the statistical characteristics of departure and arrival delays."""

    departure_delays_info = df["DEP_DELAY"].describe()
    arrival_delays_info = df["ARR_DELAY"].describe()

    print("=" * 70)
    print("Delay Statistical Analysis:")
    print("=" * 70)

    print(f"Departure delay information: \n{departure_delays_info}")
    print("=" * 70)

    print(f"Arrival delay information: \n{arrival_delays_info}")
    print("=" * 70)



def test_delay_difference(df: pd.DataFrame) -> None:
    """Test whether departure delays differ significantly between weekdays and weekends."""

    weekdays = df[df["DAY_OF_WEEK"].between(1, 5)]
    weekends = df[df["DAY_OF_WEEK"].between(6, 7)]

    weekdays_delay = weekdays["DEP_DELAY"]
    weekends_delay = weekends["DEP_DELAY"]

    weekdays_delay = weekdays_delay.dropna()
    weekends_delay = weekends_delay.dropna()

    test_stats, p_value = ttest_ind(weekdays_delay, weekends_delay, equal_var=False)
    alpha = 0.05

    if p_value < alpha:
        conclusion = "Reject H₀"
    else:
        conclusion = "Fail to reject H₀"

    print("=" * 70)
    print("Weekdays Vs. Weekends Delay Test:")
    print("=" * 70)
    print(f"Weekdays flights: {len(weekdays_delay)}")
    print(f"Weekends flights: {len(weekends_delay)}")
    print(f"Test statistics: {test_stats:.3f}")
    print(f"P-value: {p_value:.3e}")
    print(f"Significance level: {alpha}")
    print(f"Conclusion: {conclusion}")
    print("=" * 70)



def test_delay_by_day_of_week(df: pd.DataFrame) -> None:
    """Test whether departure delays differ significantly across days of the week."""

    day_1 = df[df["DAY_OF_WEEK"] == 1]["DEP_DELAY"]
    day_2 = df[df["DAY_OF_WEEK"] == 2]["DEP_DELAY"]
    day_3 = df[df["DAY_OF_WEEK"] == 3]["DEP_DELAY"]
    day_4 = df[df["DAY_OF_WEEK"] == 4]["DEP_DELAY"]
    day_5 = df[df["DAY_OF_WEEK"] == 5]["DEP_DELAY"]
    day_6 = df[df["DAY_OF_WEEK"] == 6]["DEP_DELAY"]
    day_7 = df[df["DAY_OF_WEEK"] == 7]["DEP_DELAY"]

    day_1 = day_1.dropna()
    day_2 = day_2.dropna()
    day_3 = day_3.dropna()
    day_4 = day_4.dropna()
    day_5 = day_5.dropna()
    day_6 = day_6.dropna()
    day_7 = day_7.dropna()

    test_statistics, p_value = f_oneway(day_1, day_2, day_3, day_4, day_5, day_6, day_7)
    alpha = 0.05

    if p_value < alpha:
        conclusion="Reject H₀"
    else:
        conclusion="Fail to reject H₀"

    print("=" * 70)
    print("Departure Delay By Day Of Week Test:")
    print("=" * 70)
    print(f"F-statistics: {test_statistics:.3f}")
    print(f"P-value: {p_value:.3e}")
    print(f"Significance level: {alpha}")
    print(f"Conclusion: {conclusion}")
    print("=" * 70)    



def analyze_delay_correlations(df: pd.DataFrame) -> None:
    """Analyze the strength and significance of relationships between flight variables."""

    results = []
    alpha = 0.05
    named_pairs = [("departure delay Vs. Arrival delay", df["DEP_DELAY"], df["ARR_DELAY"]), 
                   ("departure delay Vs. Taxi out", df["DEP_DELAY"], df["TAXI_OUT"]), 
                   ("departure delay Vs Air time", df["DEP_DELAY"], df["AIR_TIME"]), 
                   ("Distance Vs. Air time", df["DISTANCE"], df["AIR_TIME"])]


    for name , x, y in named_pairs:
        data = pd.concat([x, y], axis=1).dropna()
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]

        r, p_value = pearsonr(x, y)

        if p_value < alpha:
            conclusion="Reject H₀"
        else:
            conclusion="Fail to reject H₀"

        results.append({"Relation": name, "Correlation": r, "P-value":"< 1e-15" if p_value == 0 else f"{p_value:.4e}",
                         "Significance level": alpha, "Status": conclusion})

    results = pd.DataFrame(results)
    print("=" * 70)
    print("Delay Correlation Analysis:")
    print("=" * 70)

    print(results)
    print("=" * 70)