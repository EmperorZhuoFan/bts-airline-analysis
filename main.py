from airline.data.loading import load_data

from airline.data.validation import (
    validate_required_columns,
    validate_data_types,
    validate_missing_values,
    validate_duplicates,
    validate_domain_rules
)

from airline.data.preparation import prepare_data

from airline.features.engineering import engineer_features

from airline.analysis.exploratory_analysis import (
    analyze_flight_volume,
    analyze_delays,
    analyze_cancellations,
    analyze_airplorts
)

from airline.analysis.statistical_analysis import (
    summarize_delay_statistical,
    test_delay_difference,
    test_delay_by_day_of_week,
    analyze_delay_correlations
)

from airline.analysis.business_analysis import (
    analyze_operational_reliability,
    analyze_delay_propagation,
    analyze_operational_exposure,
    analyze_cancellation_exposure
)

from airline.models.supervised import run_supervised_learning

from airline.models.unsupervised import run_unsupervised_learning

from airline.models.evaluation import run_evaluation


def main() -> None:
    """Orchestrate the complete Airline BTS data science and machine learning workflow."""

    file_path = r"F:\October Plan Phases 1-7\__Projects__\U.S. Bureau of Transportation Statistics\U.S. Bureau of Transportation poject\AIRLINE_BTS.csv"

    # ============================================================
    # 1. LOAD
    # ============================================================

    df = load_data(file_path)

    initial_column_count = df.shape[1]

    # ============================================================
    # 2. VALIDATION
    # ============================================================

    required_columns = [
        "YEAR",
        "QUARTER",
        "MONTH",
        "DAY_OF_WEEK",
        "FL_DATE",
        "OP_UNIQUE_CARRIER",
        "TAIL_NUM",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "ORIGIN_STATE_ABR",
        "DEST",
        "DEST_CITY_NAME",
        "DEST_STATE_ABR",
        "CRS_DEP_TIME",
        "DEP_TIME",
        "DEP_DELAY",
        "DEP_DELAY_NEW",
        "TAXI_OUT",
        "TAXI_IN",
        "CRS_ARR_TIME",
        "ARR_TIME",
        "ARR_DELAY",
        "ARR_DELAY_NEW",
        "ARR_DEL15",
        "CANCELLED",
        "CANCELLATION_CODE",
        "DIVERTED",
        "ACTUAL_ELAPSED_TIME",
        "AIR_TIME",
        "DISTANCE",
        "CARRIER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY"
    ]

    validate_required_columns(df, required_columns)

    missing_report = validate_missing_values(df)

    duplicate_count = validate_duplicates(df)

    invalid_columns = validate_domain_rules(df)

    # ============================================================
    # 3. PREPARATION
    # ============================================================

    df = prepare_data(df)

    # ============================================================
    # 4. FEATURE ENGINEERING
    # ============================================================

    df = engineer_features(df)

    final_column_count = df.shape[1]

    # ============================================================
    # 5. EXPLORATORY ANALYSIS
    # ============================================================

    analyze_flight_volume(df)

    analyze_delays(df)

    analyze_cancellations(df)

    analyze_airplorts(df)

    # ============================================================
    # 6. STATISTICAL ANALYSIS
    # ============================================================

    summarize_delay_statistical(df)

    test_delay_difference(df)

    test_delay_by_day_of_week(df)

    analyze_delay_correlations(df)

    # ============================================================
    # 7. BUSINESS ANALYSIS
    # ============================================================

    analyze_operational_reliability(df)

    analyze_delay_propagation(df)

    analyze_operational_exposure(df)

    analyze_cancellation_exposure(df)

    # ============================================================
    # 8. SUPERVISED LEARNING
    # ============================================================

    tuned_models, cv_results, x_test, y_test = run_supervised_learning(df)

    # ============================================================
    # 9. UNSUPERVISED LEARNING
    # ============================================================

    (
        clustered_data,
        cluster_profile,
        scaled_data,
        kmeans
    ) = run_unsupervised_learning(df)

    # ============================================================
    # 10. EVALUATION
    # ============================================================

    evaluation_results, best_model, cluster_score = run_evaluation(
        tuned_models,
        x_test,
        y_test,
        scaled_data,
        kmeans
    )

    # ============================================================
    # 11. FINAL RESULTS
    # ============================================================

    print("=" * 70)
    print("SUPERVISED MODEL EVALUATION")
    print("=" * 70)
    print(evaluation_results)

    print("=" * 70)
    print("SELECTED BEST MODEL")
    print("=" * 70)
    print(best_model)

    print("=" * 70)
    print(f"CLUSTERING SILHOUETTE SCORE: {cluster_score:.4f}")

    print("=" * 70)
    print("CLUSTER PROFILES")
    print("=" * 70)
    print(cluster_profile)

    print("=" * 70)
    print("DATASET COLUMN SUMMARY")
    print("=" * 70)
    print(f"Initial columns: {initial_column_count}")
    print(f"Final columns:   {final_column_count}")
    print(f"Columns added:   {final_column_count - initial_column_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()