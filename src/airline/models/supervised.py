# import pandas as pd
# import time 

# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     BaggingClassifier,
#     AdaBoostClassifier,
#     GradientBoostingClassifier
# )


# def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
#     """Prepare engineered data for supervised learning by separating features from the target."""

#     df = df.copy()

#     features = [
#         "YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK",
#         "CRS_DEP_TIME", "CRS_ARR_TIME", "OP_UNIQUE_CARRIER",
#         "TAIL_NUM", "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR",
#         "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "DISTANCE",
#         "DEP_HOUR", "TIME_OF_DAY", "ROUTE", "DISTANCE_CATEGORY"
#     ]

#     target = "IS_DEP_DELAYED"

#     x = df[features]
#     y = df[target]

#     return x, y


# def create_preprocessor() -> ColumnTransformer:
#     """Create the preprocessing pipeline for numerical and categorical features."""

#     numerical_features = [
#         "YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK",
#         "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE", "DEP_HOUR"
#     ]

#     categorical_features = [
#         "OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN",
#         "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "DEST",
#         "DEST_CITY_NAME", "DEST_STATE_ABR", "TIME_OF_DAY",
#         "ROUTE", "DISTANCE_CATEGORY"
#     ]

#     numerical_transformers = Pipeline([
#         ("scaler", StandardScaler())
#     ])

#     categorical_transformers = Pipeline([
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ])

#     preprocessor = ColumnTransformer([
#         ("cat", categorical_transformers, categorical_features),
#         ("num", numerical_transformers, numerical_features)
#     ])

#     return preprocessor


# def split_data(
#     x: pd.DataFrame,
#     y: pd.Series
# ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     """Split features and target into training and testing datasets."""

#     x_train, x_test, y_train, y_test = train_test_split(
#         x,
#         y,
#         random_state=42,
#         test_size=0.2,
#         stratify=y
#     )

#     return x_train, x_test, y_train, y_test


# def train_baseline(
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> Pipeline:
#     """Train a baseline classifier to establish a reference performance."""

#     baseline_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", DummyClassifier(strategy="most_frequent"))
#     ])

#     baseline_model.fit(x_train, y_train)

#     return baseline_model


# def train_logistic_regression(
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> Pipeline:
#     """Train a logistic regression classifier using the prepared training data."""

#     logistic_regression_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", LogisticRegression(
#             max_iter=1000,
#             random_state=42
#         ))
#     ])

#     logistic_regression_model.fit(x_train, y_train)

#     return logistic_regression_model


# def train_decision_tree(
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> Pipeline:
#     """Train a decision tree classifier using the prepared training data."""

#     decision_tree_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", DecisionTreeClassifier(
#             max_depth=10,
#             random_state=42
#         ))
#     ])

#     decision_tree_model.fit(x_train, y_train)

#     return decision_tree_model


# def train_bagging(
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> Pipeline:
#     """Train a bagging classifier using the prepared training data."""

#     bagging_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", BaggingClassifier(
#             n_estimators=10,
#             max_samples=0.5,
#             n_jobs=-1,
#             random_state=42
#         ))
#     ])

#     bagging_model.fit(x_train, y_train)

#     return bagging_model


# def train_random_forest(
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> Pipeline:
#     """Train a random forest classifier using the prepared training data."""

#     random_forest_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", RandomForestClassifier(
#             n_estimators=50,
#             max_depth=10,
#             max_samples=0.7,
#             n_jobs=-1,
#             random_state=42
#         ))
#     ])

#     random_forest_model.fit(x_train, y_train)

#     return random_forest_model


# def train_boosting(
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> tuple[Pipeline, Pipeline]:
#     """Train AdaBoost and Gradient Boosting classifiers using the prepared training data."""

#     adaboosting_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", AdaBoostClassifier(
#             n_estimators=50,
#             learning_rate=0.1,
#             random_state=42
#         ))
#     ])

#     gradboosting_model = Pipeline([
#         ("preprocessor", create_preprocessor()),
#         ("model", GradientBoostingClassifier(
#             n_estimators=50,
#             learning_rate=0.1,
#             max_depth=3,
#             random_state=42
#         ))
#     ])

#     adaboosting_model.fit(x_train, y_train)
#     gradboosting_model.fit(x_train, y_train)

#     return adaboosting_model, gradboosting_model


# def build_model_registry(
#     baseline_model: Pipeline,
#     logistic_regression_model: Pipeline,
#     decision_tree_model: Pipeline,
#     bagging_model: Pipeline,
#     random_forest_model: Pipeline,
#     adaboosting_model: Pipeline,
#     gradboosting_model: Pipeline
# ) -> dict[str, Pipeline]:
#     """Build a registry containing all trained supervised learning models."""

#     models = {
#         "Baseline model": baseline_model,
#         "Logistic regression": logistic_regression_model,
#         "Decision tree": decision_tree_model,
#         "Bagging model": bagging_model,
#         "Random forest": random_forest_model,
#         "AdaBoosting": adaboosting_model,
#         "Gradient boosting": gradboosting_model
#     }

#     return models


# def cross_validate_model(
#     models: dict[str, Pipeline],
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> pd.DataFrame:
#     """Evaluate all supervised models using cross-validation."""

#     cv_results = []

#     for name, model in models.items():

#         score = cross_val_score(
#             model,
#             x_train,
#             y_train,
#             cv=3,
#             scoring="f1",
#             n_jobs=-1
#         )

#         cv_results.append({
#             "Model": name,
#             "Mean F1": score.mean()
#         })

#     cv_results = pd.DataFrame(cv_results).sort_values(
#         "Mean F1",
#         ascending=False
#     )

#     return cv_results


# def tune_models(
#     models: dict[str, Pipeline],
#     x_train: pd.DataFrame,
#     y_train: pd.Series
# ) -> dict[str, Pipeline]:
#     """Tune supervised models using cross-validated grid search."""

#     tuned_models = {}

#     parameter_grids = {
#         "Logistic regression": {
#             "model__C": [0.01, 0.1, 1, 10],
#             "model__solver": ["lbfgs", "liblinear"],
#             "model__max_iter": [500, 1000]
#         },

#         "Decision tree": {
#             "model__max_depth": [5, 10, 20, None],
#             "model__min_samples_split": [2, 5, 10],
#             "model__min_samples_leaf": [1, 5]
#         },

#         "Bagging model": {
#             "model__n_estimators": [10, 25],
#             "model__max_samples": [0.5, 0.75]
#         },

#         "Random forest": {
#             "model__n_estimators": [50, 100],
#             "model__max_depth": [10, 20, None],
#             "model__min_samples_split": [2, 5],
#             "model__min_samples_leaf": [1, 5]
#         },

#         "AdaBoosting": {
#             "model__n_estimators": [50, 100],
#             "model__learning_rate": [0.01, 0.1]
#         },

#         "Gradient boosting": {
#             "model__n_estimators": [50, 100, 200],
#             "model__learning_rate": [0.01, 0.1, 0.2],
#             "model__max_depth": [3, 5]
#         }
#     }

#     for model_name, model in models.items():

#         if model_name == "Baseline model":
#             continue

#         param_grid = parameter_grids[model_name]

#         grid_search = GridSearchCV(
#             estimator=model,
#             param_grid=param_grid,
#             cv=3,
#             scoring="f1",
#             n_jobs=-1
#         )

#         grid_search.fit(x_train, y_train)

#         best_model = grid_search.best_estimator_

#         tuned_models[model_name] = best_model

#     return tuned_models

# def run_supervised_learning(
#     df: pd.DataFrame
# ) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series]:
#     """Run the complete supervised learning workflow."""

#     start_time = time.time()

#     print("=" * 70)
#     print("SUPERVISED LEARNING")
#     print("=" * 70)

#     print("Preparing model data...")
#     x, y = prepare_model_data(df)
#     print(f"Completed in {time.time() - start_time:.2f} seconds")

#     print("Splitting data...")
#     x_train, x_test, y_train, y_test = split_data(x, y)
#     print(f"Completed in {time.time() - start_time:.2f} seconds")

#     max_training_rows = 10_000

#     if len(x_train) > max_training_rows:
#         x_train, _, y_train, _ = train_test_split(
#             x_train,
#             y_train,
#             train_size=max_training_rows,
#             random_state=42,
#             stratify=y_train
#         )

#     print(f"Training rows used: {len(x_train):,}")

#     print("Training baseline...")
#     baseline_model = train_baseline(x_train, y_train)
#     print(f"Baseline completed in {time.time() - start_time:.2f} seconds")

#     print("Training logistic regression...")
#     logistic_regression_model = train_logistic_regression(x_train, y_train)
#     print(f"Logistic regression completed in {time.time() - start_time:.2f} seconds")

#     print("Training decision tree...")
#     decision_tree_model = train_decision_tree(x_train, y_train)
#     print(f"Decision tree completed in {time.time() - start_time:.2f} seconds")

#     print("Training bagging...")
#     bagging_model = train_bagging(x_train, y_train)
#     print(f"Bagging completed in {time.time() - start_time:.2f} seconds")

#     print("Training random forest...")
#     random_forest_model = train_random_forest(x_train, y_train)
#     print(f"Random forest completed in {time.time() - start_time:.2f} seconds")

#     print("Training boosting models...")
#     adaboosting_model, gradboosting_model = train_boosting(
#         x_train,
#         y_train
#     )
#     print(f"Boosting completed in {time.time() - start_time:.2f} seconds")

#     models = build_model_registry(
#         baseline_model,
#         logistic_regression_model,
#         decision_tree_model,
#         bagging_model,
#         random_forest_model,
#         adaboosting_model,
#         gradboosting_model
#     )

#     print("Cross-validating models...")
#     cv_results = cross_validate_model(
#         models,
#         x_train,
#         y_train
#     )
#     print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")

#     print("Tuning models...")
#     tuned_models = tune_models(
#         models,
#         x_train,
#         y_train
#     )
#     print(f"Model tuning completed in {time.time() - start_time:.2f} seconds")

#     print("=" * 70)
#     print(f"TOTAL SUPERVISED LEARNING TIME: {time.time() - start_time:.2f} seconds")
#     print("=" * 70)

#     return tuned_models, cv_results, x_test, y_test





import pandas as pd
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target data for supervised machine learning."""

    features = [
        "YEAR",
        "QUARTER",
        "MONTH",
        "DAY_OF_WEEK",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "OP_UNIQUE_CARRIER",
        "TAIL_NUM",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "ORIGIN_STATE_ABR",
        "DEST",
        "DEST_CITY_NAME",
        "DEST_STATE_ABR",
        "DISTANCE",
        "DEP_HOUR",
        "TIME_OF_DAY",
        "ROUTE",
        "DISTANCE_CATEGORY"
    ]

    target = "IS_DEP_DELAYED"

    missing_features = []

    for feature in features:
        if feature not in df.columns:
            missing_features.append(feature)

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    if missing_features:
        raise ValueError(
            f"Required model features are missing: {', '.join(missing_features)}"
        )

    model_data = df[features + [target]].copy()
    model_data = model_data.dropna(subset=[target])

    x = model_data[features]
    y = model_data[target]

    return x, y


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline for numerical and categorical features."""

    numerical_features = [
        "YEAR",
        "QUARTER",
        "MONTH",
        "DAY_OF_WEEK",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "DISTANCE",
        "DEP_HOUR"
    ]

    categorical_features = [
        "OP_UNIQUE_CARRIER",
        "TAIL_NUM",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "ORIGIN_STATE_ABR",
        "DEST",
        "DEST_CITY_NAME",
        "DEST_STATE_ABR",
        "TIME_OF_DAY",
        "ROUTE",
        "DISTANCE_CATEGORY"
    ]

    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def split_data(
    x: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into stratified training and testing sets."""

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return x_train, x_test, y_train, y_test


def train_baseline() -> Pipeline:
    """Build the baseline classification pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent"
                )
            )
        ]
    )

    return pipeline


def train_logistic_regression() -> Pipeline:
    """Build the logistic regression classification pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    return pipeline


def train_decision_tree() -> Pipeline:
    """Build the decision tree classification pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42
                )
            )
        ]
    )

    return pipeline


def train_bagging() -> Pipeline:
    """Build the bagging classification pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                BaggingClassifier(
                    n_estimators=10,
                    max_samples=0.5,
                    n_jobs=1,
                    random_state=42
                )
            )
        ]
    )

    return pipeline


def train_random_forest() -> Pipeline:
    """Build the random forest classification pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    max_samples=0.7,
                    n_jobs=1,
                    random_state=42
                )
            )
        ]
    )

    return pipeline


def train_boosting() -> dict[str, Pipeline]:
    """Build the AdaBoost and Gradient Boosting classification pipelines."""

    boosting_models = {}

    boosting_models["AdaBoosting"] = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                AdaBoostClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    random_state=42
                )
            )
        ]
    )

    boosting_models["Gradient boosting"] = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )

    return boosting_models


def build_model_registry() -> dict[str, Pipeline]:
    """Build and register all supervised learning pipelines without fitting them."""

    models = {}

    models["Logistic regression"] = train_logistic_regression()
    models["Decision tree"] = train_decision_tree()
    models["Bagging model"] = train_bagging()
    models["Random forest"] = train_random_forest()

    boosting_models = train_boosting()

    for model_name, model in boosting_models.items():
        models[model_name] = model

    return models


def tune_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Tune supervised models using randomized hyperparameter search."""

    max_development_rows = 25_000

    if len(x_train) > max_development_rows:
        x_development, _, y_development, _ = train_test_split(
            x_train,
            y_train,
            train_size=max_development_rows,
            random_state=42,
            stratify=y_train
        )
    else:
        x_development = x_train
        y_development = y_train

    parameter_distributions = {
        "Logistic regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [500, 1000]
        },

        "Decision tree": {
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 5]
        },

        "Bagging model": {
            "model__n_estimators": [10, 25],
            "model__max_samples": [0.5, 0.75]
        },

        "Random forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 5]
        },

        "AdaBoosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1]
        },

        "Gradient boosting": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5]
        }
    }

    search_iterations = {
        "Logistic regression": 6,
        "Decision tree": 6,
        "Bagging model": 2,
        "Random forest": 3,
        "AdaBoosting": 3,
        "Gradient boosting": 4
    }

    tuned_models = {}
    search_results = []

    for model_name, model in models.items():

        print("=" * 70)
        print(f"Tuning: {model_name}")
        print("=" * 70)

        start_time = time.time()

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_distributions[model_name],
            n_iter=search_iterations[model_name],
            scoring="f1",
            cv=3,
            random_state=42,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            pre_dispatch="2*n_jobs"
        )

        search.fit(x_development, y_development)

        elapsed_time = time.time() - start_time

        tuned_models[model_name] = search.best_estimator_

        search_results.append(
            {
                "Model": model_name,
                "Mean CV F1": search.best_score_,
                "Best Parameters": search.best_params_,
                "Search Time": round(elapsed_time, 2)
            }
        )

        print(f"Best CV F1: {search.best_score_:.4f}")
        print(f"Search completed in {elapsed_time:.2f} seconds")

    cv_results = pd.DataFrame(search_results)

    return tuned_models, cv_results


def run_supervised_learning(
    df: pd.DataFrame
) -> tuple[
    dict[str, Pipeline],
    pd.DataFrame,
    pd.DataFrame,
    pd.Series
]:
    """Run the complete supervised learning workflow."""

    print("=" * 70)
    print("SUPERVISED LEARNING")
    print("=" * 70)

    start_time = time.time()

    print("Preparing model data...")

    x, y = prepare_model_data(df)

    print("Completed")

    print("Splitting data...")

    x_train, x_test, y_train, y_test = split_data(x, y)

    print(f"Training rows: {len(x_train):,}")
    print(f"Testing rows: {len(x_test):,}")
    print("Completed")

    print("Training baseline...")

    baseline = train_baseline()
    baseline.fit(x_train, y_train)

    print("Baseline completed")

    print("Building model registry...")

    models = build_model_registry()

    print("Model registry completed")

    print("Tuning models...")

    tuned_models, cv_results = tune_models(
        models,
        x_train,
        y_train
    )

    print("Model tuning completed")

    total_time = time.time() - start_time

    print("=" * 70)
    print(f"TOTAL SUPERVISED LEARNING TIME: {total_time:.2f} seconds")
    print("=" * 70)

    return tuned_models, cv_results, x_test, y_test