import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score


def prepare_model_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare supervised learning data by separating selected features from the target."""

    df = df.copy()

    features = [
        "YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK",
        "CRS_DEP_TIME", "CRS_ARR_TIME",
        "OP_UNIQUE_CARRIER",
        "ORIGIN", "DEST",
        "DISTANCE",
        "DEP_HOUR",
        "TIME_OF_DAY",
        "DISTANCE_CATEGORY"
    ]

    target = "IS_DEP_DELAYED"

    x = df[features]
    y = df[target]

    return x, y


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numerical and categorical supervised-learning features."""

    numerical_features = [
        "YEAR", "QUARTER", "MONTH", "DAY_OF_WEEK",
        "CRS_DEP_TIME", "CRS_ARR_TIME",
        "DISTANCE", "DEP_HOUR"
    ]

    categorical_features = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "TIME_OF_DAY",
        "DISTANCE_CATEGORY"
    ]

    numerical_transformers = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformers = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_transformers, categorical_features),
        ("num", numerical_transformers, numerical_features)
    ])

    return preprocessor


def split_data(
    x: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split supervised learning data into stratified training and testing sets."""

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        random_state=42,
        test_size=0.2,
        stratify=y
    )

    return x_train, x_test, y_train, y_test


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> dict[str, Pipeline]:
    """Train the selected supervised learning models and return them in a registry."""

    models = {
        "Baseline model": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "Logistic regression": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Decision tree": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ))
        ]),

        "AdaBoosting": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", AdaBoostClassifier(
                random_state=42
            ))
        ])
    }

    for name, model in models.items():
        print("=" * 70)
        print(f"Training: {name}")
        print("=" * 70)

        model.fit(x_train, y_train)

        print(f"{name}: completed")

    return models


def compare_models(
    models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Compare trained supervised models using test-set F1 score."""

    results = []

    for name, model in models.items():
        predictions = model.predict(x_test)

        score = f1_score(y_test, predictions)

        results.append({
            "Model": name,
            "F1 score": score
        })

    results = pd.DataFrame(results).sort_values(
        "F1 score",
        ascending=False
    )

    return results


def run_supervised_test(
    df: pd.DataFrame
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run a lightweight supervised-learning test on the BTS dataset."""

    x, y = prepare_model_data(df)

    x_train, x_test, y_train, y_test = split_data(x, y)

    models = train_models(x_train, y_train)

    results = compare_models(
        models,
        x_test,
        y_test
    )

    print("=" * 70)
    print("SUPERVISED TEST RESULTS")
    print("=" * 70)
    print(results)
    print("=" * 70)

    return models, results, x_test, y_test