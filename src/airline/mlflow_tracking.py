
"""MLflow tracking for supervised models in the BTS airline project."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.validation import check_is_fitted


DEFAULT_EXPERIMENT = "BTS Airline ML"


def _project_tracking_uri() -> str:
    """Return the SQLite tracking URI for the database in the project root."""
    database_path = Path(__file__).resolve().parents[2] / "mlflow.db"
    return f"sqlite:///{quote(database_path.as_posix(), safe='/:')}"


def setup_tracking(
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> str:
    """Create or select the MLflow experiment."""
    mlflow.set_tracking_uri(_project_tracking_uri())
    experiment = mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment.name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    return experiment.experiment_id


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert parameter values to MLflow-compatible strings or scalars."""
    normalized = {}

    for key, value in params.items():
        if value is None:
            normalized[key] = "None"
        elif isinstance(value, (str, bool, int, float)):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(
                value, default=str, sort_keys=True
            )

    return normalized


def log_model_run(
    model_name: str,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, str | Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one fitted model and return its MLflow run ID."""

    if not model_name.strip():
        raise ValueError("model_name cannot be empty.")

    check_is_fitted(model)

    if not metrics:
        raise ValueError("At least one metric is required.")

    clean_metrics = {}
    for name, value in metrics.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Metric {name!r} must be finite.")
        clean_metrics[name] = value

    artifact_paths = {}
    for name, file_path in (artifacts or {}).items():
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {path}")
        artifact_paths[name] = path

    setup_tracking()

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(_normalize_params(params))
        mlflow.log_metrics(clean_metrics)

        mlflow.set_tags({
            "model_name": model_name,
            "project": "BTS Airline Analysis",
            "dataset": "AIRLINE_BTS_March.csv",
            "task": "departure_delay_classification",
            "target": "IS_DEP_DELAYED",
            **(tags or {}),
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        for artifact_name, path in artifact_paths.items():
            mlflow.log_artifact(
                str(path),
                artifact_path=artifact_name,
            )

        run_id = run.info.run_id

    print(f"Logged {model_name}: {run_id}")
    return run_id


def log_supervised_workflow(
    models: dict[str, Any],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv_results: pd.DataFrame,
    training_rows: int,
) -> pd.DataFrame:
    """
    Evaluate and track each fitted model on the same held-out test set.

    Saves cross-validation and test metrics as artifacts, and logs each
    model, its parameters, metrics, and identifying tags to MLflow.
    """

    if not models:
        raise ValueError("No fitted models were provided.")

    output_dir = Path.cwd() / "reports" / "mlflow"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_path = output_dir / "cross_validation_results.csv"
    metrics_path = output_dir / "test_metrics.csv"

    cv_results.to_csv(cv_path, index=False)

    rows = []
    predictions_by_model = {}

    # Evaluate all models before logging, so every run can include the
    # same complete evaluation summary as an artifact.
    for model_name, model in models.items():
        check_is_fitted(model)

        predictions = model.predict(x_test)

        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "test_f1": f1_score(
                y_test, predictions, zero_division=0
            ),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_test)
            classes = list(model.classes_)

            if len(classes) == 2:
                positive_class = 1 if 1 in classes else classes[-1]
                positive_index = classes.index(positive_class)
                metrics["test_roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities[:, positive_index],
                )

        cv_match = cv_results.loc[
            cv_results["Model"] == model_name, "Mean F1"
        ]
        if not cv_match.empty:
            metrics["cv_mean_f1"] = float(cv_match.iloc[0])

        rows.append({
            "Model": model_name,
            **metrics,
        })
        predictions_by_model[model_name] = (model, metrics)

    summary = pd.DataFrame(rows).sort_values(
        "test_f1", ascending=False
    )
    summary.to_csv(metrics_path, index=False)

    # Each model gets its own MLflow run.
    for model_name, (model, metrics) in predictions_by_model.items():
        log_model_run(
            model_name=model_name,
            model=model,
            params=model.get_params(deep=True),
            metrics=metrics,
            artifacts={
                "evaluation": str(metrics_path),
                "cross_validation": str(cv_path),
            },
            tags={
                "evaluation_split": "held_out_test",
                "training_rows": str(training_rows),
            },
        )

    print("\nMLflow tracking completed for all models.")
    print(f"Evaluation report: {metrics_path}")
    print(f"Cross-validation report: {cv_path}")

    return summary
