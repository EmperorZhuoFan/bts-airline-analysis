from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import airline.mlflow_tracking as tracking
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def test_setup_tracking_uses_project_database_from_any_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Select the repository-local database even when launched elsewhere."""
    expected_path = (Path(__file__).resolve().parents[1] / "mlflow.db").as_posix()
    expected_uri = f"sqlite:///{quote(expected_path, safe='/:')}"
    selected_uris = []
    experiment = SimpleNamespace(
        name=tracking.DEFAULT_EXPERIMENT,
        experiment_id="1",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tracking_uri",
        selected_uris.append,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_experiment",
        lambda name: experiment,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "get_tracking_uri",
        lambda: selected_uris[-1],
    )

    assert tracking.setup_tracking() == "1"
    assert selected_uris == [expected_uri]


def test_log_model_run_allows_sklearn_tree_storage(monkeypatch) -> None:
    """Keep skops serialization safe while allowing fitted tree models."""
    model = DecisionTreeClassifier(random_state=0).fit(
        [[0], [1]], [0, 1]
    )
    logged_model = {}

    monkeypatch.setattr(tracking, "setup_tracking", lambda: "1")
    monkeypatch.setattr(
        tracking.mlflow,
        "start_run",
        lambda **kwargs: nullcontext(
            SimpleNamespace(info=SimpleNamespace(run_id="test-run"))
        ),
    )
    monkeypatch.setattr(tracking.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(tracking.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(tracking.mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(
        tracking.mlflow.sklearn,
        "log_model",
        lambda **kwargs: logged_model.update(kwargs),
    )

    tracking.log_model_run(
        model_name="Decision tree",
        model=model,
        params={},
        metrics={"test_accuracy": 1.0},
    )

    assert logged_model["skops_trusted_types"] == [
        "sklearn.tree._tree.Tree"
    ]


def test_log_supervised_workflow_records_training_row_count(
    tmp_path: Path, monkeypatch
) -> None:
    x_train = pd.DataFrame({"feature": [0, 1, 2, 3]})
    model = DecisionTreeClassifier(random_state=0).fit(
        x_train, [0, 0, 1, 1]
    )
    logged_runs = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tracking,
        "log_model_run",
        lambda **kwargs: logged_runs.append(kwargs),
    )

    tracking.log_supervised_workflow(
        models={"Decision tree": model},
        x_test=pd.DataFrame({"feature": [0, 3]}),
        y_test=pd.Series([0, 1]),
        cv_results=pd.DataFrame({
            "Model": ["Decision tree"],
            "Mean F1": [1.0],
        }),
        training_rows=4,
    )

    assert logged_runs[0]["tags"]["training_rows"] == "4"
