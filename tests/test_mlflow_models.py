import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_logged_model_predictions_match_original(tmp_path):
    """Check that an MLflow-loaded model reproduces its original predictions."""

    original_tracking_uri = mlflow.get_tracking_uri()
    tracking_db = (tmp_path / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")

    try:
        mlflow.set_experiment("test-mlflow-model-round-trip")

        x_train = np.array([[0], [1], [2], [3]])
        y_train = np.array([0, 0, 1, 1])
        model = LogisticRegression(random_state=0).fit(x_train, y_train)
        x_test = np.array([[0], [3]])

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, artifact_path="model")

        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

        np.testing.assert_array_equal(
            loaded_model.predict(x_test),
            model.predict(x_test),
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)