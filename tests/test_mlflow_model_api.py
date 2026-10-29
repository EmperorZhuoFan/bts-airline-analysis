import mlflow
import mlflow.sklearn

from airline.mlflow_tracking import setup_tracking


def test_registered_models_can_be_loaded():
    setup_tracking()

    client = mlflow.MlflowClient()
    models = client.search_registered_models()

    assert models, "No registered models found. Register a model before running this test."

    for registered_model in models:
        versions = client.get_latest_versions(registered_model.name)

        assert versions, f"No versions found for {registered_model.name}"

        version = versions[0].version
        model_uri = f"models:/{registered_model.name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)

        assert model is not None