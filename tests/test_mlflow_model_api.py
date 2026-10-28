import mlflow
import mlflow.sklearn

from airline.mlflow_tracking import setup_tracking

setup_tracking()

client = mlflow.MlflowClient()
models = client.search_registered_models()

if not models:
    raise RuntimeError("No registered models found.")

for registered_model in models:
    name = registered_model.name
    versions = client.get_latest_versions(name)

    if not versions:
        raise RuntimeError(f"No versions found for {name}")

    version = versions[0].version
    model_uri = f"models:/{name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"PASS: {name} v{version} loaded ({type(model).__name__})")