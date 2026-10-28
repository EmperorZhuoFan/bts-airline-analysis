import mlflow
from mlflow import MlflowClient


MODELS = {
    "BTS_Airline_Baseline": "a6eb118554a34b7b93f8203330c3f12d",
    "BTS_Airline_LogisticRegression": "c8963f62e0864177a6811e5bff370fbe",
    "BTS_Airline_DecisionTree": "0093d3b00db248b08bc4b0a5e78f5eec",
    "BTS_Airline_Bagging": "5532d09ae2b54252a81641b906c748f0",
    "BTS_Airline_RandomForest": "c7350844bf8d471eb29684b221d57e62",
    "BTS_Airline_AdaBoosting": "38f54da13fb6435b9eb82ff1c8b8c892",
    "BTS_Airline_GradientBoosting": "bf6a8092d28847d3898ad53530e92698",
}


def register_model(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> None:
    """Register a logged MLflow model and report its assigned version."""

    model_uri = f"runs:/{run_id}/model"

    try:
        # Register the model; MLflow assigns the version automatically.
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        print(
            f"Registered: {model_name} | "
            f"Version: {model_version.version} | "
            f"Run ID: {run_id}"
        )

    except Exception as error:
        print(f"FAILED: {model_name} | {error}")


def main() -> None:
    """Register all seven existing models."""

    client = MlflowClient()

    for model_name, run_id in MODELS.items():
        register_model(client, model_name, run_id)


if __name__ == "__main__":
    main()