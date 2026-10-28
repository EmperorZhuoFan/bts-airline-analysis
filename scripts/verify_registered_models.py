import mlflow
from mlflow import MlflowClient

MODEL_NAMES = [
    "BTS_Airline_Baseline",
    "BTS_Airline_LogisticRegression",
    "BTS_Airline_DecisionTree",
    "BTS_Airline_Bagging",
    "BTS_Airline_RandomForest",
    "BTS_Airline_AdaBoosting",
    "BTS_Airline_GradientBoosting",
]

def main():
    client = MlflowClient()

    for name in MODEL_NAMES:
        versions = client.search_model_versions(f"name='{name}'")
        latest = max(versions, key=lambda version: int(version.version))

        model_uri = f"models:/{name}/{latest.version}"
        mlflow.pyfunc.load_model(model_uri)

        print(f"PASS: {name} | Version {latest.version} loaded")

if __name__ == "__main__":
    main()
    