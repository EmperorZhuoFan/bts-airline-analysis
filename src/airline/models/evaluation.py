import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score



def evaluate_classification_model(model_name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a trained classification model using test-set performance metrics."""

    predictions = model.predict(x_test)
    predictions_proba = model.predict_proba(x_test)[:, 1]

    evaluation_dict = {
        "Model" : model_name, 
        "Accuracy score": accuracy_score(y_test, predictions),
        "Recall score": recall_score(y_test, predictions),
        "Precision score": precision_score(y_test, predictions),
        "F1 score": f1_score(y_test, predictions), 
        "ROC-AUC score": roc_auc_score(y_test, predictions_proba)
    }

    return evaluation_dict



def evaluate_supervised_models(models: dict[str, Pipeline], x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate all supervised models using test-set performance metrics."""

    evaluation_results = []

    for model_name, model in models.items():

        evaluation_result = evaluate_classification_model(model_name, model, x_test, y_test)

        evaluation_results.append(evaluation_result)

    evaluation_results = pd.DataFrame(evaluation_results)

    return evaluation_results



def select_best_model(evaluation_results: pd.DataFrame, models: dict[str, Pipeline]) -> Pipeline:
    """Select the best supervised model based on F1 score."""

    best_score_index = evaluation_results["F1 score"].idxmax()

    model = evaluation_results.loc[best_score_index, "Model"]

    best_model = models[model]

    return best_model



def evaluate_clustering(scaled_data: pd.DataFrame, kmeans: KMeans) -> float:
    """Evaluate the quality of the K-Means clustering using the silhouette score."""

    kmeans_labels = kmeans.labels_

    score = silhouette_score(scaled_data, kmeans_labels)

    return score



def run_evaluation(models: dict[str, Pipeline], x_test: pd.DataFrame,
                   y_test: pd.Series, scaled_data: pd.DataFrame, kmeans: KMeans) -> tuple[pd.DataFrame, Pipeline, float]:
    """Run the complete evaluation workflow for supervised and unsupervised learning."""

    if not models:
        raise ValueError("No supervised models were provided.")

    if len(x_test) != len(y_test):
        raise ValueError("x_test and y_test must contain the same number of rows.")

    if scaled_data.empty:
        raise ValueError("Scaled clustering data cannot be empty.")

    if not hasattr(kmeans, "labels_"):
        raise ValueError("The K-Means model must be fitted before evaluation.")

    if len(scaled_data) != len(kmeans.labels_):
        raise ValueError("Scaled data and K-Means labels must contain the same number of rows.")

    # Orchestrate Models
    evaluation_results =  evaluate_supervised_models(models, x_test, y_test)
    best_model = select_best_model(evaluation_results, models)
    cluster_score = evaluate_clustering(scaled_data, kmeans)

    return evaluation_results, best_model, cluster_score