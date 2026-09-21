import pandas as pd 
import numpy as np 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from airline.data.loading import load_data

file_path = r"F:\October Plan Phases 1-7\__Projects__\U.S. Bureau of Transportation Statistics\U.S. Bureau of Transportation poject\AIRLINE_BTS.csv"
df = load_data(file_path=file_path)

def prepare_unsupervised_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare airport-level operational features for unsupervised learning."""

    df = df.copy()

    data = df.groupby("ORIGIN").agg(flight_volume=("ORIGIN", "size"),
                                           delayed_flights=("DEP_DELAY", lambda x: (x > 0).sum()),
                                           cancelled_flights=("CANCELLED", lambda x: (x == 1).sum()),
                                           diverted_flights=("DIVERTED", lambda x: (x == 1).sum()),
                                           avg_departure_delay=("DEP_DELAY", "mean"),
                                           avg_taxi_out=("TAXI_OUT", "mean"),
                                           avg_flight_distance=("DISTANCE", "mean"))

    data["departure_delay_rate"] = ((data["delayed_flights"]  / data["flight_volume"]) * 100).round(2) 
    data["cancellation_rate"] = ((data["cancelled_flights"]  / data["flight_volume"]) * 100).round(2) 
    data["diversion_rate"] = ((data["diverted_flights"]  / data["flight_volume"]) * 100).round(2) 

    return data



def scale_unsupervised_data(data: pd.DataFrame) -> pd.DataFrame:
    """Scale airport-level features for K-Means clustering."""

    features = ["flight_volume", "delayed_flights", "cancelled_flights", "diverted_flights",
                "avg_departure_delay", "avg_taxi_out", "avg_flight_distance", "departure_delay_rate",
                "cancellation_rate", "diversion_rate"]

    scaleable_data = data[features]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(scaleable_data)

    scaled_data = pd.DataFrame(scaled_data, columns= features, index= data.index)

    return scaled_data


def determine_cluster_count(scaled_data: pd.DataFrame) -> int:
    """Determine a suitable number of clusters using inertia and silhouette scores."""

    inertia = []
    silhouette_scores = []
    k_values = range(2,9)

    for k in k_values:

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)

        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))

    best_silhoutte_index = np.argmax(silhouette_scores)
    k_value = list(k_values)[best_silhoutte_index]

    return k_value


def train_kmeans(scaled_data: pd.DataFrame, k_value: int) -> KMeans:
    """Train a K-Means clustering model using the selected number of clusters."""

    kmeans = KMeans(n_clusters= k_value, random_state= 42)
    kmeans.fit(scaled_data)

    return kmeans


def analyze_clusters(data: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Analyze and summarize the operational characteristics of each cluster."""

    clustered_data = data.copy()

    clustered_data["cluster"] = kmeans.labels_

    cluster_profile = clustered_data.groupby("cluster").agg(airport_count=("cluster", "size"), avg_flight_volume=("flight_volume", "mean"),
                                avg_delayed_flights=("delayed_flights", "mean"), avg_cancelled_flights=("cancelled_flights", "mean"),
                                avg_diverted_flights=("diverted_flights", "mean"), avg_departure_delay=("avg_departure_delay", "mean"),
                                avg_taxi_out=("avg_taxi_out", "mean"), avg_flight_distance=("avg_flight_distance", "mean"),
                                avg_departure_delay_rate=("departure_delay_rate", "mean"), avg_cancellation_rate=("cancellation_rate", "mean"),
                                avg_diversion_rate=("diversion_rate", "mean")).round(2)

    return cluster_profile




def run_unsupervised_learning(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete unsupervised learning workflow."""

    airport_data = prepare_unsupervised_data(df)
    scaled_data = scale_unsupervised_data(airport_data)
    k_value = determine_cluster_count(scaled_data)
    kmeans = train_kmeans(scaled_data, k_value)

    clustered_data = airport_data.copy()
    clustered_data["cluster"] = kmeans.labels_

    cluster_profile = analyze_clusters(airport_data, kmeans)

    return clustered_data, cluster_profile, scaled_data, kmeans
