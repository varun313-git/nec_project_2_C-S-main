from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def scale_features(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def perform_clustering(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans