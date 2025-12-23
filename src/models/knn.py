from sklearn.neighbors import KNeighborsClassifier

def build_knn(n_neighbors=5, weights="distance", n_jobs=-1):
    """
    Builds a K-Nearest Neighbors classifier.

    Args:
        n_neighbors (int): Number of neighbors
        weights (str): 'uniform' or 'distance'
        n_jobs (int): Number of parallel jobs

    Returns:
        sklearn.neighbors.KNeighborsClassifier
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=n_jobs
    )
