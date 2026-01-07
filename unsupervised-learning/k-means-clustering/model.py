import numpy as np

class KMeans:
    def __init__(self, k = 3, num_iters = 100):
        self.k = k
        self.num_iters = num_iters
        self.centroids = None
        self.labels = None

        self.record_every = 10
        self.history = []

    def assign_clusters(self, X):
        # Compute distance to every point for each centroid
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        # Pick centroid with smallest distance (0,1,2)
        return np.argmin(distances, axis=0)

    def fit(self, X):
        n_samples, n_features = X.shape

        # Randomly pick k data points and use them as centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.num_iters):
            self.labels = self.assign_clusters(X)

            # For each cluster compute the mean of points assigned to it (same label)
            new_centroids = np.array([
                X[self.labels == j].mean(axis=0)
                for j in range(self.k)
            ])

            if i % self.record_every == 0:
                self.history.append(
                    (self.centroids.copy(), self.labels.copy())
                )

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids