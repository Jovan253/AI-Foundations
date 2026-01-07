import numpy as np
import matplotlib.pyplot as plt
from model import KMeans

np.random.seed(42)

X = np.vstack([
    np.random.randn(100, 2) * 1.5 + np.array([0, 0]),
    np.random.randn(100, 2) * 1.5 + np.array([2, 2]),
    np.random.randn(100, 2) * 1.5 + np.array([4, 0])
])

model = KMeans(k=3, num_iters=100)
model.fit(X)


for step, (centroids, labels) in enumerate(model.history):
    plt.figure()
    for i in range(model.k):
        plt.scatter(
            X[labels == i, 0],
            X[labels == i, 1]
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x"
    )

    plt.title(f"Iteration {step * model.snapshot}")
    plt.show()
