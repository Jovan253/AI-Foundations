import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, snapshot=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.snapshot = snapshot
        self.weights = None
        self.bias = None
        self.losses = []
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % self.snapshot == 0:
                self.history.append((self.weights.copy(), self.bias))

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias