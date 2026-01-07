import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, n_iterations):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)

            # Loss
            loss = -np.mean(
                y * np.log(y_hat) +
                (1-y) * np.log(1 - y_hat)
            )

            self.losses.append(loss/n_samples)

            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_prob(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)
