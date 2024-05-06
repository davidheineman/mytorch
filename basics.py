import numpy as np


def sigmoid(z):
    """
    S(x) = 1/1+e^-x
    """
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.W, self.b = None, None

    def fit(self, X, y):
        N, M = X.shape # (batch size, n features)
        self.W, self.b = np.zeros(M), 0

        # Gradient descent
        for _ in range(self.n_iter):
            linear_model = X @ self.W + self.b
            y_pred = sigmoid(linear_model)

            # Update W and b
            loss = y_pred - y
            dw = (1 / N) * X.T @ loss
            db = (1 / N) * np.sum(loss)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        linear_model = X @ self.W + self.b
        y_pred = sigmoid(linear_model)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred
