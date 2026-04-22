import numpy as np


class MLP:
    """
    Two-layer MLP: 784 → 128 (ReLU) → 10 (Softmax)
    Built entirely in NumPy — no deep learning frameworks.
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # He initialization for ReLU layers
        self.params = {
            "W1": np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            "b1": np.zeros(hidden_size),
            "W2": np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            "b2": np.zeros(output_size),
        }
        self.cache = {}

    def forward(self, X):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)          # ReLU

        z2 = a1 @ W2 + b2
        # Numerically stable softmax
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        self.cache = (X, z1, a1, a2)
        return a2

    def loss(self, y_pred, y_true):
        m = len(y_true)
        log_likelihood = -np.log(y_pred[np.arange(m), y_true] + 1e-8)
        return np.mean(log_likelihood)

    def backward(self, y_true):
        X, z1, a1, a2 = self.cache
        W2 = self.params["W2"]
        m = len(y_true)

        dz2 = a2.copy()
        dz2[np.arange(m), y_true] -= 1
        dz2 /= m

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)           # ReLU gradient

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_weights(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_weights(self, weights):
        self.params = {k: v.copy() for k, v in weights.items()}

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)