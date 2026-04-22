import numpy as np
from tensorflow.keras.datasets import mnist


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test  = X_test.reshape(-1, 784) / 255.0
    return X_train, y_train, X_test, y_test


def iid_split(X, y, n_clients):
    """
    Randomly shuffle all samples and divide equally among clients.
    Each client sees all 10 digit classes in roughly equal proportion.
    """
    idx = np.random.permutation(len(X))
    splits = np.array_split(idx, n_clients)
    return [(X[i], y[i]) for i in splits]


def noniid_split(X, y, n_clients):
    """
    Each client receives data from exactly ONE digit class.
    This is maximum heterogeneity — the hardest possible Non-IID setting.

    With 10 clients on MNIST:
        Client 0 → only digit 0
        Client 1 → only digit 1
        ...
        Client 9 → only digit 9

    This guarantees a visible convergence gap vs IID.
    """
    client_data = []
    for c in range(n_clients):
        # Find all samples belonging to class c
        idx = np.where(y == c)[0]
        client_data.append((X[idx], y[idx]))
    return client_data