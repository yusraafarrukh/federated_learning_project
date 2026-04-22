class Client:
    """
    Represents one device in the federated learning network.

    In federated learning, each client:
        1. Receives the current global model weights from the server
        2. Trains locally on its own private data for E epochs
        3. Sends only the updated weights back — raw data never leaves

    This is the ClientUpdate() subroutine from FedAvg
    (McMahan et al., 2017).

    Parameters
    ----------
    X         : np.ndarray — local training features (private)
    y         : np.ndarray — local training labels   (private)
    model     : MLP        — local copy of the global model
    optimizer : Optimizer  — SGD or SGDMomentum instance
    epochs    : int        — number of local training steps per round (E)
                            More epochs = cheaper communication but
                            higher risk of client drift
    """

    def __init__(self, X, y, model, optimizer, epochs=1):
        self.X         = X
        self.y         = y
        self.model     = model
        self.optimizer = optimizer
        self.epochs    = epochs

    def train(self):
        """
        Run local training for `epochs` steps on private data.

        Returns
        -------
        weights   : dict — updated model weights to send to server
        n_samples : int  — number of local samples
                           (used by server for weighted FedAvg averaging)
        """
        for _ in range(self.epochs):
            # Forward pass — compute predictions
            preds = self.model.forward(self.X)

            # Backward pass — compute gradients via backpropagation
            grads = self.model.backward(self.y)

            # Update local model weights using optimizer
            self.model.params = self.optimizer.update(
                self.model.params, grads
            )

        # Return weights and sample count — NOT the raw data
        return self.model.get_weights(), len(self.X)
