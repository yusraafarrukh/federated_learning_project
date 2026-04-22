class Server:
    """
    Central server that runs FedAvg (McMahan et al., 2017).

    FedAvg aggregation:
        w_global = sum( n_k / N * w_k )   for each client k

    where n_k is the number of samples on client k and N is the total.
    This weighted average ensures clients with more data contribute more.
    """

    def __init__(self, model):
        self.model = model

    def aggregate(self, updates):
        total_samples = sum(n for _, n in updates)

        new_weights = {}
        for key in updates[0][0]:
            # Weighted average across all clients
            new_weights[key] = sum(
                weights[key] * (n / total_samples)
                for weights, n in updates
            )

        self.model.set_weights(new_weights)

        # Broadcast updated global weights back to all clients
        return self.model.get_weights()