import numpy as np


class SGD:
    """
    Vanilla Stochastic Gradient Descent.

    Update rule:
        theta = theta - lr * gradient

    This is the simplest optimizer and the baseline used in most
    theoretical federated learning analysis, including the convergence
    proofs in McMahan et al. (2017) and Richtarik's distributed
    optimization work.
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        return {k: params[k] - self.lr * grads[k] for k in params}


class SGDMomentum:
    """
    SGD with Heavy-Ball Momentum.

    Update rule:
        v = beta * v + (1 - beta) * gradient
        theta = theta - lr * v

    The velocity vector v is an exponential moving average of past
    gradients. This dampens oscillations and accelerates convergence
    in directions where gradients are consistent.

    Note on the (1 - beta) scaling:
        Without it, the effective learning rate changes with beta.
        With it, the velocity is a proper weighted average of gradients
        regardless of the beta value chosen.

    Parameters
    ----------
    lr   : float — learning rate
    beta : float — momentum coefficient (typically 0.9)
    """

    def __init__(self, lr=0.01, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.v    = {}   # velocity buffer — initialised on first call

    def update(self, params, grads):
        # Initialise velocity to zero on first update
        if not self.v:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        new_params = {}
        for k in params:
            # Correct standard momentum formula with (1 - beta) scaling
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * grads[k]
            new_params[k] = params[k] - self.lr * self.v[k]

        return new_params

    def reset(self):
        """Clear velocity — call at the start of each FL round."""
        self.v = {}
