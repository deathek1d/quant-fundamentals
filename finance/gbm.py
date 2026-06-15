import numpy as np

T       = 1.0
N_STEPS = 252
N_PATHS = 5_000
dt      = T / N_STEPS


def simulate(S0: float, mu: float, sigma: float,
             n_paths: int = N_PATHS, n_steps: int = N_STEPS, dt: float = dt) -> np.ndarray:
    """
    Exact GBM via log-Euler scheme.
    dS = mu S dt + sigma S dW

    Returns price matrix S of shape (n_paths, n_steps+1).
    """
    dW       = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    log_inc  = (mu - 0.5 * sigma ** 2) * dt + sigma * dW
    S        = np.empty((n_paths, n_steps + 1))
    S[:, 0]  = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(log_inc, axis=1))
    return S


def simple_returns(S: np.ndarray) -> np.ndarray:
    """Period-over-period simple returns from a price matrix."""
    return np.diff(S, axis=1) / S[:, :-1]


def log_returns(S: np.ndarray) -> np.ndarray:
    """Log returns from a price matrix."""
    return np.log(S[:, 1:] / S[:, :-1])
