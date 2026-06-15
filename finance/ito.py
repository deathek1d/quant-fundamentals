import numpy as np

T       = 1.0
N_STEPS = 252
N_PATHS = 5_000
dt      = T / N_STEPS


def simulate(b: float, sigma: float,
             n_paths: int = N_PATHS, n_steps: int = N_STEPS, dt: float = dt) -> np.ndarray:
    """Simulate paths of a linear Itô process: dX = b dt + sigma dW."""
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    dX = b * dt + sigma * dW
    return np.cumsum(dX, axis=1)


def quadratic_variation(dW: np.ndarray) -> np.ndarray:
    """Quadratic variation per path. Converges to T as dt → 0."""
    return np.sum(dW ** 2, axis=1)
