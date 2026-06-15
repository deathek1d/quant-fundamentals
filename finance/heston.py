import numpy as np

T       = 1.0
N_STEPS = 252
N_PATHS = 5_000
dt      = T / N_STEPS


def simulate(S0: float, V0: float, r: float, q: float,
             kappa: float, m: float, eta: float, rho: float,
             n_paths: int = N_PATHS, n_steps: int = N_STEPS, dt: float = dt):
    """
    Euler-Maruyama for the Heston stochastic volatility model.

    dS = (r - q) S dt + sqrt(V) S dW1
    dV = kappa (m - V) dt + eta sqrt(V) (rho dW1 + sqrt(1-rho^2) dW2)

    Returns (S, V), both of shape (n_paths, n_steps+1).
    """
    S, V       = np.zeros((n_paths, n_steps + 1)), np.zeros((n_paths, n_steps + 1))
    S[:, 0]    = S0
    V[:, 0]    = V0
    sqrt_dt    = np.sqrt(dt)

    for t in range(n_steps):
        dW1     = np.random.normal(0, 1, n_paths)
        dW2     = np.random.normal(0, 1, n_paths)
        sqrt_V  = np.sqrt(np.maximum(V[:, t], 0))

        S[:, t + 1] = S[:, t] + (r - q) * S[:, t] * dt + sqrt_V * S[:, t] * dW1 * sqrt_dt
        V[:, t + 1] = (V[:, t]
                       + kappa * (m - V[:, t]) * dt
                       + eta * sqrt_V * (rho * dW1 + np.sqrt(1 - rho ** 2) * dW2) * sqrt_dt)
    return S, V
