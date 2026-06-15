import numpy as np

T       = 1.0
N_STEPS = 252
N_PATHS = 5_000
dt      = T / N_STEPS


def simulate(r0: float, alpha: float, beta: float, gamma: float,
             n_paths: int = N_PATHS, n_steps: int = N_STEPS, dt: float = dt) -> np.ndarray:
    """
    Euler-Maruyama discretisation of the CIR short-rate model.
    dr = (alpha - beta r) dt + gamma sqrt(r) dW

    Feller condition for non-negativity: 2*alpha > gamma^2.
    Returns rate matrix r of shape (n_paths, n_steps+1).
    """
    feller = 2 * alpha > gamma ** 2
    print(f"Feller condition (2α > γ²): {2*alpha:.4f} > {gamma**2:.4f}  →  {feller}")

    r       = np.zeros((n_paths, n_steps + 1))
    r[:, 0] = r0

    for t in range(n_steps):
        dW          = np.random.normal(0, np.sqrt(dt), n_paths)
        r[:, t + 1] = (r[:, t]
                       + (alpha - beta * r[:, t]) * dt
                       + gamma * np.sqrt(np.maximum(r[:, t], 0)) * dW)
    return r
