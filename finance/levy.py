import numpy as np

T       = 1.0
N_STEPS = 252
dt      = T / N_STEPS


def simulate_log_process(mu: float, sigma: float, lambda_jump: float,
                         jump_mean: float, jump_std: float,
                         n_steps: int = N_STEPS, dt: float = dt) -> np.ndarray:
    """
    Single path of a Lévy jump-diffusion log-process.
    X_t = mu t + sigma W_t + sum_{i=1}^{N_t} J_i

    Returns log-price increment path X of shape (n_steps+1,).
    """
    X = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        brownian   = sigma * np.sqrt(dt) * np.random.normal()
        n_jumps    = np.random.poisson(lambda_jump * dt)
        total_jump = np.sum(np.random.normal(jump_mean, jump_std, n_jumps))
        X[i]       = X[i - 1] + mu * dt + brownian + total_jump
    return X


def simulate_price(S0: float, mu: float, sigma: float, lambda_jump: float,
                   jump_mean: float, jump_std: float,
                   n_steps: int = N_STEPS, dt: float = dt) -> np.ndarray:
    """
    Price process S_t = S0 * exp(X_t) for a Lévy jump-diffusion.
    Returns price path of shape (n_steps+1,).
    """
    X = simulate_log_process(mu, sigma, lambda_jump, jump_mean, jump_std, n_steps, dt)
    return S0 * np.exp(X)
