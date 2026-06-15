import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────
# BLACK-SCHOLES CLOSED FORM
# ─────────────────────────────────────────────

def bs_price(S0: float, K, r: float, T: float, sigma: float):
    """
    Black-Scholes closed-form price for European vanilla options.

    Parameters
    ----------
    S0    : spot price
    K     : strike(s), scalar or array
    r     : continuously compounded risk-free rate
    T     : time to maturity (years)
    sigma : annualised volatility

    Returns
    -------
    call, put  : arrays (or scalars) of option prices
    """
    K = np.asarray(K, dtype=float)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return call, put


# ─────────────────────────────────────────────
# MONTE CARLO — GBM (Black-Scholes)
# ─────────────────────────────────────────────

def mc_bs(S0: float, K, r: float, T: float, sigma: float,
          n_paths: int = 10_000, n_steps: int = 1, seed: int = None):
    """
    Monte Carlo pricing of European calls and puts under GBM.

    Uses the exact log-Euler scheme (single step is exact for GBM).

    Returns
    -------
    call, put   : (price, stderr) tuples
    """
    rng = np.random.default_rng(seed)
    K   = np.asarray(K, dtype=float)
    dt  = T / n_steps
    mu  = r - 0.5 * sigma ** 2

    # Simulate log-price increments, shape (n_steps, n_paths)
    dW  = mu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))
    S_T = S0 * np.exp(np.sum(dW, axis=0))   # terminal price, shape (n_paths,)

    df  = np.exp(-r * T)
    CT  = df * np.maximum(S_T[None, :] - K[:, None], 0.0)  # (n_strikes, n_paths)
    PT  = df * np.maximum(K[:, None] - S_T[None, :], 0.0)

    call_price  = CT.mean(axis=1)
    call_stderr = CT.std(axis=1, ddof=1) / np.sqrt(n_paths)
    put_price   = PT.mean(axis=1)
    put_stderr  = PT.std(axis=1, ddof=1) / np.sqrt(n_paths)

    return (call_price, call_stderr), (put_price, put_stderr)


# ─────────────────────────────────────────────
# MONTE CARLO — NIG process
# Normal Inverse Gaussian: X_t = theta*G_t + sigma*sqrt(G_t)*Z
# where G_t ~ InvGauss(t, kappa*t^2)
# ─────────────────────────────────────────────

def _sample_nig(theta: float, sigma: float, kappa: float,
                T: float, n_paths: int, rng) -> np.ndarray:
    """
    Sample terminal log-return under NIG (single step).
    """
    mu_ig   = T
    lam_ig  = T ** 2 / kappa
    # Inverse Gaussian via normal approximation method
    nu      = rng.standard_normal(n_paths)
    y       = nu ** 2
    x_ig    = (mu_ig + 0.5 * mu_ig ** 2 * y / lam_ig
                - 0.5 * mu_ig / lam_ig
                * np.sqrt(4 * mu_ig * lam_ig * y + mu_ig ** 2 * y ** 2))
    u       = rng.uniform(size=n_paths)
    g       = np.where(u <= mu_ig / (mu_ig + x_ig), x_ig, mu_ig ** 2 / x_ig)
    z       = rng.standard_normal(n_paths)
    return theta * g + sigma * np.sqrt(g) * z


def mc_nig(S0: float, K: float, r: float, T: float,
           theta: float, sigma: float, kappa: float,
           n_paths: int = 10_000, seed: int = None):
    """
    Monte Carlo pricing of European call under the NIG model.

    Risk-neutral drift: mu = r - log(E[e^{X_1}])

    Returns
    -------
    call_price, call_stderr
    """
    rng = np.random.default_rng(seed)
    # Risk-neutral correction: characteristic function at -i
    omega = np.log(1 - theta * kappa - 0.5 * sigma ** 2 * kappa) / kappa
    mu_rn = r + omega         # log-return drift so E[S_T] = S0 * exp(r*T)

    X_T  = mu_rn * T + _sample_nig(theta, sigma, kappa, T, n_paths, rng)
    S_T  = S0 * np.exp(X_T)
    CT   = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    return CT.mean(), CT.std(ddof=1) / np.sqrt(n_paths)


# ─────────────────────────────────────────────
# MONTE CARLO — HESTON (Euler-Maruyama, n_steps steps)
# ─────────────────────────────────────────────

def mc_heston(S0: float, K: float, r: float, T: float,
              v0: float, kappa: float, theta: float, eta: float, rho: float,
              n_paths: int = 1_000, n_steps: int = 250, seed: int = None):
    """
    Monte Carlo pricing of a European call under the Heston model.

    Uses Euler-Maruyama discretisation for both S and V.
    Variance is floored at zero (full-truncation scheme).

    Returns
    -------
    call_price, call_stderr
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    sdt = np.sqrt(dt)

    S = np.full(n_paths, S0, dtype=float)
    V = np.full(n_paths, v0, dtype=float)

    for _ in range(n_steps):
        dW1 = rng.standard_normal(n_paths)
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
        V_p = np.maximum(V, 0.0)
        S  *= np.exp((r - 0.5 * V_p) * dt + np.sqrt(V_p) * sdt * dW1)
        V  += kappa * (theta - V_p) * dt + eta * np.sqrt(V_p) * sdt * dW2
        V   = np.maximum(V, 0.0)

    CT = np.exp(-r * T) * np.maximum(S - K, 0.0)
    return CT.mean(), CT.std(ddof=1) / np.sqrt(n_paths)


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    S0, r, sigma, T = 100.0, 0.01, 0.2, 1.0
    K = np.arange(80, 125, 5, dtype=float)

    print("=== Black-Scholes: closed form vs Monte Carlo ===")
    bs_call, bs_put = bs_price(S0, K, r, T, sigma)
    (mc_call, mc_ce), (mc_put, mc_pe) = mc_bs(S0, K, r, T, sigma, n_paths=100_000)
    print(f"{'K':>6} {'BS Call':>10} {'MC Call':>10} {'StdErr':>10}")
    for i, k in enumerate(K):
        print(f"{k:>6.0f} {bs_call[i]:>10.4f} {mc_call[i]:>10.4f} {mc_ce[i]:>10.4f}")

    print()
    print("=== NIG model: ATM call ===")
    theta, sig_nig, kappa_nig = -0.0327, 0.1843, 0.1708
    c, se = mc_nig(S0, 100.0, r, T, theta, sig_nig, kappa_nig, n_paths=100_000)
    print(f"  MC price : {c:.4f}  ±  {se:.4f}  (95% CI: [{c-1.96*se:.4f}, {c+1.96*se:.4f}])")

    print()
    print("=== Heston model: ATM call ===")
    v0_h, kappa_h, theta_h, eta_h, rho_h = 0.010201, 6.21, 0.019, 0.61, -0.7
    c_h, se_h = mc_heston(S0, 100.0, r, T, v0_h, kappa_h, theta_h, eta_h, rho_h,
                           n_paths=1_000, n_steps=250)
    print(f"  MC price : {c_h:.4f}  ±  {se_h:.4f}  (95% CI: [{c_h-1.96*se_h:.4f}, {c_h+1.96*se_h:.4f}])")
