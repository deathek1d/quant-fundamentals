import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────
# VALUE AT RISK  (recap — already in var.py)
# ─────────────────────────────────────────────

def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR at the given confidence level (negative number = loss)."""
    return float(np.percentile(returns, (1 - confidence) * 100))


def parametric_var(mu: float, sigma: float, confidence: float = 0.95) -> float:
    """Gaussian parametric VaR."""
    return mu - norm.ppf(confidence) * sigma


# ─────────────────────────────────────────────
# CONDITIONAL VALUE AT RISK  (Expected Shortfall)
# CVaR_alpha = E[L | L > VaR_alpha]
# ─────────────────────────────────────────────

def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical (simulation) Conditional Value at Risk.

    CVaR is the expected loss in the worst (1 - alpha)% of scenarios.
    Returns a negative number (consistent with VaR sign convention).
    """
    threshold = historical_var(returns, confidence)
    tail      = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold


def parametric_cvar(mu: float, sigma: float, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) CVaR.

    CVaR_alpha = mu - sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
    where phi is the standard normal PDF and Phi its CDF.
    """
    z     = norm.ppf(confidence)
    return mu - sigma * norm.pdf(z) / (1 - confidence)


def mc_cvar(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Monte Carlo CVaR: sort simulated P&L, average the tail.

    Parameters
    ----------
    returns    : simulated portfolio P&L (positive = gain, negative = loss)
    confidence : e.g. 0.95 means worst 5% scenarios

    Returns
    -------
    var, cvar  : both as negative numbers
    """
    sorted_r = np.sort(returns)
    cut_idx  = int(np.floor((1 - confidence) * len(sorted_r)))
    var      = sorted_r[cut_idx]
    cvar     = sorted_r[:cut_idx].mean()
    return float(var), float(cvar)


# ─────────────────────────────────────────────
# MARGINAL VAR
# MVar_i = dVaR/dw_i  (sensitivity of portfolio VaR to weight i)
# Under normality: MVar_i = beta_i * sigma_P * Phi^{-1}(alpha)
# ─────────────────────────────────────────────

def marginal_var(weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """
    Marginal VaR under multivariate normality (assumes zero portfolio drift).

    MVar_i = rho_{i,P} * sigma_P * Phi^{-1}(alpha)
           = (Sigma @ w)_i / sigma_P * Phi^{-1}(alpha)

    Parameters
    ----------
    weights    : portfolio weights, shape (n,)
    cov        : covariance matrix of asset returns, shape (n, n)
    confidence : VaR confidence level

    Returns
    -------
    mvar : array of shape (n,), marginal VaR for each asset
    """
    w     = np.asarray(weights, dtype=float)
    Sigma = np.asarray(cov, dtype=float)
    z     = norm.ppf(confidence)
    sigma_P = np.sqrt(w @ Sigma @ w)          # portfolio volatility
    beta_i  = (Sigma @ w) / sigma_P           # beta-like sensitivities
    return z * beta_i


def component_var(weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """
    Component VaR: CVaR_i = w_i * MVar_i.
    Sums to total portfolio VaR.
    """
    w    = np.asarray(weights, dtype=float)
    mvar = marginal_var(w, cov, confidence)
    return w * mvar


def portfolio_var(weights: np.ndarray, cov: np.ndarray, confidence: float = 0.95) -> float:
    """Total parametric (Gaussian) portfolio VaR."""
    w     = np.asarray(weights, dtype=float)
    Sigma = np.asarray(cov, dtype=float)
    z     = norm.ppf(confidence)
    return z * np.sqrt(w @ Sigma @ w)


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ── 1. Single-asset CVaR ──────────────────────────────────────────
    print("=== CVaR demo (single asset, GBM returns) ===")
    mu, sigma = -0.001, 0.015
    daily_ret = rng.normal(mu, sigma, 10_000)

    var_h  = historical_var(daily_ret)
    cvar_h = historical_cvar(daily_ret)
    var_p  = parametric_var(mu, sigma)
    cvar_p = parametric_cvar(mu, sigma)

    print(f"  Historical VaR  (95%): {var_h:.4f}")
    print(f"  Historical CVaR (95%): {cvar_h:.4f}")
    print(f"  Parametric  VaR (95%): {var_p:.4f}")
    print(f"  Parametric  CVaR(95%): {cvar_p:.4f}")

    # ── 2. Portfolio Marginal VaR ─────────────────────────────────────
    print()
    print("=== Marginal VaR demo (3-asset portfolio) ===")
    # Covariance matrix (annualised)
    Sigma = np.array([[0.04, 0.01, 0.005],
                      [0.01, 0.09, 0.02 ],
                      [0.005, 0.02, 0.01 ]])
    w = np.array([0.5, 0.3, 0.2])

    pvar  = portfolio_var(w, Sigma)
    mvar  = marginal_var(w, Sigma)
    cvar_components = component_var(w, Sigma)

    print(f"  Portfolio VaR (95%):   {pvar:.4f}")
    print(f"  Marginal VaR per unit: {mvar}")
    print(f"  Component VaR:         {cvar_components}  (sum = {cvar_components.sum():.4f})")

    # ── 3. Monte Carlo CVaR ───────────────────────────────────────────
    print()
    print("=== Monte Carlo CVaR (portfolio, 100k scenarios) ===")
    sim_ret = rng.multivariate_normal(np.zeros(3), Sigma / 252, 100_000) @ w
    mc_v, mc_cv = mc_cvar(sim_ret)
    print(f"  MC daily VaR  (95%): {mc_v:.5f}")
    print(f"  MC daily CVaR (95%): {mc_cv:.5f}")
