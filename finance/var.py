import numpy as np


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical VaR at a given confidence level.
    Returns the loss quantile (a negative number for a loss).
    """
    return float(np.percentile(returns, (1 - confidence) * 100))


def rolling_var(returns: np.ndarray, window: int, confidence: float = 0.95) -> np.ndarray:
    """
    Rolling historical VaR over a sliding window of daily returns.
    Returns an array of length len(returns) - window + 1.
    """
    return np.array([
        historical_var(returns[i - window: i], confidence)
        for i in range(window, len(returns) + 1)
    ])


def parametric_var(mu: float, sigma: float, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR: mu - z * sigma.
    Returns the loss quantile as a negative number.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - confidence)
    return mu - z * sigma
