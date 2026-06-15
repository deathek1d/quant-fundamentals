import numpy as np


def vomm_probabilities(lnR: np.ndarray, PDistr: np.ndarray, Rsafe: float):
    """
    Compute variance-optimal martingale measure (VOMM) risk-neutral probabilities.

    Parameters
    ----------
    lnR    : lattice log-returns
    PDistr : physical probabilities
    Rsafe  : gross risk-free return per period

    Returns
    -------
    QDistr : risk-neutral probabilities under the variance-optimal measure
    a, b   : Esscher transform coefficients
    """
    R   = np.exp(lnR)
    X   = R - Rsafe
    EX  = PDistr @ X
    EX2 = PDistr @ (X ** 2)
    a   = EX / EX2
    b   = 1 - EX ** 2 / EX2
    Q   = PDistr * (1 - a * X) / b
    return Q, a, b


def hedge(S0: float, strike: float, Tidx: int, Rsafe: float,
          lnR: np.ndarray, PDistr: np.ndarray):
    """
    Variance-optimal (VOMM) hedge for a European call on a discrete lattice.

    Parameters
    ----------
    S0     : initial stock price
    strike : option strike
    Tidx   : number of rebalancing periods + 1
    Rsafe  : gross risk-free return per period
    lnR    : array of log-returns on the lattice
    PDistr : physical probability of each log-return

    Returns
    -------
    eps2   : hedging error variance at each node (index 0 = today)
    QDistr : variance-optimal risk-neutral probabilities
    price  : VOMM option price
    """
    R              = np.exp(lnR)
    X              = R - Rsafe
    EX             = PDistr @ X
    EX2            = PDistr @ (X ** 2)
    QDistr, a, b   = vomm_probabilities(lnR, PDistr, Rsafe)

    n      = len(lnR)
    dlnR   = lnR[0] - lnR[1]
    MaxDim = 1 + (n - 1) * (Tidx - 1)

    log_S_T = np.log(S0) + (Tidx - 1) * lnR[0] - np.arange(MaxDim) * dlnR
    S_T     = np.exp(log_S_T)
    V       = np.maximum(S_T - strike, 0)        # call payoff
    eps2    = np.zeros(MaxDim)
    k_phi   = (b * Rsafe ** 2) ** np.arange(Tidx - 1, -1, -1)

    for tt in range(Tidx - 1, 0, -1):
        Vnext   = V.copy()
        epsnext = eps2.copy()
        for ii in range(1 + (n - 1) * (tt - 1)):
            focus    = Vnext[ii: ii + n]
            delta    = (PDistr * X) @ (focus - Rsafe * V[ii]) / EX2   # optimal hedge
            error    = focus - delta * X - Rsafe * V[ii]
            ESRE     = (error * PDistr) @ error
            eps2[ii] = (PDistr @ epsnext[ii: ii + n]) + k_phi[tt] * ESRE

    price = (QDistr / Rsafe) @ V[:n]
    return eps2, QDistr, price
