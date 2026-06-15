import numpy as np
from scipy.stats import norm


def integrate_1d(func, a: float, b: float, n_samples: int = 10_000) -> tuple[float, float, float]:
    """
    Monte Carlo estimate of a 1-D integral over [a, b].

    Uses the importance-sampling identity:
        E[f(U)] ≈ (b - a) * mean(f(u_i))   where u_i ~ Uniform(a, b)

    Parameters
    ----------
    func      : callable, the integrand f(x)
    a, b      : integration bounds
    n_samples : number of uniform draws

    Returns
    -------
    estimate  : MC approximation of the integral
    stderr    : standard error of the estimate
    ci_95     : half-width of the 95% confidence interval
    """
    u = np.random.uniform(a, b, n_samples)
    f_vals = func(u)
    volume = b - a
    estimate = volume * np.mean(f_vals)
    stderr = volume * np.std(f_vals, ddof=1) / np.sqrt(n_samples)
    ci_95 = norm.ppf(0.975) * stderr
    return estimate, stderr, ci_95


def integrate_nd(func, lb: np.ndarray, ub: np.ndarray, n_samples: int = 10_000) -> tuple[float, float, float]:
    """
    Monte Carlo estimate of a multi-dimensional integral over a hyper-rectangle.

    Parameters
    ----------
    func      : callable, f(x) where x has shape (d, n_samples)
    lb, ub    : lower/upper bounds, arrays of shape (d,)
    n_samples : number of uniform draws

    Returns
    -------
    estimate, stderr, ci_95
    """
    lb, ub = np.asarray(lb), np.asarray(ub)
    volume = np.prod(ub - lb)
    d = len(lb)
    u = lb[:, None] + (ub - lb)[:, None] * np.random.uniform(0, 1, (d, n_samples))
    f_vals = func(u)
    estimate = volume * np.mean(f_vals)
    stderr = volume * np.std(f_vals, ddof=1) / np.sqrt(n_samples)
    ci_95 = norm.ppf(0.975) * stderr
    return estimate, stderr, ci_95


def convergence_table(func, a: float, b: float, exact: float, exponents: range = range(1, 9)):
    """
    Print a convergence table (M, exact, MC approx, abs error, stderr, %, 95% CI)
    as M grows from 10^exponents[0] to 10^exponents[-1].

    Parameters
    ----------
    func      : integrand
    a, b      : integration bounds
    exact     : known exact value of the integral
    exponents : iterable of integer exponents for M = 10^j
    """
    print(f"{'M':>12} {'Exact':>10} {'MC':>10} {'|Error|':>10} {'StdErr':>10} {'%(StdErr)':>10} {'95% CI':>20}")
    print("-" * 85)
    for j in exponents:
        M = 10 ** j
        est, se, ci = integrate_1d(func, a, b, n_samples=M)
        print(f"{M:>12,} {exact:>10.6f} {est:>10.6f} {abs(exact-est):>10.6f} "
              f"{se:>10.6f} {se/est*100:>10.4f} [{est-ci:.6f}, {est+ci:.6f}]")


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import math

    # 1-D: integral of e^x from 0 to 1  →  e - 1 ≈ 1.71828
    print("=== 1-D: integral of exp(x) from 0 to 1 ===")
    exact_1d = math.e - 1
    convergence_table(np.exp, 0.0, 1.0, exact_1d)

    print()

    # 1-D: area of unit circle: A = 2*pi * integral of x from 0 to 1 = pi
    print("=== 1-D: A = 2*pi * integral of x from 0 to 1 (= pi) ===")
    est, se, ci = integrate_1d(lambda x: 2 * math.pi * x, 0.0, 1.0, n_samples=1_000_000)
    print(f"  Estimate : {est:.6f}  (exact pi = {math.pi:.6f})")
    print(f"  StdErr   : {se:.6f},  95% CI: [{est-ci:.6f}, {est+ci:.6f}]")

    print()

    # 3-D: integral of (x1^2 + 4*x2 + 1) over [11,14] x [7,10] x [0,1] = 1728
    print("=== 3-D: integral of (x1^2 + 4x2 + 1) over [11,14]x[7,10]x[0,1] = 1728 ===")
    lb3 = np.array([11.0, 7.0, 0.0])
    ub3 = np.array([14.0, 10.0, 1.0])
    f3d = lambda u: u[0] ** 2 + 4 * u[1] + 1
    est3, se3, ci3 = integrate_nd(f3d, lb3, ub3, n_samples=1_000_000)
    print(f"  Estimate : {est3:.2f}  (exact = 1728)")
    print(f"  StdErr   : {se3:.4f},  95% CI: [{est3-ci3:.2f}, {est3+ci3:.2f}]")
