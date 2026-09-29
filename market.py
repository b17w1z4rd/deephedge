"""
market.py - price-path simulators and Black-Scholes analytics.

Two worlds to hedge in:
  - GBM: the Black-Scholes world, where delta hedging is optimal with no costs.
  - Heston: stochastic volatility, where a fixed-vol delta is already imperfect
    (so a learned hedger has more room to win).

Everything is vectorized: paths are (n_paths, n_steps + 1) arrays.
"""
import numpy as np
from scipy.stats import norm


def gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=0):
    """Geometric Brownian motion under the real-world / pricing drift `r`."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(logS)


def heston_paths(S0, r, v0, kappa, theta, xi, rho, T, n_steps, n_paths, seed=0):
    """Heston stochastic-vol paths (full-truncation Euler). Returns (S, v)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1)); S[:, 0] = S0
    v = np.empty((n_paths, n_steps + 1)); v[:, 0] = v0
    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
        vt = np.maximum(v[:, t], 0.0)
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * vt) * dt + np.sqrt(vt * dt) * z1)
        v[:, t + 1] = v[:, t] + kappa * (theta - vt) * dt + xi * np.sqrt(vt * dt) * z2
    return S, v


def bs_price(S, K, T, r, sigma, call=True):
    """Black-Scholes price. Scalar or array S/T (T -> 0 handled)."""
    S = np.asarray(S, dtype=float); T = np.asarray(T, dtype=float)
    intrinsic = np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if call:
            px = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            px = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(T > 1e-12, px, intrinsic)


def bs_delta(S, K, T, r, sigma, call=True):
    """Black-Scholes delta = shares of underlying to hold to hedge one option."""
    S = np.asarray(S, dtype=float); T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1) if call else norm.cdf(d1) - 1.0
    # at expiry, delta is a step function of moneyness
    step = (S > K).astype(float) if call else -(S < K).astype(float)
    return np.where(T > 1e-12, delta, step)


if __name__ == "__main__":
    # Verify: Monte-Carlo call price under GBM converges to the BS closed form.
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.0, 0.2
    S = gbm_paths(S0, r, sigma, T, n_steps=50, n_paths=400_000, seed=1)
    mc = np.exp(-r * T) * np.mean(np.maximum(S[:, -1] - K, 0.0))
    analytic = float(bs_price(S0, K, T, r, sigma))
    print(f"MC call price      : {mc:.4f}")
    print(f"BS closed form     : {analytic:.4f}")
    print(f"abs error          : {abs(mc - analytic):.4f}")
    assert abs(mc - analytic) < 0.05, "MC price does not match BS"
    print("PASS: simulator + BS analytics agree")
