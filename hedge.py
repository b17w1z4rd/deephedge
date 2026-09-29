"""
hedge.py - the deep hedger, its training, and the delta-hedge benchmark.

Setup: we are SHORT one at-the-money European call and receive its Black-Scholes
premium. We hedge by trading the underlying at each step, paying a proportional
transaction cost. Terminal wealth is

    W = premium + sum_t H_t (S_{t+1} - S_t) - costs - payoff(S_T)

A perfect continuous hedge with no costs gives W = 0. With discrete steps and
costs it's a distribution, and we train a neural network policy H_t = f(state)
to minimize the *entropic risk* of W - a smooth, convex risk measure that
punishes losses. The classical benchmark holds the Black-Scholes delta.

The interesting result: with no costs the network reproduces delta hedging; with
costs it learns to trade less and at better moments, giving a tighter P&L
distribution and smaller tail risk.
"""
import sys
import numpy as np
from autodiff import Tensor
from market import bs_delta, bs_price

sys.setrecursionlimit(1_000_000)     # the hedging graph unrolls over many steps


# --------------------------------------------------------------------------- #
# Policy network (built on the autodiff engine). State: [log-moneyness, tau] +
# the current holding fed as its own input so the net can see its position.
# --------------------------------------------------------------------------- #
class Policy:
    def __init__(self, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        def w(a, b): return Tensor(rng.standard_normal((a, b)) * np.sqrt(2.0 / a))
        self.Wa = w(2, hidden)               # base features -> hidden
        self.Wh = w(1, hidden)               # current holding -> hidden
        self.b1 = Tensor(np.zeros((1, hidden)))
        self.W2 = w(hidden, hidden)
        self.b2 = Tensor(np.zeros((1, hidden)))
        self.W3 = w(hidden, 1)
        self.b3 = Tensor(np.zeros((1, 1)))

    def params(self):
        return [self.Wa, self.Wh, self.b1, self.W2, self.b2, self.W3, self.b3]

    def act(self, base, hprev):              # Tensors -> Tensor (N,1)
        z = (base @ self.Wa + hprev @ self.Wh + self.b1).tanh()
        z = (z @ self.W2 + self.b2).tanh()
        return z @ self.W3 + self.b3

    def act_np(self, base, hprev):           # numpy fast path (no graph)
        z = np.tanh(base @ self.Wa.data + hprev @ self.Wh.data + self.b1.data)
        z = np.tanh(z @ self.W2.data + self.b2.data)
        return z @ self.W3.data + self.b3.data


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(t.data) for t in params]
        self.v = [np.zeros_like(t.data) for t in params]
        self.t = 0

    def zero_grad(self):
        for t in self.p:
            t.grad = np.zeros_like(t.data)

    def step(self):
        self.t += 1
        for i, t in enumerate(self.p):
            g = t.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            t.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------- #
# Hedging simulations.
# --------------------------------------------------------------------------- #
def _features(S, t, tau, K):
    n = S.shape[0]
    return np.stack([np.log(S[:, t] / K), np.full(n, tau)], axis=1)


def deep_wealth_tensor(policy, S, K, premium, kappa, T):
    """Differentiable terminal wealth of the neural hedger (for training)."""
    n, steps = S.shape[0], S.shape[1] - 1
    dt = T / steps
    hprev = Tensor(np.zeros((n, 1)))
    pnl = Tensor(np.zeros((n, 1)))
    cost = Tensor(np.zeros((n, 1)))
    for t in range(steps):
        tau = T - t * dt
        base = Tensor(_features(S, t, tau, K))
        H = policy.act(base, hprev)
        dS = Tensor((S[:, t + 1] - S[:, t]).reshape(n, 1))
        pnl = pnl + H * dS
        cost = cost + (H - hprev).abs() * (kappa * S[:, t].reshape(n, 1))
        hprev = H
    cost = cost + hprev.abs() * (kappa * S[:, -1].reshape(n, 1))          # liquidate at T
    payoff = np.maximum(S[:, -1] - K, 0.0).reshape(n, 1)
    return premium + pnl - cost - Tensor(payoff)


def deep_wealth_np(policy, S, K, premium, kappa, T, record=False):
    """Fast, memory-light wealth + optional per-step positions for plotting."""
    n, steps = S.shape[0], S.shape[1] - 1
    dt = T / steps
    hprev = np.zeros((n, 1))
    pnl = np.zeros(n); cost = np.zeros(n)
    positions = np.zeros((n, steps)) if record else None
    for t in range(steps):
        tau = T - t * dt
        base = _features(S, t, tau, K)
        H = policy.act_np(base, hprev)
        pnl += H[:, 0] * (S[:, t + 1] - S[:, t])
        cost += np.abs(H - hprev)[:, 0] * (kappa * S[:, t])
        hprev = H
        if record:
            positions[:, t] = H[:, 0]
    cost += np.abs(hprev)[:, 0] * (kappa * S[:, -1])
    payoff = np.maximum(S[:, -1] - K, 0.0)
    w = premium + pnl - cost - payoff
    return (w, positions) if record else w


def delta_wealth_np(S, K, r, sigma, premium, kappa, T, record=False):
    """Terminal wealth of the Black-Scholes delta-hedging benchmark."""
    n, steps = S.shape[0], S.shape[1] - 1
    dt = T / steps
    hprev = np.zeros(n)
    pnl = np.zeros(n); cost = np.zeros(n)
    positions = np.zeros((n, steps)) if record else None
    for t in range(steps):
        tau = T - t * dt
        H = bs_delta(S[:, t], K, tau, r, sigma)
        pnl += H * (S[:, t + 1] - S[:, t])
        cost += kappa * np.abs(H - hprev) * S[:, t]
        hprev = H
        if record:
            positions[:, t] = H
    cost += kappa * np.abs(hprev) * S[:, -1]
    payoff = np.maximum(S[:, -1] - K, 0.0)
    w = premium + pnl - cost - payoff
    return (w, positions) if record else w


# --------------------------------------------------------------------------- #
# Risk measures.
# --------------------------------------------------------------------------- #
def entropic_risk_tensor(w, lam):
    return (((-lam) * w).exp().mean().log()) * (1.0 / lam)


def metrics(w, lam=1.0, alpha=0.95):
    losses = -w
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()
    ent = float(np.log(np.mean(np.exp(-lam * w))) / lam)
    return {"mean": w.mean(), "std": w.std(), "cvar95": cvar, "entropic": ent}


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #
def train(policy, S, K, premium, kappa, T, lam=1.0, iters=400, batch=2048,
          lr=1e-2, seed=0, log_every=50):
    rng = np.random.default_rng(seed)
    opt = Adam(policy.params(), lr=lr)
    history = []
    n = S.shape[0]
    for it in range(1, iters + 1):
        if it == int(iters * 0.6):
            opt.lr *= 0.3                      # mild LR decay for a cleaner landing
        idx = rng.integers(0, n, size=batch)
        opt.zero_grad()
        w = deep_wealth_tensor(policy, S[idx], K, premium, kappa, T)
        loss = entropic_risk_tensor(w, lam)
        loss.backward()
        opt.step()
        history.append(float(loss.data))
        if log_every and (it % log_every == 0 or it == 1):
            print(f"  iter {it:4d}   entropic risk {float(loss.data):+.4f}")
    return history


if __name__ == "__main__":
    from market import gbm_paths
    S0 = K = 100.0
    r, sigma, T, n_steps = 0.0, 0.2, 1.0 / 12, 30
    kappa = 0.01
    premium = float(bs_price(S0, K, T, r, sigma))

    Str = gbm_paths(S0, r, sigma, T, n_steps, 40_000, seed=1)
    Ste = gbm_paths(S0, r, sigma, T, n_steps, 100_000, seed=99)

    print(f"premium (BS): {premium:.4f}   transaction cost kappa={kappa}")
    print("training deep hedger...")
    pol = Policy(hidden=32, seed=0)
    train(pol, Str, K, premium, kappa, T, lam=1.0, iters=400, batch=2048, lr=1e-2)

    wd = deep_wealth_np(pol, Ste, K, premium, kappa, T)
    wb = delta_wealth_np(Ste, K, r, sigma, premium, kappa, T)
    md, mb = metrics(wd), metrics(wb)
    print("\n            mean      std     CVaR95   entropic")
    print(f"delta   {mb['mean']:+.4f}  {mb['std']:.4f}  {mb['cvar95']:+.4f}  {mb['entropic']:+.4f}")
    print(f"deep    {md['mean']:+.4f}  {md['std']:.4f}  {md['cvar95']:+.4f}  {md['entropic']:+.4f}")
    print(f"\nCVaR95 improvement: {(mb['cvar95']-md['cvar95'])/abs(mb['cvar95'])*100:.1f}%")
    assert md["entropic"] <= mb["entropic"] + 1e-3, "deep hedge did not beat delta under costs"
    print("PASS: deep hedger achieves lower risk than delta hedging under costs")
