"""
test_autodiff.py - check the engine's gradients against central finite
differences on a function that exercises every op the model uses.
"""
import numpy as np
from autodiff import Tensor


def test_gradients():
    rng = np.random.default_rng(0)
    N, d, h = 8, 4, 5
    x = rng.standard_normal((N, d))
    r = rng.standard_normal((N, 1))          # stands in for price moves
    lam = 1.3

    params = {
        "W1": Tensor(rng.standard_normal((d, h)) * 0.5),
        "b1": Tensor(np.zeros((1, h))),
        "W2": Tensor(rng.standard_normal((h, 1)) * 0.5),
        "b2": Tensor(np.zeros((1, 1))),
    }

    def loss_fn():
        X = Tensor(x)
        z = (X @ params["W1"] + params["b1"]).tanh()
        pos = z @ params["W2"] + params["b2"]        # (N,1) hedge position
        pnl = (pos * Tensor(r)).sum(axis=1)          # (N,) elementwise + reduce
        cost = (pos.abs()).sum(axis=1) * 0.01        # exercise abs
        wealth = pnl - cost
        # entropic risk: (1/lam) * log(mean(exp(-lam * wealth)))
        risk = (((-lam) * wealth).exp().mean().log()) * (1.0 / lam)
        return risk

    # analytic gradients
    loss = loss_fn()
    loss.backward()
    analytic = {k: v.grad.copy() for k, v in params.items()}

    # numerical gradients (central differences)
    eps = 1e-6
    max_rel = 0.0
    for k, p in params.items():
        num = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = p.data[idx]
            p.data[idx] = orig + eps; fp = loss_fn().data
            p.data[idx] = orig - eps; fm = loss_fn().data
            p.data[idx] = orig
            num[idx] = (fp - fm) / (2 * eps)
            it.iternext()
        denom = np.maximum(np.abs(analytic[k]) + np.abs(num), 1e-8)
        rel = np.max(np.abs(analytic[k] - num) / denom)
        max_rel = max(max_rel, rel)
        print(f"  {k:4}  max rel error vs finite-diff: {rel:.2e}")
        # reset accumulated grads for cleanliness
    print(f"\nworst relative error: {max_rel:.2e}")
    assert max_rel < 1e-4, "gradient check FAILED"
    print("PASS: autodiff gradients match finite differences")


if __name__ == "__main__":
    test_gradients()
