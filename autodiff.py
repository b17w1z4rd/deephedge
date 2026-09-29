"""
autodiff.py - a tiny reverse-mode automatic differentiation engine over NumPy.

Enough to train a neural network by backprop, with no deep-learning framework.
A Tensor wraps a NumPy array and records the operations that produced it; calling
.backward() walks that graph in reverse topological order and accumulates
gradients. Broadcasting is handled by summing gradients back down to each
operand's shape.

The whole deep-hedging model is built on this - and every gradient here is
checked against central finite differences in test_autodiff.py.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` so its shape matches `shape` (reverse of NumPy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---- construction helpers ----
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @staticmethod
    def _t(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- elementwise ops ----
    def __add__(self, other):
        other = self._t(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._t(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Tensor(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += _unbroadcast(out.grad * p * self.data ** (p - 1), self.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        other = self._t(other)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    __matmul__ = matmul

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        def _backward():
            self.grad += _unbroadcast(out.grad * (1 - t * t), self.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")
        def _backward():
            self.grad += _unbroadcast(out.grad * (self.data > 0), self.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")
        def _backward():
            self.grad += _unbroadcast(out.grad * e, self.data.shape)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")
        def _backward():
            self.grad += _unbroadcast(out.grad / self.data, self.data.shape)
        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,), "abs")
        def _backward():
            self.grad += _unbroadcast(out.grad * np.sign(self.data), self.data.shape)
        out._backward = _backward
        return out

    # ---- reductions ----
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- sugar ----
    def __neg__(self):        return self * -1.0
    def __sub__(self, o):     return self + (-self._t(o))
    def __rsub__(self, o):    return self._t(o) + (-self)
    def __radd__(self, o):    return self + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * (self._t(o) ** -1)

    # ---- backprop ----
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
