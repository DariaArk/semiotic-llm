"""Finite-difference gradient checks for every hand-derived backward pass.

    pytest tests/test_gradients.py -v
"""

import numpy as np
import pytest

from sign_prediction import (
    OppositionHead, InterpretantOperator, ScalingOperators,
    sign_loss, roundtrip_loss, grad_check,
)

TOL = 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_opposition_head_grad_r(rng):
    head = OppositionHead(rng.normal(size=(5, 3)))
    r = rng.normal(size=5)
    gv = rng.normal(size=3)
    _, cache = head.value_and_cache(r)
    gr, _ = head.backward(gv, cache)
    f = lambda x: float(np.dot(gv, head.value(x)))
    assert grad_check(f, r.copy(), gr) < TOL


def test_opposition_head_grad_A(rng):
    A = rng.normal(size=(5, 3))
    head = OppositionHead(A)
    r = rng.normal(size=5)
    gv = rng.normal(size=3)
    _, cache = head.value_and_cache(r)
    _, gA = head.backward(gv, cache)

    def f(Aflat):
        return float(np.dot(gv, OppositionHead(Aflat.reshape(A.shape)).value(r)))

    assert grad_check(f, A.flatten().copy(), gA.flatten()) < TOL


def test_interpretant_grad_W(rng):
    d = 4
    W, b = rng.normal(size=(d, 2 * d)), rng.normal(size=d)
    interp = InterpretantOperator(W, b)
    rt, od = rng.normal(size=d), rng.normal(size=d)
    g = rng.normal(size=d)
    _, x = interp.forward(rt, od)
    gW, _ = interp.backward(g, x)

    def f(Wflat):
        rh, _ = InterpretantOperator(Wflat.reshape(W.shape), b).forward(rt, od)
        return float(np.dot(g, rh))

    assert grad_check(f, W.flatten().copy(), gW.flatten()) < TOL


def test_sign_loss_grad(rng):
    d = 4
    head = OppositionHead(rng.normal(size=(d, 2)))
    rg = rng.normal(size=d)
    negs = [rng.normal(size=d), rng.normal(size=d)]
    rh = rng.normal(size=d)
    _, g, _ = sign_loss(head, rh, rg, negs)
    f = lambda x: sign_loss(head, x, rg, negs)[0]
    assert grad_check(f, rh.copy(), g) < TOL


def test_roundtrip_grad_lambda(rng):
    d = 4
    scal = ScalingOperators(rng.normal(size=(3, d)), rng.normal(size=(d, 3)))
    r = rng.normal(size=d)
    _, g, _ = roundtrip_loss(scal, r, delta_min=0.03, mu=1.0)

    def f(Lflat):
        s = ScalingOperators(Lflat.reshape(scal.Lam.shape), scal.Vop)
        return roundtrip_loss(s, r, delta_min=0.03, mu=1.0)[0]

    assert grad_check(f, scal.Lam.flatten().copy(), g["Lam"].flatten()) < TOL


def test_roundtrip_grad_V(rng):
    d = 4
    scal = ScalingOperators(rng.normal(size=(3, d)), rng.normal(size=(d, 3)))
    r = rng.normal(size=d)
    _, g, _ = roundtrip_loss(scal, r, delta_min=0.03, mu=1.0)

    def f(Vflat):
        s = ScalingOperators(scal.Lam, Vflat.reshape(scal.Vop.shape))
        return roundtrip_loss(s, r, delta_min=0.03, mu=1.0)[0]

    assert grad_check(f, scal.Vop.flatten().copy(), g["Vop"].flatten()) < TOL
