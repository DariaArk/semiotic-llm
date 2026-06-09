"""Regression tests pinning the worked-example numbers (docs/worked_example.md).

    pytest tests/test_worked_example.py -v
"""

import numpy as np
import pytest

from sign_prediction import (
    OppositionHead, ScalingOperators,
    sign_loss, opposition_loss, roundtrip_loss, cosine,
)


@pytest.fixture
def setup():
    A = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    return dict(
        head=OppositionHead(A),
        r_t=np.array([0.10, 0.40, 0.20, 0.05]),
        r_hat=np.array([0.85, 0.30, 0.15, 0.10]),
        r_gold=np.array([0.90, 0.20, 0.10, 0.00]),
        r_cold=np.array([-0.80, 0.20, 0.10, 0.00]),
        r_large=np.array([0.10, 0.90, 0.10, 0.00]),
    )


def test_opposition_values(setup):
    h = setup["head"]
    assert np.allclose(h.value(setup["r_t"]),    [0.0997, 0.3799], atol=1e-4)
    assert np.allclose(h.value(setup["r_hat"]),  [0.6911, 0.2913], atol=1e-4)
    assert np.allclose(h.value(setup["r_gold"]), [0.7163, 0.1974], atol=1e-4)


def test_cosine_similarities(setup):
    rh = setup["r_hat"]
    assert cosine(rh, setup["r_gold"])  == pytest.approx(0.9854, abs=1e-3)
    assert cosine(rh, setup["r_cold"])  == pytest.approx(-0.7923, abs=1e-3)
    assert cosine(rh, setup["r_large"]) == pytest.approx(0.4418, abs=1e-3)


def test_sign_loss_components(setup):
    L, _, aux = sign_loss(
        setup["head"], setup["r_hat"], setup["r_gold"],
        negatives=[setup["r_cold"], setup["r_large"]],
    )
    assert aux["L_reg"] == pytest.approx(0.0250, abs=1e-4)
    assert aux["L_val"] == pytest.approx(0.009461, abs=1e-5)
    assert aux["L_nce"] == pytest.approx(0.5594, abs=1e-3)
    assert L == pytest.approx(0.5939, abs=1e-3)


def test_infonce_probabilities(setup):
    _, _, aux = sign_loss(
        setup["head"], setup["r_hat"], setup["r_gold"],
        negatives=[setup["r_cold"], setup["r_large"]],
    )
    p = aux["p"]
    assert p[0] == pytest.approx(0.5715, abs=1e-3)   # hot (gold)
    assert p[2] == pytest.approx(0.3319, abs=1e-3)   # large (near-miss by value)


def test_opposition_pair_loss(setup):
    L, _ = opposition_loss(
        setup["head"],
        opposed_pairs=[(setup["r_gold"], setup["r_cold"])],
        synonym_pairs=[], lambda_perp=0.0,
    )
    assert L == pytest.approx(0.1586, abs=1e-3)


def test_roundtrip_floor_penalizes_identity():
    d = 4
    scal = ScalingOperators(np.eye(d)[:d], np.eye(d))
    r = np.array([0.9, 0.2, 0.1, 0.0])
    L, _, aux = roundtrip_loss(scal, r, delta_min=0.03, mu=1.0)
    assert aux["dist"] == pytest.approx(0.0, abs=1e-6)
    assert L == pytest.approx(0.03, abs=1e-6)
