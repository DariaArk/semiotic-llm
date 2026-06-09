"""One full next-sign prediction step, with printed output.

    python examples/worked_example.py
"""

import numpy as np
from sign_prediction import (
    OppositionHead, ScalingOperators, sign_loss, roundtrip_loss, cosine,
)

np.set_printoptions(precision=4, suppress=True)
bar = lambda: print("-" * 60)

# Two opposition axes: temperature (cold<->hot), size (small<->large).
A = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
head = OppositionHead(A)

r_t     = np.array([0.10, 0.40, 0.20, 0.05])   # "was"
r_hat   = np.array([0.85, 0.30, 0.15, 0.10])   # predicted next representamen
r_gold  = np.array([0.90, 0.20, 0.10, 0.00])   # gold sign "hot"
r_cold  = np.array([-0.80, 0.20, 0.10, 0.00])
r_large = np.array([0.10, 0.90, 0.10, 0.00])

print("OPPOSITION VALUES  v(r) = tanh(<r, a_k> / ||a_k||)")
bar()
print("v('was')  =", head.value(r_t))
print("v(pred)   =", head.value(r_hat))
print("v('hot')  =", head.value(r_gold))

print("\nSIGN LOSS")
bar()
L, _, aux = sign_loss(head, r_hat, r_gold, negatives=[r_cold, r_large])
print(f"L_reg = {aux['L_reg']:.4f}   (representamen regression)")
print(f"L_val = {aux['L_val']:.6f} (oppositional value match)")
print(f"L_nce = {aux['L_nce']:.4f}   (InfoNCE contrast)")
print(f"L_sign = {L:.4f}")

print("\nINFONCE PROBABILITIES")
bar()
for lab, pi, c in zip(["hot (gold)", "cold", "large"],
                      aux["p"], [r_gold, r_cold, r_large]):
    print(f"  {lab:12s} sim={cosine(r_hat, c):+.4f}  p={pi:.4f}")
print("  gold wins, yet 'large' (close by coordinate, wrong by value) takes")
print("  ~1/3 of the mass. A flat-vocabulary / pure-MSE target never sees this.")

print("\nROUND-TRIP BETWEEN SCALES")
bar()
scal = ScalingOperators(np.eye(4)[:2], np.eye(4)[:, :2])
L_rt, _, rt = roundtrip_loss(scal, r_gold, delta_min=0.03, mu=1.0)
print(f"identity-like Lambda/V -> dist={rt['dist']:.4f}, L_rt={L_rt:.4f}")
print("  the floor term penalises collapse to identity (0.03), forcing the")
print("  coarser scale to be a genuinely reduced description.")
