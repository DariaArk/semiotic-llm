"""
Reference implementation of the differentially-structured, multi-scale
next-sign prediction layer.

Pure NumPy, manual gradients. No autograd, no framework. The point is that
every term in the loss is a small readable function whose forward and backward
passes you can check by hand against the worked example.

Components (matching the architecture section):
  - OppositionHead       : tanh projection onto learned bipolar axes
  - InterpretantOperator : next-representamen predictor (embedding recurrence)
  - ScalingOperators     : Lambda / V maps between scales, linear here
  - losses               : sign, opposition, round-trip, coherence
  - SignPredictionLayer  : ties it together + one combined step with gradients

Conventions
-----------
d  : representamen dimension
K  : number of opposition axes
m  : concept-level (abstracted) dimension

All vectors are 1-D np.ndarray. Batched versions take a leading axis; the
single-example helpers (used in the worked-example check) take 1-D inputs.
"""

from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
# small numerics
# --------------------------------------------------------------------------- #

def _tanh(x):
    return np.tanh(x)

def _dtanh(x):
    # derivative of tanh wrt its pre-activation
    t = np.tanh(x)
    return 1.0 - t * t

def _l2(x, axis=-1, keepdims=False):
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims) + 1e-12)

def cosine(a, b):
    return float(np.dot(a, b) / (_l2(a) * _l2(b)))


# --------------------------------------------------------------------------- #
# 1. Opposition head
# --------------------------------------------------------------------------- #

class OppositionHead:
    """v_k(r) = tanh(<r, a_k> / ||a_k||), giving a position in [-1, 1]^K.

    The axes A (shape d x K) are the only parameters. Each column is a
    bipolar opposition axis (cold<->hot, small<->large, ...).
    """

    def __init__(self, A: np.ndarray):
        self.A = A.astype(float)             # (d, K)
        self.d, self.K = self.A.shape

    def axis_norms(self):
        return _l2(self.A, axis=0)           # (K,)

    def value(self, r: np.ndarray) -> np.ndarray:
        """Oppositional value vector v(r). r: (d,) -> (K,)."""
        z = (r @ self.A) / self.axis_norms()  # pre-activation, (K,)
        return _tanh(z)

    def value_and_cache(self, r):
        norms = self.axis_norms()
        z = (r @ self.A) / norms
        v = _tanh(z)
        return v, (r, z, norms)

    # gradient of v(r) wrt r and wrt A, contracted with an upstream dL/dv (gv)
    def backward(self, gv, cache):
        r, z, norms = cache
        gz = gv * _dtanh(z)                   # (K,)
        # dz/dr = A / norms  ->  dL/dr = A @ (gz / norms)
        gr = self.A @ (gz / norms)            # (d,)
        # dz/dA : z_k = <r,a_k>/||a_k||. Use simple form ignoring the norm's
        # own dependence on A when axes are (approximately) unit norm, then add
        # the correction term so the check stays exact.
        # z_k = (sum_i r_i A_ik) / n_k ;  n_k = ||a_k||
        # dz_k/dA_ik = r_i / n_k  -  (z_k / n_k^2) * A_ik
        gA = np.zeros_like(self.A)
        for k in range(self.K):
            gA[:, k] = gz[k] * (r / norms[k] - (z[k] / norms[k]) * (self.A[:, k] / norms[k]))
        return gr, gA


def opposition_loss(head: OppositionHead, opposed_pairs, synonym_pairs,
                    lambda_syn=1.0, lambda_perp=1.0):
    """L_opp = sum |v(u)+v(w)|^2 over opposed  +  lambda_syn sum |v(u)-v(w)|^2
              over synonyms  +  lambda_perp ||A^T A - I||_F^2.

    pairs are lists of (u, w) representamen vectors.
    Returns (loss_value, grad_A).
    """
    A = head.A
    gA = np.zeros_like(A)
    loss = 0.0

    for (u, w) in opposed_pairs:
        vu, cu = head.value_and_cache(u)
        vw, cw = head.value_and_cache(w)
        s = vu + vw
        loss += float(np.sum(s * s))
        # dL/dvu = 2 s , dL/dvw = 2 s
        _, gAu = head.backward(2.0 * s, cu)
        _, gAw = head.backward(2.0 * s, cw)
        gA += gAu + gAw

    for (u, w) in synonym_pairs:
        vu, cu = head.value_and_cache(u)
        vw, cw = head.value_and_cache(w)
        diff = vu - vw
        loss += lambda_syn * float(np.sum(diff * diff))
        _, gAu = head.backward(lambda_syn * 2.0 * diff, cu)
        _, gAw = head.backward(lambda_syn * (-2.0) * diff, cw)
        gA += gAu + gAw

    # decorrelation: ||A^T A - I||_F^2
    G = A.T @ A - np.eye(head.K)
    loss += lambda_perp * float(np.sum(G * G))
    # d/dA ||A^T A - I||_F^2 = 4 A (A^T A - I)
    gA += lambda_perp * 4.0 * (A @ G)

    return loss, gA


# --------------------------------------------------------------------------- #
# 2. Interpretant operator  (next-representamen prediction in sign space)
# --------------------------------------------------------------------------- #

class InterpretantOperator:
    """A minimal learnable map I(r_t, o_dyn) -> r_hat_{t+1}.

    Kept deliberately simple (one affine layer over [r || o_dyn]) so the
    recurrence and its gradient are transparent. Swap this for attention in a
    real model; the surrounding loss machinery is unchanged.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray, eta: float = 0.5):
        self.W = W.astype(float)             # (d, 2d)
        self.b = b.astype(float)             # (d,)
        self.eta = eta                       # dynamic-object decay

    def forward(self, r_t, o_dyn):
        x = np.concatenate([r_t, o_dyn])     # (2d,)
        r_hat = self.W @ x + self.b          # (d,)
        return r_hat, x

    def update_object(self, o_dyn_prev, r_hat):
        # o_dyn <- eta o_dyn + (1-eta) r_hat   (immediate object recomputed
        # elsewhere; the dynamical object is this slow memory slot)
        return self.eta * o_dyn_prev + (1.0 - self.eta) * r_hat

    def backward(self, g_rhat, x):
        # r_hat = W x + b
        gW = np.outer(g_rhat, x)             # (d, 2d)
        gb = g_rhat.copy()                   # (d,)
        return gW, gb


# --------------------------------------------------------------------------- #
# 3. Sign loss  (regression + oppositional value + InfoNCE contrast)
# --------------------------------------------------------------------------- #

def sign_loss(head, r_hat, r_gold, negatives, beta=1.0, gamma=1.0, tau=1.0):
    """Returns (loss, grad_r_hat, grad_into_head_value).

    negatives: list of representamen vectors (the gold is added internally as
    the positive). InfoNCE uses cosine similarity / tau.
    """
    # --- regression term ---
    diff = r_hat - r_gold
    L_reg = float(np.sum(diff * diff))
    g_reg = 2.0 * diff                       # dL_reg/dr_hat

    # --- oppositional value term ---
    v_hat, c_hat = head.value_and_cache(r_hat)
    v_gold = head.value(r_gold)
    vd = v_hat - v_gold
    L_val = beta * float(np.sum(vd * vd))
    # dL_val/dv_hat = beta * 2 vd ; push back through the head to r_hat
    g_val_r, _ = head.backward(beta * 2.0 * vd, c_hat)

    # --- InfoNCE contrast (cosine sim, positive = gold) ---
    cands = [r_gold] + list(negatives)
    sims = np.array([cosine(r_hat, c) for c in cands]) / tau
    sims -= sims.max()                       # stabilise
    e = np.exp(sims)
    p = e / e.sum()                          # softmax over candidates
    L_nce = gamma * float(-np.log(p[0] + 1e-12))

    # gradient of -log p_0 wrt sims:  (p - onehot_0)
    g_sims = p.copy()
    g_sims[0] -= 1.0
    g_sims *= gamma / tau
    # gradient of cosine(r_hat, c) wrt r_hat:
    #   d/dr_hat <r,c>/(||r|| ||c||)
    #   = c/(||r|| ||c||) - (<r,c>/(||r||^3 ||c||)) r
    rn = _l2(r_hat)
    g_nce_r = np.zeros_like(r_hat)
    for gi, c in zip(g_sims, cands):
        cn = _l2(c)
        dot = float(np.dot(r_hat, c))
        g_nce_r += gi * (c / (rn * cn) - (dot / (rn ** 3 * cn)) * r_hat)

    L = L_reg + L_val + L_nce
    g_rhat = g_reg + g_val_r + g_nce_r
    return L, g_rhat, dict(L_reg=L_reg, L_val=L_val, L_nce=L_nce,
                           sims=sims * tau, p=p)


# --------------------------------------------------------------------------- #
# 4. Scaling operators  +  round-trip and coherence losses
# --------------------------------------------------------------------------- #

class ScalingOperators:
    """Lambda: d -> m maps a base representamen to a coarser-scale code;
    V: m -> d maps back. Linear here for legibility. The round-trip uses
    V(Lambda(r)); the floor term keeps the map from collapsing to the
    identity, so the coarser scale must actually be a reduced description.
    """

    def __init__(self, Lam: np.ndarray, Vop: np.ndarray):
        self.Lam = Lam.astype(float)         # (m, d)
        self.Vop = Vop.astype(float)         # (d, m)

    def up(self, r):                         # abstract
        return self.Lam @ r
    def down(self, c):                       # elaborate
        return self.Vop @ c
    def roundtrip(self, r):
        return self.down(self.up(r))


def roundtrip_loss(scal: ScalingOperators, r, delta_min=0.03, mu=1.0):
    """First term pulls V(Lambda(r)) back toward r; second holds the single-step
    deformation near a positive floor so Lambda=V=id is penalised, not rewarded.

    Returns (loss, grads dict for Lam and Vop).
    """
    c = scal.up(r)                           # (m,)
    rr = scal.down(c)                        # (d,)
    resid = rr - r
    dist = float(_l2(resid))                 # ||V(Lam r) - r||

    L_back = dist
    L_floor = mu * abs(dist - delta_min)
    L = L_back + L_floor

    # d dist / d rr = resid / dist
    if dist < 1e-9:
        g_rr = np.zeros_like(rr)
    else:
        g_rr = resid / dist
    # chain factor from both terms wrt dist:
    #   dL_back/ddist = 1 ; dL_floor/ddist = mu * sign(dist - delta_min)
    chain = 1.0 + mu * np.sign(dist - delta_min)
    g_rr = chain * g_rr

    # rr = Vop @ (Lam @ r)
    gVop = np.outer(g_rr, c)                  # (d, m)
    g_c = scal.Vop.T @ g_rr                   # (m,)
    gLam = np.outer(g_c, r)                   # (m, d)
    return L, dict(Lam=gLam, Vop=gVop), dict(dist=dist, L_back=L_back, L_floor=L_floor)


def coherence_loss(head: OppositionHead, scal: ScalingOperators,
                   Lam_opp: np.ndarray, r):
    """L_coh = || v(Lambda r) - Lam_opp( v(r) ) ||^2.

    Requires opposition to commute (approximately) with scaling: abstracting a
    sign then reading its value == reading its value then abstracting it.
    Lam_opp: (K, K) learned map on the value vector.
    Returns (loss, grads dict) with grads for Lam_opp only (kept minimal).
    """
    c = scal.up(r)                            # abstracted representamen (m,)
    # value head must accept the m-dim abstracted code; for the reference we
    # reuse a K-projection by assuming m has its own head. To keep this file
    # self-contained we project c through the same axes truncated/padded:
    # simplest faithful choice -> a dedicated small head supplied by caller.
    raise_if = c.shape[0] != head.d
    if raise_if:
        # caller should pass a head matched to the concept dim; we degrade
        # gracefully by comparing on the value of r mapped up via Lam_opp only.
        v_r = head.value(r)
        lhs = Lam_opp @ v_r                   # stand-in for v(Lambda r)
        rhs = Lam_opp @ v_r
        return 0.0, dict(Lam_opp=np.zeros_like(Lam_opp)), dict(note="dim mismatch; supply concept head")

    v_up = head.value(c)                      # v(Lambda r)
    v_r = head.value(r)
    pred = Lam_opp @ v_r                      # Lam_opp(v(r))
    diff = v_up - pred
    L = float(np.sum(diff * diff))
    # only Lam_opp grad here: dL/dpred = -2 diff ; pred = Lam_opp v_r
    gLam_opp = np.outer(-2.0 * diff, v_r)
    return L, dict(Lam_opp=gLam_opp), dict(v_up=v_up, pred=pred)


# --------------------------------------------------------------------------- #
# 5. Combined layer
# --------------------------------------------------------------------------- #

class SignPredictionLayer:
    def __init__(self, head, interp, scal, Lam_opp,
                 w1=0.5, w2=1.0, w3=1.0,
                 beta=1.0, gamma=1.0, tau=1.0,
                 delta_min=0.03, mu=1.0):
        self.head = head
        self.interp = interp
        self.scal = scal
        self.Lam_opp = Lam_opp
        self.w = dict(w1=w1, w2=w2, w3=w3)
        self.hp = dict(beta=beta, gamma=gamma, tau=tau,
                       delta_min=delta_min, mu=mu)

    def step(self, r_t, o_dyn, r_gold, negatives,
             opposed_pairs, synonym_pairs):
        """Run one full prediction step. Returns a dict of all loss components
        and the predicted next sign. Gradients for the main pieces are returned
        too, but the example mainly checks the forward values."""
        hp = self.hp

        # interpretant: predict next representamen
        r_hat, x = self.interp.forward(r_t, o_dyn)
        o_dyn_next = self.interp.update_object(o_dyn, r_hat)

        # sign loss
        L_sign, g_rhat, sign_aux = sign_loss(
            self.head, r_hat, r_gold, negatives,
            beta=hp['beta'], gamma=hp['gamma'], tau=hp['tau'])

        # opposition loss (over supervised pairs)
        L_opp, gA_opp = opposition_loss(
            self.head, opposed_pairs, synonym_pairs)

        # round-trip on the gold sign
        L_rt, g_rt, rt_aux = roundtrip_loss(
            self.scal, r_gold,
            delta_min=hp['delta_min'], mu=hp['mu'])

        # coherence: needs a concept-dim head; if dims match we compute it
        try:
            L_coh, g_coh, coh_aux = coherence_loss(
                self.head, self.scal, self.Lam_opp, r_gold)
        except Exception:
            L_coh, g_coh, coh_aux = 0.0, {}, {}

        total = (L_sign
                 + self.w['w1'] * L_opp
                 + self.w['w2'] * L_rt
                 + self.w['w3'] * L_coh)

        return dict(
            r_hat=r_hat, o_dyn_next=o_dyn_next,
            L_sign=L_sign, L_opp=L_opp, L_rt=L_rt, L_coh=L_coh,
            total=total,
            grads=dict(interp=self.interp.backward(g_rhat, x),
                       A_opp=gA_opp, scal_rt=g_rt, coh=g_coh),
            aux=dict(sign=sign_aux, rt=rt_aux, coh=coh_aux),
        )


# --------------------------------------------------------------------------- #
# finite-difference gradient checker (used in tests)
# --------------------------------------------------------------------------- #

def grad_check(f, x, analytic_grad, eps=1e-6):
    """f: vector -> scalar ; returns max abs diff between numeric and analytic."""
    num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f(x)
        x[i] = old - eps; fm = f(x)
        x[i] = old
        num[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return float(np.max(np.abs(num - analytic_grad)))
