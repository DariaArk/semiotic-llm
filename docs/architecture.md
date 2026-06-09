# Architecture: Next-Sign Prediction in a Differentially-Structured, Multi-Scale Embedding Space

## 1. The premise

A standard language model maximizes the probability of the next token over a flat vocabulary $\mathcal{V}$:

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \ln \hat{P}\big(s_{t+1} \mid s_t, \dots, s_1\big), \qquad s_i \in \{0,1,\dots,|\mathcal{V}|-1\}.
$$

Two design commitments are built into this objective:

**Granularity.** The token is a sub-lexical unit chosen for tokenizer convenience, beneath the smallest unit that carries meaning. Predicting it commits the model to reconstructing significance one fragment at a time.

**Relational structure.** The vocabulary is an *unordered set*: token $i$ and token $j$ stand in no intrinsic relation prior to whatever the embedding matrix happens to learn. Meaning, however, is differential — the value of a unit is constituted in part by its *opposition* to other units (Saussure, Eco). The flat softmax encodes no opposition as an architectural fact; it survives only as an accident of learned geometry.

This repository takes the position that semiotics is useful here as **a theory of what the statistics should be computed over** — not a replacement for statistical learning. Everything below is a differentiable loss minimized by gradient descent. What changes is the *target* (from token to **sign**) and the *space* (from a flat vocabulary to a differentially-structured, multi-scale embedding space).

We develop three components: the **opposition head** (Section 3), the **interpretant operator recurrence** (Section 4), and the **Λ/V round-trip loss** (Section 5).

## 2. The sign as node, semiosis as edge

A Peircean sign is a triad — representamen, object, interpretant — but the interpretant is *itself a sign*: "the interpretant is not the interpreter," but a further sign that refers the representamen onward. The triad is therefore recursive, and cannot be encoded as a fixed three-slot tensor. We treat the sign as a **node** and semiosis as an **operator that emits the next node**.

A sign at position $t$ is the pair

$$
\sigma_t = (r_t, o_t), \qquad r_t \in \mathbb{R}^d,\; o_t \in \mathbb{R}^d,
$$

with $r_t$ the **representamen** (a contextual embedding) and $o_t$ the **object**, decomposed into immediate and dynamic parts:

$$
o_t^{\text{imm}} = f_{\text{obj}}(r_t, c_t),\qquad
o_t^{\text{dyn}} = g_{\text{mem}}\!\big(o_{t-1}^{\text{dyn}}, r_t\big).
$$

The interpretant is **not stored** — it is produced by the operator $I$ of Section 4, which consumes $(r_t, o_t)$ and emits the next representamen. The recursion lives in the dynamics, not the data layout.

## 3. The opposition head: differential value as a geometric relation

We learn $K$ bipolar **opposition axes** $\{a_1,\dots,a_K\}$, $a_k\in\mathbb{R}^d$, and project each representamen onto them. The **oppositional value vector** $v(r)\in[-1,1]^K$ is

$$
v_k(r) = \tanh\!\left(\frac{\langle r, a_k\rangle}{\lVert a_k\rVert}\right), \qquad k=1,\dots,K.
$$

Each coordinate places the sign on a spectrum between two opposed poles ($+1$ / $-1$), rather than at one entry of an unordered set. The full upward representation is the concatenation $\tilde r = [\,r \,\Vert\, v(r)\,] \in \mathbb{R}^{d+K}$.

**Opposition loss.** Given opposed pairs $\mathcal{O}$ and near-synonym pairs $\mathcal{N}$, with $A=[a_1\cdots a_K]$:

$$
\mathcal{L}_{\text{opp}}
= \sum_{(u,w)\in\mathcal{O}} \lVert v(u)+v(w)\rVert_2^2
\;+\; \lambda_{\text{syn}}\!\!\sum_{(u,w)\in\mathcal{N}} \lVert v(u)-v(w)\rVert_2^2
\;+\; \lambda_{\perp}\,\lVert A^\top A - I_K\rVert_F^2.
$$

The first term places opposed pairs at $v$ and $-v$ (summing to zero) — opposition as an explicit geometric relation. The third decorrelates the axes so each captures a distinct dimension of contrast. The head is a thin trainable layer mountable on an existing encoder.

## 4. The interpretant operator recurrence

The operator $I$ produces the next representamen in embedding space:

$$
\hat r_{t+1} = I(r_t, o_t, \Theta_t), \qquad \Theta_t = \{\sigma_1,\dots,\sigma_t\},
$$

after which the object slots update. Generation is the orbit $r_t \mapsto \hat r_{t+1}\mapsto\cdots$; each step is an interpretant of the last. $I$ may be realized by the same attention machinery used in current models, but its output lives in the embedding space, not over a discrete vocabulary.

**Sign loss.** Against the gold next sign $\sigma_{t+1}=(r_{t+1}, o_{t+1})$:

$$
\mathcal{L}_{\text{sign}}
= \lVert \hat r_{t+1} - r_{t+1}\rVert_2^2
\;+\; \beta\,\lVert v(\hat r_{t+1}) - v(r_{t+1})\rVert_2^2
\;-\; \gamma\,\ln\frac{\exp(\text{sim}(\hat r_{t+1}, r_{t+1})/\tau)}{\sum_{r'\in\mathcal{B}}\exp(\text{sim}(\hat r_{t+1}, r')/\tau)}.
$$

The contrastive (InfoNCE) term is essential, not optional. Pure MSE in an embedding space is fragile: a small displacement can sit close to a semantically distant neighbour. The contrastive term requires the prediction to be closer to the true next sign than to its alternatives — the dynamic, sign-space analogue of the opposition principle from Section 3. The worked example shows this with explicit numbers.

## 5. The Λ/V round-trip loss: the same skeleton at every scale

A single base level is not enough. We use two maps between scales: $\Lambda:\mathbb{R}^d\to\mathbb{R}^m$ to a coarser-scale code, and $V:\mathbb{R}^m\to\mathbb{R}^d$ back. The requirement is that going up and back down approximately returns the original, while the coarser scale remains a genuinely *reduced* description rather than a copy.

**Round-trip loss**, with a positive floor $\delta_{\min}$:

$$
\mathcal{L}_{\text{rt}}
= \mathbb{E}\big[\, d\big(r,\, V(\Lambda(r))\big) \,\big]
\;+\; \mu\,\mathbb{E}\big[\, \big| d\big(r, V(\Lambda(r))\big) - \delta_{\min}\big| \,\big].
$$

The first term pulls the round trip back toward its origin; the second holds the single-step deformation near a positive floor. Without the floor, the trivial solution $\Lambda=V=\text{id}$ minimizes the first term and the scale structure degenerates into copying.

**Cross-scale coherence.** Opposition must remain meaningful at every scale, so we require it to commute (approximately) with scaling:

$$
\mathcal{L}_{\text{coh}} = \sum_k \big\lVert v\big(\Lambda^k(r)\big) - \Lambda^k_{\text{opp}}\big(v(r)\big)\big\rVert_2^2.
$$

This makes "the same opposition skeleton at every scale" a checkable equation: the differential structure visible at the base must align with the differential structure visible at the coarser scale. It is the constraint that most distinguishes this design from a plain stack of independently-trained levels.

## 6. The combined objective

$$
\mathcal{L} = \mathcal{L}_{\text{sign}}
\;+\; w_1\,\mathcal{L}_{\text{opp}}
\;+\; w_2 \sum_k \mathcal{L}_{\text{rt}}^{(k)}
\;+\; w_3\,\mathcal{L}_{\text{coh}}.
$$

Each term carries a distinct commitment: $\mathcal{L}_{\text{sign}}$ predicts the next *sign* rather than the next token (granularity); $\mathcal{L}_{\text{opp}}$ makes the base layer *differential* rather than an unordered set (relational structure at one scale); $\mathcal{L}_{\text{rt}}+\mathcal{L}_{\text{coh}}$ tie the scales together (relational structure across scales). All four are implemented in `src/sign_prediction/layer.py` with hand-derived gradients verified to ≈`1e-10`.

## 7. Scope

This is a reference implementation, not a trained model. It establishes that the design is coherent, internally consistent, and trainable — and it makes the central mechanism (contrastive sign prediction separating a near-by-coordinate, wrong-by-value neighbour) concrete and reproducible. The two supervision sets for the opposition head and the value of the floor $\delta_{\min}$ are open practical questions; see `docs/roadmap.md`.
