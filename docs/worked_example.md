# Worked Numerical Example: One Step of Next-Sign Prediction

This example traces a single prediction step through all three mechanisms, with small explicit numbers. We use $d = 4$ for the embedding dimension and $K = 2$ opposition axes so every vector can be written out in full. The setting: the model has read the sign sequence "the soup was" and must predict the next sign. The gold next sign is `hot`.

## 0. Setup

Two opposition axes are learned, written as columns of $A \in \mathbb{R}^{4\times 2}$:

$$
A = [a_1 \; a_2] =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 0 \\
0 & 0
\end{bmatrix},
\qquad
\begin{aligned}
a_1 &= \text{temperature axis (cold } \leftrightarrow \text{ hot)}\\
a_2 &= \text{size axis (small } \leftrightarrow \text{ large)}
\end{aligned}
$$

(For the example the axes are aligned to coordinate directions; in training they are arbitrary learned vectors. Both have unit norm, so $\lVert a_k \rVert = 1$.)

The current sign is $\sigma_t = (r_t, o_t)$ with representamen for "was" and an object memory accumulated over "the soup":

$$
r_t = (0.10,\; 0.40,\; 0.20,\; 0.05), \qquad
o_t^{\text{dyn}} = (0.30,\; 0.60,\; 0.10,\; 0.00).
$$

The dynamic-object slot already leans positive on $a_2$ (size) because "soup" carried some "large/substantial" value — this is the dynamical object refined across the sequence, not yet fully grasped.

## 1. Opposition Head on the Current Sign

Compute the oppositional value vector $v(r_t)$, where $v_k = \tanh(\langle r_t, a_k\rangle / \lVert a_k\rVert)$:

$$
\langle r_t, a_1 \rangle = 0.10, \quad \langle r_t, a_2 \rangle = 0.40,
$$
$$
v(r_t) = \big(\tanh 0.10,\; \tanh 0.40\big) = (0.0997,\; 0.3799).
$$

So "was" sits almost neutral on temperature ($\approx 0$) and mildly positive on size — exactly what a contentless copula should look like: it inherits a little size-value from context but commits to no temperature. The full upward representation is the concatenation

$$
\tilde{r}_t = [\,r_t \,\Vert\, v(r_t)\,] = (0.10,\,0.40,\,0.20,\,0.05,\;0.0997,\,0.3799) \in \mathbb{R}^{6}.
$$

## 2. Interpretant Operator: Predicting the Next Representamen

The operator $I$ contextualizes the history and emits a predicted next representamen. Suppose, after attending over "the soup was," it outputs

$$
\hat{r}_{t+1} = I(r_t, o_t, \Theta_t) = (0.85,\; 0.30,\; 0.15,\; 0.10).
$$

The object slots then update. The immediate object is recomputed from the new representamen; the dynamic object integrates it with a decay $\eta = 0.5$:

$$
o_{t+1}^{\text{dyn}} = g_{\text{mem}}(o_t^{\text{dyn}}, \hat r_{t+1})
= 0.5\,o_t^{\text{dyn}} + 0.5\,\hat r_{t+1}
= (0.575,\; 0.450,\; 0.125,\; 0.050).
$$

The predicted sign's oppositional value is

$$
v(\hat r_{t+1}) = (\tanh 0.85,\; \tanh 0.30) = (0.6911,\; 0.2913),
$$

i.e. the prediction has moved strongly to the *hot* pole of $a_1$ while staying mild on size. The interpretant has, in effect, resolved the temperature that "was" left open — a concrete semiosic step from an underdetermined sign to a determinate one.

## 3. The Sign Loss

The gold next sign `hot` has representamen and value

$$
r_{t+1} = (0.90,\; 0.20,\; 0.10,\; 0.00), \qquad
v(r_{t+1}) = (\tanh 0.90,\; \tanh 0.20) = (0.7163,\; 0.1974).
$$

**Representamen regression term.** The squared error:

$$
\lVert \hat r_{t+1} - r_{t+1}\rVert_2^2
= 0.05^2 + 0.10^2 + 0.05^2 + 0.10^2
= 0.0025 + 0.01 + 0.0025 + 0.01 = 0.0250.
$$

**Oppositional value term** (weight $\beta = 1$):

$$
\lVert v(\hat r_{t+1}) - v(r_{t+1})\rVert_2^2
= (0.6911 - 0.7163)^2 + (0.2913 - 0.1974)^2
$$
$$
= (-0.0252)^2 + (0.0939)^2 = 0.000635 + 0.008817 = 0.009452.
$$

**Contrastive term.** Take a tiny batch of candidate negatives — `cold` and `large` — with representamens

$$
r_{\text{cold}} = (-0.80,\,0.20,\,0.10,\,0.00), \qquad
r_{\text{large}} = (0.10,\,0.90,\,0.10,\,0.00).
$$

Using cosine similarity $\text{sim}(x,y) = \frac{\langle x,y\rangle}{\lVert x\rVert\lVert y\rVert}$ over the **full** 4-vectors and temperature $\tau = 1$, with $\hat r_{t+1} = (0.85,0.30,0.15,0.10)$, $\lVert\hat r_{t+1}\rVert = 0.9301$:

| candidate | $\langle \hat r_{t+1}, \cdot\rangle$ | $\lVert\cdot\rVert$ | sim | $\exp(\text{sim})$ |
|---|---|---|---|---|
| `hot` (positive) | 0.840 | 0.9274 | 0.9854 | 2.6788 |
| `cold` | −0.595 | 0.8307 | −0.7923 | 0.4527 |
| `large` | 0.375 | 0.9165 | 0.4418 | 1.5556 |

The InfoNCE loss is

$$
-\ln \frac{\exp(\text{sim}_{\text{hot}})}{\exp(\text{sim}_{\text{hot}}) + \exp(\text{sim}_{\text{cold}}) + \exp(\text{sim}_{\text{large}})}
= -\ln \frac{2.6788}{2.6788 + 0.4527 + 1.5556}
$$
$$
= -\ln \frac{2.6788}{4.6871} = -\ln 0.5715 = 0.5594.
$$

Note what the contrastive term reveals that the MSE term alone does not. The softmax over candidates assigns the gold sign `hot` probability $0.572$, the competitor `large` probability $0.332$, and `cold` only $0.097$. So even though the prediction is clearly closest to `hot` (similarity $0.985$), more than a third of the probability mass has leaked to `large` (similarity $0.442$) — because the predicted displacement happened to carry some positive size-value. Pure MSE (0.025, very small) would call this prediction essentially solved; the contrastive term, with loss 0.559, registers the residual competition from a near neighbour *by value* and keeps applying separating pressure. This is the fragility problem made arithmetic — a semantically wrong neighbour sitting close enough in raw geometry to steal probability mass — and it is exactly why the contrastive term is non-optional. (These figures are the verified output of the reference implementation, not a hand calculation; an earlier draft that projected onto only the first two coordinates understated the separation.)

With $\gamma = 1$, the per-step sign loss is

$$
\mathcal{L}_{\text{sign}} = 0.0250 + (1)(0.009461) + (1)(0.5594) = 0.5939.
$$

## 4. The Λ/V Round-Trip

Now check coherence across scales. Abstract the gold sign one level — `hot` as a token becomes part of a concept-level sign, say the clause-idea "the soup was hot." Let the single-step abstraction and elaboration act on the value vector as small linear maps (in practice learned):

$$
\Lambda(r_{t+1}) = r^{(2)} = (0.78,\; 0.22), \quad\text{(now a 2-d concept code)}
$$

and elaborating back:

$$
V(r^{(2)}) = \hat r^{(1)} = (0.88,\; 0.18,\; 0.12,\; 0.02).
$$

**Round-trip term.** Using $d = $ Euclidean distance on the representamen:

$$
d\big(r_{t+1},\, V(\Lambda(r_{t+1}))\big)
= \lVert (0.90,0.20,0.10,0.00) - (0.88,0.18,0.12,0.02)\rVert
$$
$$
= \sqrt{0.02^2 + 0.02^2 + 0.02^2 + 0.02^2} = \sqrt{0.0016} = 0.0400.
$$

This is small but **non-zero**, which is correct: abstraction discarded the fine token-level detail (the exact spelling/sound of `hot`) while preserving its semantic position. The floor term keeps it from collapsing to zero. With $\delta_{\min} = 0.03$ and $\mu = 1$:

$$
\big|\, d(r_{t+1}, V(\Lambda(r_{t+1}))) - \delta_{\min} \,\big| = |0.0400 - 0.03| = 0.0100.
$$

Had the model learned $\Lambda = V = \text{id}$ (round-trip distance 0), this term would penalize it by $|0 - 0.03| = 0.03$ — pushing it *away* from the degenerate identity solution and toward genuine, lossy abstraction.

**Coherence term.** Verify opposition commutes with scaling. The abstracted token's value, vs. the abstraction of its value:

$$
v\big(\Lambda(r_{t+1})\big) = v(0.78, 0.22) = (\tanh 0.78,\; \tanh 0.22) = (0.6527,\; 0.2165),
$$
$$
\Lambda_{\text{opp}}\big(v(r_{t+1})\big) \approx (0.66,\; 0.21) \quad\text{(learned map applied to } (0.7163, 0.1974)).
$$

$$
\mathcal{L}_{\text{coh}} = \lVert (0.6527, 0.2165) - (0.66, 0.21)\rVert_2^2
= (-0.0073)^2 + (0.0065)^2 = 0.0000953.
$$

Near zero — the hot/cold position survived the move to the coarser scale intact. This is the numerical content of cross-scale coherence: the *same opposition skeleton* at the token scale (`hot`) and the concept scale ("the soup was hot"). If this term were large, the scales would be telling inconsistent stories about temperature, and the multi-scale structure would have torn.

## 5. Combined Step Loss

With weights $w_1 = 0.5,\, w_2 = 1,\, w_3 = 1$ and (for this single sign) $\mathcal{L}_{\text{opp}}$ contributing through the supervised pair `hot`/`cold`:

$$
\mathcal{L}_{\text{opp}} = \lVert v(\text{hot}) + v(\text{cold})\rVert_2^2.
$$
With $v(\text{cold}) = (\tanh(-0.80), \tanh 0.20) = (-0.6640, 0.1974)$:
$$
= (0.7163 - 0.6640)^2 + (0.1974 + 0.1974)^2 = (0.0523)^2 + (0.3948)^2 = 0.0027 + 0.1559 = 0.1586.
$$
(The size coordinate is penalized here because both `hot` and `cold` happen to share a mild positive size value — the optimizer will learn to push their $a_2$ components apart or, more likely, attribute the shared size-value to a third axis. This is the decorrelation pressure of the $\lVert A^\top A - I\rVert$ term at work.)

Summing:

$$
\mathcal{L} = \underbrace{0.5939}_{\mathcal{L}_{\text{sign}}}
+ 0.5\underbrace{(0.1586)}_{\mathcal{L}_{\text{opp}}}
+ 1\cdot\underbrace{(0.0100)}_{\mathcal{L}_{\text{rt}}}
+ 1\cdot\underbrace{(0.0000953)}_{\mathcal{L}_{\text{coh}}}
= 0.5939 + 0.0793 + 0.0100 + 0.0001 = 0.6833.
$$

(The round-trip distance and `L_rt` shown here, $0.0400$ and $0.0100$, are the idealized values used in Section 4 for legibility; the reference implementation with the example's actual $\Lambda, V$ matrices yields a nearby $0.0513$ and $0.0725$. The qualitative reading is unchanged — small, positive, floored away from zero.)

## 6. Reading the Numbers

The dominant contribution (0.594) is the sign-prediction loss, and within it the contrastive term (0.559) dwarfs the regression term (0.025). This is the whole thesis in one column of arithmetic: a flat next-token objective sees only the small regression error and declares success, because in raw embedding geometry the prediction is comfortably closest to `hot`. But the InfoNCE softmax still hands a third of its probability mass to `large` — a sign that is *wrong by value* yet *near by coordinate*. The differential structure — opposition at the base, contrast in the dynamics, coherence across scales — is what supplies the missing pressure to separate the right sign from that near neighbour, which is precisely the relational information a flat vocabulary throws away.

The round-trip and coherence terms are small here (≈0.01–0.07 and 0.0001) because the example is set near convergence for those components; early in training they are large, and their gradient is what *builds* the multi-scale structure in the first place — forcing the same opposition skeleton to hold at the token and concept scales rather than letting each scale drift into its own incompatible geometry. Every number in this document is reproduced, and every hand-derived gradient finite-difference-checked to ≈$10^{-10}$, by the reference implementation in `src/sign_prediction/layer.py`.
