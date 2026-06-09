Predicting the next sign in a differentially-structured, multi-scale embedding space — a reference implementation with hand-derived, gradient-checked losses.

Next-token prediction over a flat vocabulary commits to the wrong granularity (sub-meaning fragments) and the wrong relational structure (an unordered set). A semiotic reading suggests predicting the next sign in a differentially-structured, multi-scale embedding space instead.
Semiotics here is treated as a theory of what the statistics should be computed over — not a replacement for statistical learning. Every clause of the sentence above is turned into a differentiable loss whose gradient is derived by hand and finite-difference-checked to ≈1e-10.

pip install -e ".[test]"
pytest -v                       # 12 tests: 6 gradient checks, 6 worked-example regressions
python examples/worked_example.py
 
The three components

Component	Idea	What the loss enforces
OppositionHead	Saussurean differential value as an architectural constraint	a sign’s value is its position on learned bipolar axes — a geometric relation, not an index; opposed pairs land at v and -v
InterpretantOperator	Peirce’s recursion as a next-embedding recurrence	the next representamen is predicted in embedding space; the interpretant is produced, not stored
ScalingOperators (Λ/V)	the same skeleton at every scale, as a checkable equation	V(Λr) ≈ r with a floor so the coarser scale is a genuine reduction; opposition commutes with scaling
Key design decision: a sign is a node, semiosis is an edge. The Peircean triad cannot be a fixed three-slot tensor, because the interpretant is itself a sign (recursive). So the recursion lives in the dynamics (InterpretantOperator), not the data layout. Full derivation and equations: docs/architecture.md.

The combined objective

L  =  L_sign  +  w1 · L_opp  +  w2 · Σ_k L_rt^(k)  +  w3 · L_coh
 
L_sign — next-sign prediction: representamen regression + oppositional-value match + InfoNCE contrast. Granularity.
L_opp — base layer is differential, not an unordered set. Relational structure at one scale.
L_rt — ‖V(Λr) − r‖ with a positive floor δ_min that penalizes the degenerate Λ = V = identity.
L_coh — opposition commutes with scaling: ‖v(Λr) − Λ_opp(v(r))‖². Relational structure across scales.
The worked example, in one number

The example predicts the sign after “the soup was”; the gold answer is hot. The prediction lands at cosine similarity 0.985 to hot and only 0.442 to large — clearly closest to the right answer. Yet the InfoNCE softmax still hands 33% of its probability mass to large:

hot (gold)   sim=+0.9854  p=0.5715
cold         sim=-0.7923  p=0.0966
large        sim=+0.4418  p=0.3319   <- wrong by value, near by coordinate
 
A pure-MSE target sees the small regression error (0.025) and calls the prediction solved. The contrastive term registers the residual competition from a neighbour that is wrong by value but near by coordinate — exactly the relational information a flat vocabulary discards. This is why L_sign includes a contrastive term rather than regression alone. Full derivation: docs/worked_example.md.

Repository layout

src/sign_prediction/   the layer (pure NumPy, manual gradients)
tests/                 gradient checks + worked-example regression tests
examples/              runnable walk-through
docs/                  architecture derivation, worked example, roadmap
 
Scope

This is a reference implementation, not a trained model or a benchmark result. It establishes that the design is coherent and trainable, with every number reproducible and every gradient verified. What it leaves open — bootstrapping the opposition supervision, giving each scale its own head, deriving the floor δ_min rather than tuning it, and porting to an autograd framework for training at scale — is listed with falsifiable predictions in docs/roadmap.md. Coded together with Claude.

License

MIT — see LICENSE.
