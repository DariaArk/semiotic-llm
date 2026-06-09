# Roadmap

This implementation establishes that the design is *coherent and trainable*. It does not establish that it *helps*. This document lists what would be needed to find out, with falsifiable predictions that would confirm or kill each part.

## Falsifiable predictions

1. **Opposition structure improves contrast-sensitive tasks per parameter.** Mounting `OppositionHead` on a frozen encoder and training only the axes `A` should improve antonym detection, sentiment polarity, and negation handling more than adding an equal number of free parameters elsewhere. *Kill condition:* no improvement over a parameter-matched MLP head.

2. **Contrastive sign prediction reduces decode fragility.** A next-embedding model trained with the `L_sign` InfoNCE term should decode to fewer semantically-wrong-but-geometrically-close continuations than one trained on MSE alone, under matched compute. *Kill condition:* equal decode-error rates.

3. **The round-trip floor prevents representational collapse.** The floored round-trip loss should yield measurably more reduced coarse-scale codes (higher `d(r, V(Λr))`) while preserving downstream accuracy — genuine reduction rather than copying. *Kill condition:* the floor either collapses accuracy or makes no difference.

4. **Cross-scale coherence transfers.** An opposition axis learned at the base scale should remain meaningful at the coarser scale without retraining (probe above chance). *Kill condition:* base-scale axes are useless at the coarser scale.

## Near-term (weeks)

- **Port to PyTorch / JAX with autograd.** The NumPy version exists to prove the gradients by hand; a framework version is needed to train anything. The manual backward passes here are the ground-truth reference for testing the autograd version.
- **Mount the opposition head on a frozen pretrained encoder.** Cheapest experiment, highest information value (tests prediction 1). Freeze the encoder, learn only `A`, the synonym/antonym supervision, and the decorrelation term.
- **Replace the toy `InterpretantOperator` with attention.** The one-layer affine map is a placeholder; the surrounding loss machinery is unchanged when a transformer block is swapped in.

## Medium-term (months)

- **Bootstrap the opposition supervision.** The largest practical weakness: `L_opp` presupposes a partial inventory of opposed/synonym pairs (`O`, `N`), which is hand-curation that does not scale. A principled fix mines oppositions from co-occurrence statistics — pairs that are distributionally similar but occur in mutually-exclusive contexts (e.g. "hot"/"cold" share neighbours but rarely co-predicate). Building and validating that miner is a project in itself.
- **Give each scale its own opposition head.** `coherence_loss` currently assumes matching dimensions across scales and degrades gracefully when they differ (flagged in the code). A genuine `d → m` reduction needs a coarse-scale head so `L_coh` compares `v_base(r)` against `v_coarse(Λr)` through a learned alignment.
- **Derive the floor rather than tune it.** `δ_min` is a hyperparameter standing in for a real question: how much should the coarser scale reduce the base description? Deriving it from the information content of each scale, rather than sweeping, is the most interesting open problem.

## Long-term

- **Does sign-level prediction scale?** Whether oppositional and multi-scale structure compounds the known scaling advantages of concept-level prediction, or just adds overhead, is only answerable at scale.
- **Is the interpretant recurrence stable over long chains?** Iterating `r → r'` risks drift or collapse. A stopping condition — when generation commits to an output — is not implemented here and is necessary for generation rather than scoring.
- **What does cross-scale coherence buy that hierarchy-plus-contrast does not?** The clearest candidate for a genuinely distinctive contribution is `L_coh`: it imposes a relationship *between* scales that standard hierarchical models do not require. Demonstrating that this constraint improves results beyond what InfoNCE plus an ordinary hierarchy achieves would be the result worth chasing.
