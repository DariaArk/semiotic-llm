"""sign_prediction — predicting the next sign in a differentially-structured,
multi-scale embedding space.

The premise: next-token prediction over a flat vocabulary commits to the wrong
granularity (sub-meaning fragments) and the wrong relational structure (an
unordered set). A semiotic reading suggests predicting the next *sign* in a
differentially-structured, multi-scale embedding space instead — semiotics here
is a theory of *what the statistics should be computed over*, not a replacement
for statistical learning.

Each clause becomes a differentiable component, every gradient hand-derived and
finite-difference checked to ~1e-10:

  - OppositionHead       Saussurean differential value as an architectural
                         constraint: a sign's value is its position on learned
                         bipolar axes (a geometric relation), not an index.
  - InterpretantOperator Peirce's recursion as a next-embedding recurrence:
                         the interpretant is produced, not stored.
  - ScalingOperators     Lambda / V maps with a round-trip loss that makes
                         "the same skeleton at every scale" a checkable
                         equation between scales.
"""

from .layer import (
    OppositionHead,
    InterpretantOperator,
    ScalingOperators,
    SignPredictionLayer,
    sign_loss,
    opposition_loss,
    roundtrip_loss,
    coherence_loss,
    cosine,
    grad_check,
)

__version__ = "0.1.0"

__all__ = [
    "OppositionHead",
    "InterpretantOperator",
    "ScalingOperators",
    "SignPredictionLayer",
    "sign_loss",
    "opposition_loss",
    "roundtrip_loss",
    "coherence_loss",
    "cosine",
    "grad_check",
]
