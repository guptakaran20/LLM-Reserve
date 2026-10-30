"""Metrics and the matched-coverage protocol (PRD: Data and evaluation method).

y is the actual output length and the bound is what a method reserves for it:

    coverage          = share of rows with y <= bound
    reservation ratio = sum(bound) / sum(y)
    oracle gap closed = (ratio_baseline - ratio_model) / (ratio_baseline - ratio_oracle)

Every method is a family of bounds with one knob θ (a quantile or a scale factor), and its bounds
never shrink as θ grows. `tune` finds the smallest θ that reaches the target coverage on validation.
That θ is frozen, and test is scored once with it. Methods are compared by reservation ratio at
the same coverage: lower is better.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

TARGET = 0.95
# The coverage vs. reservation curve: targets 0.80, 0.81, ..., 0.99.
SWEEP = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))

BoundFn = Callable[[float], np.ndarray]


def coverage(y: np.ndarray, bound: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) <= np.asarray(bound)))


def reservation_ratio(y: np.ndarray, bound: np.ndarray) -> float:
    return float(np.sum(bound, dtype=float) / np.sum(y, dtype=float))


def oracle_gap_closed(ratio_baseline: float, ratio_model: float, ratio_oracle: float) -> float:
    """Share of the way from the baseline to the oracle: 0 is no better than the baseline, 1 is the oracle."""
    return (ratio_baseline - ratio_model) / (ratio_baseline - ratio_oracle)


@dataclass(frozen=True)
class Score:
    theta: float
    coverage: float
    ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


def score(bounds: BoundFn, y: np.ndarray, theta: float) -> Score:
    b = bounds(theta)
    return Score(theta, coverage(y, b), reservation_ratio(y, b))


def tune(bounds: BoundFn, y: np.ndarray, target: float = TARGET, lo: float = 0.0, hi: float = 1.0,
         tol: float = 1e-7) -> float:
    """Smallest θ in [lo, hi] (to within `tol`) whose bounds cover at least `target` of y.

    Bisection, so `bounds(θ)` must be nondecreasing in θ. The returned θ always reaches the target.
    """
    if coverage(y, bounds(hi)) < target:
        raise ValueError(f"coverage {target} is out of reach: θ = {hi} only covers {coverage(y, bounds(hi)):.4f}")
    if coverage(y, bounds(lo)) >= target:
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if coverage(y, bounds(mid)) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def sweep(bounds: BoundFn, y: np.ndarray, targets: Sequence[float] = SWEEP, lo: float = 0.0,
          hi: float = 1.0) -> list[Score]:
    """Tune to each target coverage in turn: the coverage vs. reservation curve."""
    return [score(bounds, y, tune(bounds, y, t, lo, hi)) for t in targets]
