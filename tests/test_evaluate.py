import numpy as np
import pytest

from llm_reserve.evaluate import SWEEP, coverage, oracle_gap_closed, reservation_ratio, score, sweep, tune


def test_coverage_counts_ties_as_covered():
    y = np.array([10, 20, 30, 40])
    assert coverage(y, np.array([10, 19, 30, 50])) == 0.75


def test_reservation_ratio_hand_computed():
    # Reserved 20 + 20 + 60 = 100 for 10 + 20 + 20 = 50 used.
    assert reservation_ratio(np.array([10, 20, 20]), np.array([20, 20, 60])) == 2.0


def test_oracle_gap_closed():
    # Baseline 2.25, oracle 1.31: halfway is 1.78.
    assert oracle_gap_closed(2.25, 1.78, 1.31) == pytest.approx(0.5)
    assert oracle_gap_closed(2.25, 2.25, 1.31) == 0.0
    assert oracle_gap_closed(2.25, 1.31, 1.31) == 1.0
    assert oracle_gap_closed(2.25, 2.72, 1.31) == pytest.approx(-0.5)  # worse than the baseline


def test_tune_finds_smallest_scale_reaching_target():
    y = np.arange(1, 101)  # 1..100
    base = np.full(100, 10.0)
    k = tune(lambda k: k * base, y, target=0.95, hi=20, tol=1e-9)
    # 95 of the 100 lengths are <= 95, so the bound must be 95: k = 9.5.
    assert k == pytest.approx(9.5)
    assert coverage(y, k * base) >= 0.95
    assert coverage(y, (k - 1e-6) * base) < 0.95


def test_tune_returns_lo_when_already_covered():
    y = np.array([1, 2, 3])
    assert tune(lambda q: np.full(3, 5.0), y, target=0.95) == 0.0


def test_tune_rejects_unreachable_target():
    with pytest.raises(ValueError):
        tune(lambda k: k * np.ones(3), np.array([1, 2, 30]), target=0.95, hi=10)


def test_score_and_sweep_are_monotone():
    rng = np.random.default_rng(0)
    y = rng.lognormal(6, 1, size=5000)
    bounds = lambda q: np.full(len(y), np.quantile(y, q))  # noqa: E731
    s = score(bounds, y, 0.5)
    assert s.coverage == pytest.approx(0.5, abs=1e-3)
    assert s.ratio == pytest.approx(np.median(y) / y.mean())

    curve = sweep(bounds, y)
    assert len(curve) == len(SWEEP) == 20
    assert SWEEP[0] == 0.80 and SWEEP[-1] == 0.99
    assert all(c.coverage >= t for c, t in zip(curve, SWEEP))
    assert all(a.ratio <= b.ratio for a, b in zip(curve, curve[1:]))
