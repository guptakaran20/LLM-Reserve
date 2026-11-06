import json

import numpy as np
import pandas as pd
import pytest

from llm_reserve import mispredict as mp
from llm_reserve import sweep as sw
from llm_reserve.baselines import BAR
from llm_reserve.data import OUT_DIR
from llm_reserve.simulate import Config, bounds_policy

FIG_DIR = mp.FIG_DIR


def test_a_cap_rounds_up_and_never_exceeds_max_tokens():
    assert mp.capped(np.array([10.0, 10.2, 3000.0]), 1.5).tolist() == [15, 16, 4096]


def test_truncation_rates_separate_real_answers_from_degenerate_ones():
    y = np.array([10, 20, 30, 5_000])
    degenerate = np.array([False, False, False, True])
    rates = mp.truncation_rates(y, degenerate, np.full(4, 15.0), margins=(1.0, 2.0))
    assert rates["1"] == {"clean": 0.5, "all": 0.75}  # 20 and 30 cut, and the runaway
    assert rates["2"] == {"clean": 0.0, "all": 0.25}  # the cap is 30; the runaway is cut at 4,096 anyway


def test_the_margin_is_the_smallest_that_meets_the_truncation_goal():
    rates = {"1": {"clean": 0.05}, "1.5": {"clean": 0.009}, "2": {"clean": 0.003}}
    assert mp.choose_margin(rates, goal=0.01) == 1.5
    with pytest.raises(RuntimeError):
        mp.choose_margin(rates, goal=0.001)


def rows(n_samples: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for s in range(n_samples):
        for c in range(5):
            long = c == 4 and s % 3 == 0  # some completions run far past the bound
            out.append({"sample_id": f"s{s}", "input_size": 20, "output_tokens": int(rng.integers(50, 150) * (2 if long else 1)),
                        "model": "m", "dataset": "d", "is_degenerate": long})
    return pd.DataFrame(out)


def test_the_holdback_is_the_smallest_share_that_keeps_overshoot_under_the_goal():
    r = rows()
    bounds = np.full(len(r), 110.0)
    configs = [Config(budget=20_000, rate=r_, duration=300.0, tokens_per_second=50.0) for r_ in (8.0, 12.0)]
    sizing = mp.size_holdback(r, bounds, configs, seeds=(0, 1), shares=tuple(i / 100 for i in range(31)), goal=0.01)
    assert sizing["unprotected_worst"] > 0.01  # the budget binds and misses overshoot it
    tried = sizing["tried"]
    assert tried[-1]["passed"] and tried[-1]["share"] == sizing["share"] > 0
    assert not any(t["passed"] for t in tried[:-1])
    worst = max(mp.worst_overshoot(r, mp.with_strategy(bounds_policy(bounds), holdback=sizing["share"]), c, (0, 1))
                for c in configs)
    assert worst <= 0.01


def test_a_capped_policy_that_overshoots_fails_the_check():
    good = [{"setting": {}, "policies": {"x+cap1": {"overshoot_max_share_worst_seed": 0.0, "under_reserved_tokens": 0}}}]
    mp.check_cap_never_overshoots(good)
    bad = [{"setting": {}, "policies": {"x+cap1": {"overshoot_max_share_worst_seed": 0.01, "under_reserved_tokens": 3}}}]
    with pytest.raises(AssertionError):
        mp.check_cap_never_overshoots(bad)


@pytest.mark.skipif(not (OUT_DIR / "completions_validate.parquet").exists()
                    or not (mp.holdout.predict.PRED_DIR / "mlp_validate.npz").exists(), reason="run `make eval`")
def test_validation_bounds_reproduce_the_frozen_validation_numbers():
    bounds, y, degenerate = mp.validation_bounds()  # raises if coverage or ratio drift
    assert set(bounds) == {BAR, sw.WINNER} and len(y) == len(degenerate) == len(bounds[BAR])
    assert bounds[sw.WINNER].mean() < bounds[BAR].mean()


@pytest.mark.skipif(not (FIG_DIR / "mispredict.json").exists(), reason="run `make mispredict`")
def test_day10_gate_both_strategies_in_one_table_with_choices_made_on_validation():
    """The Day-10 gate: A's truncation and B's overshoot side by side, A never overshooting, knobs from validation."""
    r = json.loads((FIG_DIR / "mispredict.json").read_text())
    assert r["split"] == "test" and r["seeds"] == [0, 1, 2] and set(r["validation_seeds"]).isdisjoint(r["seeds"])
    assert len(r["sweep"]) == len(sw.grid()) and (FIG_DIR / "mispredict.png").exists()
    for base in mp.BASES:
        m, share = r["chosen_margin"][base], r["chosen_holdback"][base]
        assert m in mp.MARGINS and r["margin_choice"][base]["validation_truncation"][f"{m:g}"]["clean"] <= mp.GOAL_TRUNCATION
        last = r["holdback_sizing"][base]["tried"][-1]
        assert last["share"] == share and last["passed"] and last["worst_overshoot"] <= mp.GOAL_OVERSHOOT
        for s in r["sweep"] + [r["default_result"], r["stress_result"]]:
            for margin in mp.MARGINS:
                p = s["policies"][mp.cap_name(base, margin)]
                assert p["overshoot_max_share_worst_seed"] == 0 and p["under_reserved_tokens"] == 0
            assert {"truncated_clean_rate", "overshoot_max_share"} <= set(s["policies"][mp.hold_name(base, share)])
    stress = r["stress_result"]["policies"]
    # A cuts more answers the tighter the margin; B never truncates beyond what max_tokens does.
    for base in mp.BASES:
        cut = [stress[mp.cap_name(base, m)]["truncated_clean_rate"] for m in mp.MARGINS]
        assert cut == sorted(cut, reverse=True)
        held = stress[mp.hold_name(base, r["chosen_holdback"][base])]
        assert held["truncated_clean_rate"] == pytest.approx(stress[base]["truncated_clean_rate"], abs=1e-3)
