import json

import numpy as np
import pandas as pd
import pytest

from llm_reserve import sweep as sw
from llm_reserve.baselines import BAR
from llm_reserve.data import OUT_DIR
from llm_reserve.simulate import Config, bounds_policy

FIG_DIR = sw.FIG_DIR


def test_the_grid_crosses_rate_with_budget_and_with_speed_without_repeating_the_default():
    g = sw.grid(rates=(1.0, 2.0), budgets=(100, 200), speeds=(25.0, 50.0), base=Config())
    keys = [(c.rate, c.budget, c.tokens_per_second) for c in g]
    assert len(keys) == len(set(keys)) == 2 * 2 + 2 * 1
    assert {k for k in keys if k[2] == 25.0} == {(1.0, 200_000, 25.0), (2.0, 200_000, 25.0)}
    assert all(c.duration == Config().duration and c.window == Config().window for c in g)


def test_paired_multiples_compare_seeds_on_the_same_trace():
    seeds = {"a": [{"admitted": 30}, {"admitted": 20}], "b": [{"admitted": 20}, {"admitted": 20}]}
    assert sw.paired(seeds, "a", "b") == {"mean": 1.25, "min": 1.0, "max": 1.5}


def test_group_admit_rates_pool_the_seeds_counts():
    tables = [{"x": {"requests": 10, "admitted": 10, "admit_rate": 1.0}},
              {"x": {"requests": 30, "admitted": 0, "admit_rate": 0.0}}]
    m = sw.mean_groups(tables)["x"]
    assert m["admit_rate"] == 0.25 and m["requests"] == 20


def rows(n_samples: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for s in range(n_samples):
        for c in range(5):
            out.append({"sample_id": f"s{s}", "input_size": 50, "output_tokens": int(rng.integers(50, 400) * (1 + s % 2)),
                        "model": f"m{s % 2}", "dataset": f"d{s % 3}"})
    return pd.DataFrame(out)


def test_a_setting_runs_every_policy_on_the_same_traces_and_compares_them():
    r = rows()
    worst = r.groupby("sample_id").output_tokens.transform("max").to_numpy().astype(float)
    policies = {"max_tokens": bounds_policy(np.full(len(r), 4096.0)), BAR: bounds_policy(worst * 1.5),
                "tight": bounds_policy(worst)}  # no policy reserves less than a request uses
    cfg = Config(budget=60_000, rate=4.0, duration=600.0, tokens_per_second=50.0)
    res = sw.run_setting(r, policies, cfg, seeds=(0, 1), groups=True)
    p = res["policies"]
    assert res["setting"] == {"rate": 4.0, "budget": 60_000, "tokens_per_second": 50.0}
    assert len({q["requests"] for q in p.values()}) == 1  # same traces
    assert p["max_tokens"]["admitted"] < min(p[BAR]["admitted"], p["tight"]["admitted"])
    assert p["max_tokens"]["vs_max_tokens"] == {"mean": 1.0, "min": 1.0, "max": 1.0}
    seeds = {n: q["seeds"] for n, q in p.items()}
    assert p["tight"]["vs_bar"] == sw.paired(seeds, "tight", BAR) and p["tight"]["vs_max_tokens"]["min"] > 1
    for q in p.values():
        assert q["overshoot_windows"] == 0 and q["under_reserved_tokens"] == 0
    assert set(p["tight"]["by_dataset"]) == {"d0", "d1", "d2"} and set(p["tight"]["by_model"]) == {"m0", "m1"}


@pytest.mark.skipif(not (FIG_DIR / "test_results.json").exists() or not (OUT_DIR / "completions_test.parquet").exists(),
                    reason="run `make eval`")
def test_frozen_bounds_reproduce_day7_on_the_original_config():
    bounds, check = sw.frozen_bounds()
    day7 = json.loads((FIG_DIR / "test_results.json").read_text())["methods"]
    assert set(bounds) == set(sw.POLICIES)
    for name, b in bounds.items():
        assert len(b) == 140_480 and (b > 0).all()
        assert check[name]["theta"] == day7[name]["theta"]
        assert check[name]["coverage_original"] == pytest.approx(day7[name]["coverage_original"], abs=1e-12)
    assert (bounds["max_tokens"] == 4096).all()
    assert bounds[sw.WINNER].mean() < bounds[BAR].mean() < bounds["max_tokens"].mean()


@pytest.mark.skipif(not (FIG_DIR / "sim_policies.json").exists(), reason="run `make sweep`")
def test_day9_gate_a_comparison_table_over_rates_budgets_and_speeds():
    """The Day-9 gate: every policy at every setting, 3 seeds, compared with max_tokens and with the bar."""
    r = json.loads((FIG_DIR / "sim_policies.json").read_text())
    assert r["split"] == "test" and r["config"] == "original" and r["seeds"] == [0, 1, 2]
    assert len(r["sweep"]) == len(sw.grid())
    assert (FIG_DIR / "sim_policies.png").exists()
    for s in r["sweep"] + [r["default_result"], r["stress_result"]]:
        assert set(s["policies"]) == set(sw.POLICIES)
        for p in s["policies"].values():
            assert {"vs_max_tokens", "vs_bar", "wrongful_rejections", "overshoot_windows"} <= set(p)
        assert s["policies"]["max_tokens"]["overshoot_windows"] == 0
    default = r["default_result"]["policies"]
    assert default[sw.WINNER]["admitted"] >= default[BAR]["admitted"] >= default["max_tokens"]["admitted"]
