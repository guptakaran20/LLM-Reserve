import json

import numpy as np
import pandas as pd
import pytest

from llm_reserve.baselines import BAR, MAX_TOKENS, ORACLE, with_input_bucket
from llm_reserve.evaluate import tune
from llm_reserve.holdout import (
    CAUSES, FIG_DIR, MLP, SERVING, WINNER, Frozen, breakdown, bucket_labels, cause, control_score, lodo_scores,
    prompt_bootstrap, sample_table, score_frozen, tail_accounting, worst_misses,
)
from llm_reserve.models import SHIFT, shifted_bounds


def test_score_frozen_keeps_validation_knobs_and_tunes_only_the_oracle():
    rng = np.random.default_rng(0)
    y = np.ceil(rng.lognormal(5, 0.8, size=4000))
    fitted = np.ceil(rng.lognormal(5, 0.8, size=4000))  # "train": the bar's percentiles come from here
    bar = lambda q: np.full(len(y), np.ceil(np.quantile(fitted, q)))  # noqa: E731
    model = lambda s: np.full(len(y), 0.8) * bar(0.95) * np.exp(s)  # noqa: E731
    oracle = lambda k: np.ceil(k * y)  # noqa: E731  (reads the lengths it bounds)
    methods = [
        Frozen("max_tokens", "max", True, None, (), None),
        Frozen(BAR, "bar", True, 0.9, (0.8, 0.9), (0.0, 1.0)),
        Frozen("m", "model", True, 0.25, (0.0, 0.25), SHIFT),
        Frozen(ORACLE, "oracle", False, None, (), (0.0, 10.0)),
    ]
    fns = {"max_tokens": lambda _: np.full(len(y), 4096.0), BAR: bar, "m": model, ORACLE: oracle}
    r = score_frozen(methods, fns, fns, y, y, targets=(0.8, 0.9))
    assert r[BAR]["theta"] == 0.9 and r["m"]["theta"] == 0.25  # frozen, never re-tuned on test
    assert [c["coverage"] for c in r[BAR]["curve"]] == pytest.approx([np.mean(y <= bar(t)) for t in (0.8, 0.9)])
    assert r[ORACLE]["theta"] == pytest.approx(tune(oracle, y, 0.95, 0, 10)) and r[ORACLE]["coverage"] >= 0.95
    assert r[ORACLE]["curve"][0]["coverage"] >= 0.8  # the ceiling is swept on test itself
    assert r[BAR]["vs_bar"] == 0 and r[BAR]["gap_closed"] == 0 and r[ORACLE]["gap_closed"] == pytest.approx(1)
    assert "matched" not in r["max_tokens"] and r["max_tokens"]["curve"] == []
    for name in (BAR, "m"):  # the diagnostic re-tunes to 95% on test
        assert 0.95 <= r[name]["matched"]["coverage"] < 0.96
    assert r["m"]["matched"]["vs_bar"] == pytest.approx(r["m"]["matched"]["ratio"] / r[BAR]["matched"]["ratio"] - 1)


def test_breakdown_by_hand():
    y, b = np.array([10, 20, 30, 40]), np.array([10, 10, 40, 40])
    out = breakdown(y, b, np.array(["a", "a", "b", "b"]), order=["b", "a", "z"])
    assert list(out) == ["b", "a"]  # the given order; groups that do not occur are skipped
    assert out["a"] == {"rows": 2, "coverage": 0.5, "ratio": pytest.approx(20 / 30)}
    assert out["b"] == {"rows": 2, "coverage": 1.0, "ratio": pytest.approx(80 / 70)}


def test_bucket_labels_follow_the_quintile_assignment():
    edges = np.array([38.0, 68.4, 157.0, 336.0])
    labels = bucket_labels(edges)
    assert labels == ["< 38", "38–68", "68–157", "157–336", "≥ 336"]
    buckets = with_input_bucket(pd.DataFrame({"input_size": [37, 38, 157, 1000]}), edges).input_bucket
    assert [labels[i] for i in buckets] == ["< 38", "38–68", "157–336", "≥ 336"]


def test_prompt_bootstrap_resamples_whole_prompts():
    rng = np.random.default_rng(1)
    prompts = np.repeat(np.arange(200), 12).astype(str)
    y = rng.lognormal(5, 1, size=len(prompts))
    bar = y * rng.uniform(0.5, 3, size=len(y))
    out = prompt_bootstrap(prompts, y, {"bar": bar, "m": 0.7 * bar}, "bar", n=300, goal=-0.2)
    assert out["prompts"] == 200 and out["resamples"] == 300
    assert out["methods"]["m"]["vs_reference"] == pytest.approx([-0.3, -0.3])  # a constant factor is certain
    assert out["methods"]["m"]["share_reaching_goal"] == 1.0
    assert out["methods"]["bar"]["vs_reference"] == [0.0, 0.0]
    lo, hi = out["methods"]["bar"]["coverage"]
    assert lo <= np.mean(y <= bar) <= hi and lo < hi


def rows_for(samples: dict[str, tuple[str, str, float, list[int]]]) -> pd.DataFrame:
    """sample_id -> (prompt_id, model, bound, completions), one row per completion."""
    out = [{"sample_id": s, "prompt_id": p, "model": m, "dataset": "D", "has_hint": False, "bound": b,
            "output_tokens": y} for s, (p, m, b, ys) in samples.items() for y in ys]
    return pd.DataFrame(out)


def test_causes_follow_the_rule_order():
    rows = rows_for({
        "runaway": ("p1", "m1", 100, [50, 60, 5000]),  # past max_tokens, though its median is fine
        "lone": ("p1", "m2", 100, [50, 60, 300]),  # median within the bound
        "llm": ("p1", "m3", 100, [150, 160, 170]),  # median over; neither other LLM on p1 is
        "misread_a": ("p2", "m1", 100, [200, 210, 220]),  # median over, and so is the other LLM's
        "misread_b": ("p2", "m2", 100, [300, 310, 320]),
        "alone": ("p3", "m1", 100, [150, 150]),  # no other LLM on its prompt
        "fine": ("p3b", "m1", 100, [10, 20]),
    })
    s = sample_table(rows)
    assert s.loc["llm", "others_over"] == 0 and s.loc["misread_a", "others_over"] == 1
    assert np.isnan(s.loc["alone", "others_over"])
    assert s.loc["lone", "n_over"] == 1 and s.loc["runaway", "miss"] == pytest.approx(49.0)
    assert s.loc["runaway", "worst"] > MAX_TOKENS
    causes = cause(s).to_dict()
    assert causes == {"runaway": "runaway", "lone": "lone_long_run", "llm": "llm_specific",
                      "misread_a": "prompt_misread", "misread_b": "prompt_misread", "alone": "llm_specific",
                      "fine": "lone_long_run"}
    assert set(causes.values()) <= set(CAUSES)
    top = worst_misses(s, n=3)
    assert list(top.index) == ["runaway", "misread_b", "lone"]  # largest (y - bound) / bound first
    assert "fine" not in worst_misses(s, n=10).index  # never under-predicted


def test_sample_table_rejects_a_bound_that_varies_in_a_sample():
    rows = rows_for({"s": ("p", "m", 100, [1, 2])})
    rows.loc[1, "bound"] = 101
    with pytest.raises(ValueError):
        sample_table(rows)


def test_tail_accounting_by_hand():
    y, b = np.array([100.0, 5000.0, 50.0]), np.array([80.0, 1000.0, 60.0])
    t = tail_accounting(y, b, np.array([False, True, False]), max_tokens=4096)
    assert (t["missed"], t["missed_degenerate"]) == (2, 1)
    assert t["over_tokens"] == 4020 and t["over_share_degenerate"] == pytest.approx(4000 / 4020)
    # A gateway reserves min(bound, 4096) and is billed min(y, 4096): 5000 overshoots by 3096, not 4000.
    assert t["over_tokens_clipped"] == 3116 and t["clipped_share_degenerate"] == pytest.approx(3096 / 3116)
    assert t["clipped_over_per_used"] == pytest.approx(3116 / (100 + 4096 + 50))


def synthetic_split(rng, prompts: int, offset: int, datasets=("A", "B"), models=("m1", "m2")) -> pd.DataFrame:
    """Completions with a per-dataset length scale; the dataset is a column, as in holdout's rows."""
    rows = []
    for p in range(prompts):
        d = datasets[p % len(datasets)]
        for m in models:
            for y in np.ceil(rng.lognormal(5 + 0.3 * (d == "B"), 0.8, size=5)):
                rows.append({"prompt_id": f"p{offset + p}", "sample_id": f"p{offset + p}|{m}", "model": m,
                             "dataset": d, "output_tokens": y})
    return pd.DataFrame(rows)


def test_lodo_tunes_without_the_held_out_dataset_and_reads_its_test_rows():
    rng = np.random.default_rng(2)
    train, va, te = (synthetic_split(rng, n, o) for n, o in ((40, 0), (20, 100), (20, 200)))
    log = lambda df: np.log1p(np.where(df.dataset == "B", 400.0, 150.0)) + rng.normal(0, 0.05, len(df))  # noqa: E731
    held = {d: {"winner": {"validate": log(va), "test": log(te)}} for d in ("A", "B")}
    full = {"winner": (log(te), 0.5)}
    out = lodo_scores(["A", "B"], train, va, va.output_tokens.to_numpy(), te, te.output_tokens.to_numpy(), held, full,
                      0.95)
    r = out["datasets"]["A"]
    outside = va.dataset.to_numpy() != "A"
    shift = tune(shifted_bounds(held["A"]["winner"]["validate"][outside]), va.output_tokens.to_numpy()[outside],
                 0.95, *SHIFT)
    assert r["lodo_winner"]["shift"] == pytest.approx(shift)
    assert r["test_rows"] == int((te.dataset == "A").sum()) and r["test_prompts"] == 10
    assert r["gap_closed_full"] == pytest.approx(
        (r["bar_full"]["matched_ratio"] - r["full_winner"]["matched_ratio"])
        / (r["bar_full"]["matched_ratio"] - r["oracle_ratio"]))
    assert r["full_winner"]["shift"] == 0.5 and r["bar_full"]["theta"] == 0.95
    pooled = out["pooled"]
    assert 0 < pooled["coverage"] <= 1 and pooled["matched_vs_bar"] == pytest.approx(
        pooled["matched_ratio"] / pooled["bar_matched_ratio"] - 1)


def test_control_score_tunes_on_validation_and_reports_matched():
    rng = np.random.default_rng(3)
    yv, yt = np.ceil(rng.lognormal(5, 0.6, 3000)), np.ceil(rng.lognormal(5, 0.6, 3000))
    pv, pt = np.full(3000, np.log1p(150.0)), np.full(3000, np.log1p(150.0))
    r = control_score(pv, yv, pt, yt, bar_matched=3.0, oracle=1.2)
    assert r["shift"] == pytest.approx(tune(shifted_bounds(pv), yv, 0.95, *SHIFT))
    assert r["matched_vs_bar"] == pytest.approx(r["matched_ratio"] / 3.0 - 1)


@pytest.mark.skipif(not (FIG_DIR / "test_results.json").exists(), reason="run `make eval`")
def test_day7_gate_test_scored_once_with_frozen_knobs():
    """The Day-7 gate: test scored with the validation knobs, results, errors and LODO all recorded."""
    r = json.loads((FIG_DIR / "test_results.json").read_text())
    base = json.loads((FIG_DIR / "baselines_validate.json").read_text())["methods"]
    day5 = json.loads((FIG_DIR / "models_validate.json").read_text())["methods"]
    day6 = json.loads((FIG_DIR / "mlp_validate.json").read_text())
    assert r["split"] == "test" and r["frozen"]["winner"] == WINNER == day6["winner"]
    m = r["methods"]
    for name in ("global", BAR, "per_model_input", "per_model_dataset"):
        assert m[name]["theta"] == base[name]["theta"]
    assert m[SERVING]["theta"] == day5[SERVING]["theta"]
    assert m[MLP]["theta"] == day6["methods"][day6["chosen"]]["theta"]
    assert m[WINNER]["theta"] == day6["methods"][WINNER]["theta"] == r["frozen"]["shifts"][WINNER]
    assert r["frozen"]["winner_validate_ratio"] == pytest.approx(day6["methods"][WINNER]["ratio"], abs=1e-6)
    assert 0.95 <= m[ORACLE]["coverage"] < 0.951
    for name, row in m.items():
        assert len(row["curve"]) == (0 if name == "max_tokens" else 20)
        if "matched" in row:
            assert 0.95 <= row["matched"]["coverage"] < 0.951
    assert r["goals"]["vs_bar"] == pytest.approx(m[WINNER]["ratio"] / m[BAR]["ratio"] - 1)
    errors = r["errors"]
    assert errors["n"] == len(errors["worst"]) == 100
    assert sum(c["count"] for c in errors["causes"].values()) == 100
    assert len(r["lodo"]["datasets"]) == 7 and "control" in r["lodo"]
    assert set(r["breakdown"]["by_model"][WINNER]) == set(r["breakdown"]["by_model"][BAR])
