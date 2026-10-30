import json

import numpy as np
import pandas as pd
import pytest

from llm_reserve.baselines import (
    BAR, FIG_DIR, MAX_TOKENS, ORACLE, evaluate_methods, group_quantile, input_quintile_edges, make_methods,
    oracle_bounds, with_input_bucket,
)


def test_group_quantile_uses_train_groups_and_rounds_up():
    train = pd.DataFrame({"model": ["a"] * 4 + ["b"] * 2, "output_tokens": [10, 20, 30, 40, 100, 200]})
    rows = pd.DataFrame({"model": ["b", "a", "a"]})
    bounds = group_quantile(train, rows, ["model"])
    assert bounds(1.0).tolist() == [200, 40, 40]
    assert bounds(0.5).tolist() == [150, 25, 25]
    assert bounds(0.1).tolist() == [110, 13, 13]  # a: 10 + 0.3 × 10 = 13; b: 110


def test_group_quantile_global_when_no_keys():
    train = pd.DataFrame({"output_tokens": [1, 2, 3, 4, 5]})
    bounds = group_quantile(train, pd.DataFrame(index=range(3)), [])
    assert bounds(0.5).tolist() == [3, 3, 3]


def test_group_quantile_rejects_group_unseen_in_train():
    train = pd.DataFrame({"model": ["a"], "output_tokens": [10]})
    with pytest.raises(ValueError):
        group_quantile(train, pd.DataFrame({"model": ["z"]}), ["model"])


def test_input_buckets_are_quintiles_of_train():
    train = pd.DataFrame({"input_size": np.arange(1, 101)})
    edges = input_quintile_edges(train)
    assert np.allclose(edges, [20.8, 40.6, 60.4, 80.2])
    rows = pd.DataFrame({"input_size": [0, 20, 21, 50, 80, 81, 10_000]})
    # Values outside the train range land in the end buckets.
    assert with_input_bucket(rows, edges).input_bucket.tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_oracle_is_k_times_own_mean_plus_own_std():
    rows = pd.DataFrame({"sample_id": ["s", "s", "s", "t"], "output_tokens": [100, 200, 300, 50]})
    # s: mean 200, sample std 100. t: one completion, so std 0.
    assert oracle_bounds(rows)(1.0).tolist() == [300, 300, 300, 50]
    assert oracle_bounds(rows)(0.5).tolist() == [200, 200, 200, 25]


def synthetic(n_prompts: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two models, two datasets; length depends on the dataset and a per-prompt scale."""
    rng = np.random.default_rng(seed)
    prompts = pd.DataFrame({"prompt_id": [f"p{seed}_{i}" for i in range(n_prompts)],
                            "dataset": rng.choice(["chat", "code"], n_prompts)})
    rows = []
    for p in prompts.itertuples():
        scale = rng.lognormal(6 if p.dataset == "chat" else 4, 0.5)
        for model in ("m1", "m2"):
            for i in range(10):
                rows.append((f"{p.prompt_id}_{model}", p.prompt_id, model, int(rng.integers(10, 500)),
                             max(1, int(scale * rng.lognormal(0, 0.2)))))
    cols = ["sample_id", "prompt_id", "model", "input_size", "output_tokens"]
    return pd.DataFrame(rows, columns=cols), prompts


def test_evaluate_methods_orders_baselines_and_oracle():
    train, p_train = synthetic(300, seed=0)
    val, p_val = synthetic(200, seed=1)
    prompts = pd.concat([p_train, p_val])
    results = evaluate_methods(make_methods(train, val, prompts), val.output_tokens.to_numpy(), targets=(0.9, 0.95))

    for name, r in results.items():
        if name != "max_tokens":
            assert r["coverage"] == pytest.approx(0.95, abs=0.005)
            assert len(r["curve"]) == 2
    assert results["max_tokens"]["theta"] == 0.0 and results["max_tokens"]["curve"] == []
    assert results[BAR]["vs_bar"] == 0.0 and results[BAR]["gap_closed"] == 0.0
    assert results[ORACLE]["gap_closed"] == 1.0
    # Knowing the dataset helps here by construction, and the oracle beats everything.
    assert results["per_model_dataset"]["ratio"] < results[BAR]["ratio"]
    assert results[ORACLE]["ratio"] == min(r["ratio"] for r in results.values())


@pytest.mark.skipif(not (FIG_DIR / "baselines_validate.json").exists(), reason="run `make baselines`")
def test_locked_baselines_reproduce_the_data_check():
    """The Day-3 gate: the committed validation table matches the PRD data check (2.25 / 2.14 / 2.00 / 1.31)."""
    summary = json.loads((FIG_DIR / "baselines_validate.json").read_text())
    assert summary["split"] == "validate"
    m = summary["methods"]
    assert m[BAR]["ratio"] == pytest.approx(2.25, abs=0.01)
    assert m["per_model_input"]["ratio"] == pytest.approx(2.14, abs=0.01)
    assert m["per_model_dataset"]["ratio"] == pytest.approx(2.00, abs=0.01)
    assert m[ORACLE]["ratio"] == pytest.approx(1.31, abs=0.01)
    assert m["max_tokens"]["coverage"] < 1.0 and m["max_tokens"]["ratio"] > 9
    assert MAX_TOKENS == 4096
