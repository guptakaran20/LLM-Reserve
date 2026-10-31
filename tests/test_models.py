import json

import numpy as np
import pandas as pd
import pytest

from llm_reserve.evaluate import coverage
from llm_reserve.models import (
    FIG_DIR, MODELS_DIR, TRAIN_QUANTILE, design, evaluate_predictions, feature_group, fit_linear, fit_xgb,
    grouped_importance, pinball, predict_xgb, prompt_block, shifted_bounds,
)


def test_shifted_bounds_round_up_and_never_shrink():
    bounds = shifted_bounds(np.log1p([9.5, 98.5, 0.0]))
    assert bounds(0.0).tolist() == [10, 99, 1]  # whole tokens, at least 1
    assert bounds(np.log(3)).tolist()[:2] == [31, 298]  # a shift of log 3 triples 1 + y: 3 × 10.5 - 1 = 30.5
    assert bounds(-4.0).tolist() == [1, 1, 1]  # 99.5 / e⁴ - 1 = 0.82: never below 1 token
    grid = [bounds(s) for s in np.linspace(-4, 4, 41)]
    assert all((a <= b).all() for a, b in zip(grid, grid[1:]))


def test_pinball_loss_by_hand():
    # Under-prediction by 10 costs q × 10, over-prediction by 10 costs (1 - q) × 10.
    assert pinball(np.array([10.0, 0.0]), np.array([0.0, 10.0]), 0.9) == pytest.approx((9 + 1) / 2)


def heteroscedastic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Log length grows with x0, and so does its spread: the 95% quantile is not a constant offset."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)
    y_log = 3 + 2 * X[:, 0] + (0.2 + 0.6 * X[:, 0]) * rng.normal(size=n)
    return X, y_log


def test_linear_quantile_regression_covers_its_quantile():
    X, y_log = heteroscedastic(4000, seed=0)
    lin = fit_linear(X, y_log, sample=2000)
    assert coverage(y_log, lin.predict(X)) == pytest.approx(TRAIN_QUANTILE, abs=0.02)
    assert lin.coef[0] > 0 and abs(lin.coef[1]) < 0.1  # the bound rises with x0 and ignores noise features


def test_xgboost_early_stops_and_tunes_to_matched_coverage():
    X, y_log = heteroscedastic(6000, seed=1)
    Xv, yv_log = heteroscedastic(3000, seed=2)
    cols = ["x0", "x1", "x2"]
    booster = fit_xgb(X, y_log, Xv, yv_log, cols, params={"eta": 0.3, "nthread": 1}, rounds=300, early_stop=20)
    assert booster.best_iteration < 299
    pred = predict_xgb(booster, Xv, cols)
    assert np.corrcoef(Xv[:, 0], pred)[0, 1] > 0.9
    y = np.round(np.expm1(yv_log))
    r = evaluate_predictions(pred, y, np.where(Xv[:, 1] > 0.5, "a", "b"), targets=(0.9, 0.95))
    assert r["coverage"] == pytest.approx(0.95, abs=0.002) and r["coverage"] >= 0.95
    assert [c["coverage"] for c in r["curve"]] == pytest.approx([0.9, 0.95], abs=0.002)
    assert set(r["coverage_by_model"]) == {"a", "b"}
    assert grouped_importance(booster)["x0"] > 0.5


def test_importance_groups_the_one_hot_and_the_embedding():
    assert feature_group("model=gemma") == "target model (one-hot)"
    assert feature_group("minilm128_17") == "MiniLM-128 embedding"
    assert feature_group("log_chars") == "log_chars"

    class Fake:
        def get_score(self, importance_type):
            assert importance_type == "total_gain"
            return {"model=a": 1.0, "model=b": 2.0, "minilm128_0": 3.0, "minilm128_5": 3.0, "log_chars": 1.0}

    imp = grouped_importance(Fake())
    assert list(imp) == ["MiniLM-128 embedding", "target model (one-hot)", "log_chars"]
    assert sum(imp.values()) == pytest.approx(1.0) and imp["MiniLM-128 embedding"] == pytest.approx(0.6)


def test_design_appends_the_prompt_block_and_rejects_leaks():
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    block = prompt_block(emb, 128, np.array([0, 1]), None)
    assert block.columns == ["minilm128_0", "minilm128_1", "minilm128_2", "minilm128_3"]
    X = pd.DataFrame({"log_chars": [1.0, 2.0], "model=m1": [1.0, 0.0]})
    meta = pd.DataFrame({"prompt_row": [2, 0]})
    A, cols = design(X, meta, block)
    assert cols == ["log_chars", "model=m1", *block.columns]
    assert A.dtype == np.float32 and A[:, 2:].tolist() == [emb[2].tolist(), emb[0].tolist()]
    assert design(X, meta, None)[1] == ["log_chars", "model=m1"]
    with pytest.raises(ValueError):
        design(X.assign(output_mean=0.0), meta, None)


def test_pca_block_is_fitted_on_train_prompts_only():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 8)).astype(np.float32)
    emb[40:] += 100  # far-away non-train prompts must not move the projection
    train_rows = np.repeat(np.arange(40), 3)  # prompts repeat across rows, as in the real matrix
    block = prompt_block(emb, 128, train_rows, 2)
    assert block.columns == ["minilm128_pc0", "minilm128_pc1"] and block.matrix.shape == (50, 2)
    assert block.pca.mean_ == pytest.approx(emb[:40].mean(0), abs=1e-5)
    assert feature_group(block.columns[0]) == "MiniLM-128 embedding"


@pytest.mark.skipif(not (FIG_DIR / "models_validate.json").exists(), reason="run `make models`")
def test_model_rows_are_validation_only_and_at_matched_coverage():
    """The Day-5 gate: every model has a validation row at 95% coverage, and its frozen shift is saved."""
    summary = json.loads((FIG_DIR / "models_validate.json").read_text())
    assert summary["split"] == "validate"
    assert {"linear", "xgb_hand", "xgb_minilm128_raw", "xgb_minilm128", "xgb_minilm256"} <= set(summary["methods"])
    for r in summary["methods"].values():
        assert 0.95 <= r["coverage"] < 0.951
        assert len(r["curve"]) == 20
    if (MODELS_DIR / "manifest.json").exists():
        manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
        for name, m in manifest["models"].items():
            assert m["shift"] == summary["methods"][name]["theta"]
            assert (MODELS_DIR / m["file"]).exists()
