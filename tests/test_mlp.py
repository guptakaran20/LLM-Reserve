import json

import numpy as np
import pandas as pd
import pytest
import torch

from llm_reserve.evaluate import coverage
from llm_reserve.mlp import (
    CHOSEN, CONFIGS, ENSEMBLE, MLPConfig, QuantileMLP, Samples, collapse, config, fit_mlp, pinball_loss,
    predict_mlp,
)
from llm_reserve.models import FIG_DIR, MODELS_DIR, TRAIN_QUANTILE, pinball


def test_collapse_keeps_one_row_per_sample_and_maps_back():
    X = pd.DataFrame({"log_chars": [1.0, 1.0, 2.0, 1.0], "model=a": [1.0, 1.0, 0.0, 1.0]})
    meta = pd.DataFrame({"sample_id": ["s1", "s1", "s2", "s1"], "prompt_row": [0, 0, 1, 0]})
    s = collapse(X, np.array([9.0, 99.0, 0.0, 999.0]), meta, None)
    assert s.columns == ["log_chars", "model=a"]
    assert s.X.tolist() == [[1.0, 1.0], [2.0, 0.0]]
    assert s.codes.tolist() == [0, 0, 1, 0]
    assert s.Y.shape == (2, 3)
    np.testing.assert_allclose(s.Y[0], np.log1p([9, 99, 999]), rtol=1e-6)
    assert s.Y[1, 0] == 0.0 and np.isnan(s.Y[1, 1:]).all()  # padding, not zeros


def test_sample_pinball_equals_the_row_level_loss():
    """Collapsing to samples must not change the objective: each completion counts once."""
    Y = torch.tensor([[1.0, 2.0, 3.0], [5.0, np.nan, np.nan]])
    pred = torch.tensor([2.0, 4.0])
    rows_y, rows_pred = np.array([1.0, 2.0, 3.0, 5.0]), np.array([2.0, 2.0, 2.0, 4.0])
    assert float(pinball_loss(pred, Y, 0.9)) == pytest.approx(pinball(rows_y, rows_pred, 0.9))
    # By hand: over-predictions cost 0.1 × 1, under-predictions 0.9 × 1, exact predictions 0.
    assert float(pinball_loss(pred, Y, 0.9)) == pytest.approx((0.1 + 0 + 0.9 + 0.9) / 4)


def test_standardization_is_part_of_the_module():
    mean, scale = np.array([10.0, -1.0], dtype=np.float32), np.array([2.0, 0.5], dtype=np.float32)
    model = QuantileMLP(mean, scale, hidden=(4,), dropout=0.5, input_dropout=0.5).eval()
    x = torch.tensor([[12.0, 0.0], [10.0, -1.0]])
    assert torch.allclose(model(x), model.net((x - torch.from_numpy(mean)) / torch.from_numpy(scale)).squeeze(-1))
    assert torch.equal(model(x), model(x))  # eval mode: no dropout, deterministic
    assert {"mean", "scale"} <= set(model.state_dict())  # saved and exported with the weights


def heteroscedastic_samples(n: int, k: int, seed: int) -> Samples:
    """Log length grows with x0 and so does its spread; k completions per sample, like CASTILLO's 10."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)
    Y = (3 + 2 * X[:, :1] + (0.2 + 0.6 * X[:, :1]) * rng.normal(size=(n, k))).astype(np.float32)
    return Samples(X, Y, np.repeat(np.arange(n), k), ["x0", "x1", "x2"])


def test_mlp_learns_the_quantile_and_early_stops():
    train, val = heteroscedastic_samples(3000, 5, seed=0), heteroscedastic_samples(1000, 5, seed=1)
    cfg = MLPConfig("t", "t", hidden=(32, 16), dropout=0.0, max_epochs=300, patience=5, lr=3e-3, batch_size=256)
    model, history = fit_mlp(train, val, cfg, seed=0, verbose=False)
    assert len(history) < cfg.max_epochs
    best = min(h["val_loss"] for h in history)
    assert best == pytest.approx(history[int(np.argmin([h["val_loss"] for h in history]))]["val_loss"])
    pred = predict_mlp(model, val.X)
    assert coverage(val.Y.ravel(), np.repeat(pred, 5)) == pytest.approx(TRAIN_QUANTILE, abs=0.02)
    assert np.corrcoef(val.X[:, 0], pred)[0, 1] > 0.95  # the bound rises with x0
    # The returned model is the best epoch's, and training is reproducible from the seed.
    again, _ = fit_mlp(train, val, cfg, seed=0, verbose=False)
    np.testing.assert_array_equal(predict_mlp(again, val.X), pred)


def test_config_labels_name_every_knob_off_its_default():
    assert config("a", embedding=None).label == "MLP (hand-built)"
    assert config("b", embedding="pca", input_dropout=0.2).label == "MLP (hand-built + MiniLM-128 PCA-32; input dropout 0.2)"
    assert len({c.name for c in CONFIGS}) == len(CONFIGS) and CHOSEN in {c.name for c in CONFIGS}


@pytest.mark.skipif(not (FIG_DIR / "mlp_validate.json").exists(), reason="run `make mlp`")
def test_mlp_rows_are_validation_only_and_at_matched_coverage():
    """The Day-6 gate: every configuration has a validation row at 95% coverage, and the winner is recorded."""
    summary = json.loads((FIG_DIR / "mlp_validate.json").read_text())
    assert summary["split"] == "validate"
    assert {c.name for c in CONFIGS} | {ENSEMBLE} <= set(summary["methods"])
    for r in [summary["xgboost"], *summary["methods"].values()]:
        assert 0.95 <= r["coverage"] < 0.951
        assert len(r["curve"]) == 20
    assert summary["seeds"][CHOSEN][0]["ratio"] == summary["methods"][CHOSEN]["ratio"]
    if (MODELS_DIR / "manifest.json").exists():
        manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
        assert manifest["winner"] == summary["winner"] and manifest["winner"] in manifest["models"]
        assert manifest["models"]["mlp"]["shift"] == summary["methods"][CHOSEN]["theta"]
        assert manifest["models"][ENSEMBLE]["shift"] == summary["methods"][ENSEMBLE]["theta"]
        assert (MODELS_DIR / "mlp.pt").exists()
