"""Day-7 inference: the frozen models' log predictions for validate and test, saved for scoring.

Nothing here is fitted. Every model is loaded from data/models/ as Day 5 and Day 6 left it, with the
frozen MiniLM-128 PCA from its saved file (the exact sklearn formula), so this is the code path that
Day 11 must reproduce in ONNX.

torch and xgboost can't share a process on macOS (docs/findings.md, Day 6), so inference runs twice:

    python -m llm_reserve.predict xgb   # linear quantile regression and XGBoost: never imports torch
    python -m llm_reserve.predict mlp   # the MLP: torch first, xgboost imported but never run (Day 6)

Each writes data/predictions/{model}_{split}.npz holding `log_pred`, one value per row of
load_features(split, "original"); the sanitized rows are its ~is_degenerate subset. Before test is
predicted, each model's validation predictions must reproduce the numbers it was frozen with: the
Day-5 saved XGBoost predictions, and every model's validation ratio at its frozen shift. So the
artifacts scored on test are the ones that were chosen on validation.

Run with `make eval`, which also runs llm_reserve.lodo and llm_reserve.holdout.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from llm_reserve.data import ROOT
from llm_reserve.evaluate import score
from llm_reserve.features import SERVING_LENGTH, load_embeddings, load_features

MODELS_DIR = ROOT / "data" / "models"
FIG_DIR = ROOT / "docs" / "figures"
PRED_DIR = ROOT / "data" / "predictions"
SPLITS = ("validate", "test")
FAMILIES = {"xgb": ("linear", "xgb_hand", f"xgb_minilm{SERVING_LENGTH}"), "mlp": ("mlp",)}
TOL = 1e-6  # validation ratio drift allowed when a frozen model is reloaded


def manifest() -> dict:
    return json.loads((MODELS_DIR / "manifest.json").read_text())


def pca_block(pca_file: str, embed_length: int = SERVING_LENGTH):
    """The frozen PCA projection as a PromptBlock, computed as sklearn's PCA.transform does it."""
    from llm_reserve.models import PromptBlock

    p = np.load(MODELS_DIR / pca_file)
    mean, comps = p["mean"], p["components"]
    emb = load_embeddings(embed_length)
    matrix = (emb @ comps.T - mean[None, :] @ comps.T).astype(np.float32)
    return PromptBlock(matrix, [f"minilm{embed_length}_pc{i}" for i in range(len(comps))])


def path(name: str, split: str):
    return PRED_DIR / f"{name}_{split}.npz"


def load(name: str, split: str) -> np.ndarray:
    """Saved log predictions, one per row of load_features(split, "original")."""
    return np.load(path(name, split))["log_pred"]


def frozen_ratio(name: str) -> float:
    """The validation reservation ratio each model was frozen with (Day 5 or Day 6)."""
    if name == "mlp":
        day6 = json.loads((FIG_DIR / "mlp_validate.json").read_text())
        return day6["methods"][day6["chosen"]]["ratio"]
    return json.loads((FIG_DIR / "models_validate.json").read_text())["methods"][name]["ratio"]


def check_reproduces(name: str, log_pred: np.ndarray, y_sanitized: np.ndarray, is_degenerate: np.ndarray,
                     shift: float) -> float:
    """Raise unless the reloaded model's validation ratio at its frozen shift matches Day 5 / Day 6."""
    from llm_reserve.models import shifted_bounds

    ratio = score(shifted_bounds(log_pred[~is_degenerate]), y_sanitized, shift).ratio
    expected = frozen_ratio(name)
    if abs(ratio - expected) > TOL:
        raise AssertionError(f"{name}: reloaded model gives validation ratio {ratio:.6f}, frozen {expected:.6f}")
    return ratio


def predict_xgb_family() -> None:
    """Linear and XGBoost predictions. Only xgboost (no torch) is loaded in this process."""
    import xgboost as xgb

    from llm_reserve.models import LinearQuantile, design, predict_xgb

    m = manifest()["models"]
    block = pca_block(m[f"xgb_minilm{SERVING_LENGTH}"]["pca_file"])
    lin = json.loads((MODELS_DIR / "linear.json").read_text())
    linear = LinearQuantile(np.asarray(lin["mean"]), np.asarray(lin["scale"]), np.asarray(lin["coef"]),
                            float(lin["intercept"]))
    boosters = {}
    for name in FAMILIES["xgb"][1:]:
        boosters[name] = xgb.Booster()
        boosters[name].load_model(MODELS_DIR / m[name]["file"])
        if boosters[name].best_iteration != m[name]["best_iteration"]:  # predict_xgb stops at the best round
            raise AssertionError(f"{name}: saved best_iteration differs from the manifest")

    def predict(name: str, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        A, cols = design(X, meta, block if m[name]["embed_length"] else None)
        if cols != m[name]["columns"]:
            raise AssertionError(f"{name}: feature columns differ from the frozen model's")
        return linear.predict(A) if name == "linear" else predict_xgb(boosters[name], A, cols)

    for split in SPLITS:
        X, y, meta = load_features(split, "original")
        deg = meta.is_degenerate.to_numpy()
        for name in FAMILIES["xgb"]:
            pred = predict(name, X, meta)
            if split == "validate":
                ratio = check_reproduces(name, pred, y[~deg], deg, m[name]["shift"])
                if name == f"xgb_minilm{SERVING_LENGTH}":  # the exact predictions Day 6 averaged with the MLP
                    saved = np.load(MODELS_DIR / f"{name}_validate_pred.npz")
                    np.testing.assert_allclose(pred[~deg], saved["sanitized"], rtol=0, atol=1e-6)
                    np.testing.assert_allclose(pred, saved["original"], rtol=0, atol=1e-6)
                print(f"{name:16} validation ratio {ratio:.4f}: reproduces the frozen model")
            np.savez(path(name, split), log_pred=pred)
        print(f"wrote {split} predictions for {', '.join(FAMILIES['xgb'])}")


def load_mlp(file: str = "mlp.pt"):
    """The frozen MLP and its input columns, rebuilt from the saved config (standardization is in the weights)."""
    import torch

    from llm_reserve.mlp import QuantileMLP

    ckpt = torch.load(MODELS_DIR / file, weights_only=True)
    cfg, state = ckpt["config"], ckpt["state_dict"]
    model = QuantileMLP(state["mean"].numpy(), state["scale"].numpy(), tuple(cfg["hidden"]), cfg["dropout"],
                        cfg["input_dropout"])
    model.load_state_dict(state)
    return model.eval(), list(ckpt["columns"])


def predict_mlp_family() -> None:
    import torch

    from llm_reserve.mlp import THREADS, collapse, predict_mlp

    torch.set_num_threads(THREADS)
    m = manifest()["models"]["mlp"]
    model, columns = load_mlp(m["file"])
    block = pca_block(m["pca_file"])
    for split in SPLITS:
        X, y, meta = load_features(split, "original")
        samples = collapse(X, y, meta, block)
        if samples.columns != columns:
            raise AssertionError("mlp: feature columns differ from the frozen model's")
        pred = predict_mlp(model, samples.X)[samples.codes]
        if split == "validate":
            deg = meta.is_degenerate.to_numpy()
            ratio = check_reproduces("mlp", pred, y[~deg], deg, m["shift"])
            print(f"mlp              validation ratio {ratio:.4f}: reproduces the frozen model")
        np.savez(path("mlp", split), log_pred=pred)
        print(f"wrote {split} predictions for mlp")


def main(argv: list[str]) -> None:
    if len(argv) != 1 or argv[0] not in FAMILIES:
        raise SystemExit(f"usage: python -m llm_reserve.predict {{{','.join(FAMILIES)}}}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    {"xgb": predict_xgb_family, "mlp": predict_mlp_family}[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
