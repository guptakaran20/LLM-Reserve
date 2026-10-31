"""Day-5 models: linear quantile regression and XGBoost, at matched coverage on validation.

Both models predict a high quantile of log1p(output tokens) from the Day-4 features, trained on
the sanitized train split with the pinball (quantile) loss at TRAIN_QUANTILE:

- Linear quantile regression (sklearn `QuantileRegressor`) on the hand-built features, log input
  size and the model one-hot, fitted on a random sample of train completions: the exact LP solver
  is quadratic in rows.
- XGBoost (`reg:quantileerror`) on all train completions, early-stopped on validation pinball loss.
  Feature sets: hand-built alone; plus the raw 384-dim MiniLM-128 embedding; and plus the embedding
  projected onto its first PCA_DIMS principal components (fitted on train prompts), at 128 and 256
  tokens to rerun the Day-4 truncation check on the reservation ratio. Each train prompt repeats in
  ~120 rows (10 completions × 12 models), so on the raw embedding the trees can isolate single
  prompts; the projection and a large `min_child_weight` stop that (docs/findings.md, Day 5).

The matched-coverage knob θ is an additive shift on the log prediction: bound = ceil(expm1(pred + θ)).
It is nondecreasing in θ, so `evaluate.tune` finds the smallest shift reaching 95% coverage on
validation, the same protocol as the baselines. Early stopping and θ both use validation; test is
never loaded here (Day 7 scores it once, with the frozen models and shifts in data/models/).

Writes docs/figures/models_validate.json, models_curve.png and feature_importance.png, and the
fitted models plus their frozen shifts to data/models/. Run with `make models`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor

from llm_reserve.baselines import BAR, COLORS, ORACLE
from llm_reserve.data import ROOT
from llm_reserve.evaluate import SWEEP, TARGET, BoundFn, oracle_gap_closed, score, sweep, tune
from llm_reserve.explore import BLUE, INK_2, INK_3, SURFACE, add_titles
from llm_reserve.features import EMBED_LENGTHS, SERVING_LENGTH, check_no_leaks, load_embeddings, load_features

FIG_DIR = ROOT / "docs" / "figures"
MODELS_DIR = ROOT / "data" / "models"
SPLIT = "validate"
SEED = 0

TRAIN_QUANTILE = 0.95
SHIFT = (-4.0, 4.0)  # range of θ, in log1p tokens: a factor of e^4 ≈ 55 either way
LINEAR_SAMPLE = 50_000  # train completions for the linear model
PCA_DIMS = 32
XGB_PARAMS = {
    "objective": "reg:quantileerror", "quantile_alpha": TRAIN_QUANTILE, "eval_metric": "quantile",
    "tree_method": "hist", "eta": 0.05, "max_depth": 6, "min_child_weight": 300, "subsample": 0.8,
    "colsample_bytree": 0.5, "lambda": 1.0, "seed": SEED, "nthread": -1,
}
XGB_ROUNDS = 4000
XGB_EARLY_STOP = 100

SERVING = f"xgb_minilm{SERVING_LENGTH}"  # the serving candidate: hand-built + PCA of MiniLM-128
# Palette slots 6-8 for the models; the baselines keep their Day-3 colors (COLORS).
MODEL_COLORS = {"linear": "#008300", "xgb_hand": "#6250d6", SERVING: "#e34948"}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    label: str
    embed_length: int | None = None  # MiniLM input length, or None for hand-built features only
    pca_dims: int | None = None  # project the embedding onto this many principal components; None keeps all


@dataclass(frozen=True)
class PromptBlock:
    """Per-prompt embedding features (one row per prompt_row) and their column names."""
    matrix: np.ndarray
    columns: list[str]
    pca: PCA | None = None


def prompt_block(embeddings: np.ndarray, n: int, train_rows: np.ndarray, pca_dims: int | None) -> PromptBlock:
    """The embedding as features: raw, or projected onto principal components fitted on train prompts only."""
    if pca_dims is None:
        return PromptBlock(embeddings, [f"minilm{n}_{i}" for i in range(embeddings.shape[1])])
    pca = PCA(pca_dims, random_state=SEED).fit(embeddings[np.unique(train_rows)])
    return PromptBlock(pca.transform(embeddings).astype(np.float32), [f"minilm{n}_pc{i}" for i in range(pca_dims)],
                       pca)


def design(X: pd.DataFrame, meta: pd.DataFrame, block: PromptBlock | None) -> tuple[np.ndarray, list[str]]:
    """The float32 design matrix and its column names: Day-4 features, then the prompt block if given."""
    cols = list(X.columns)
    mats = [X.to_numpy(np.float32)]
    if block is not None:
        mats.append(block.matrix[meta.prompt_row.to_numpy()])
        cols += block.columns
    check_no_leaks(cols)
    return np.hstack(mats) if len(mats) > 1 else mats[0], cols


def shifted_bounds(pred_log: np.ndarray) -> BoundFn:
    """θ -> ceil(expm1(pred_log + θ)), at least 1 token: a whole-token bound, nondecreasing in θ."""
    pred_log = np.asarray(pred_log, dtype=np.float64)
    return lambda shift: np.maximum(np.ceil(np.expm1(pred_log + shift)), 1.0)


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ---- Linear quantile regression -----------------------------------------------------------------

@dataclass(frozen=True)
class LinearQuantile:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept}


def fit_linear(X: np.ndarray, y_log: np.ndarray, q: float = TRAIN_QUANTILE, sample: int = LINEAR_SAMPLE,
               seed: int = SEED) -> LinearQuantile:
    """Unpenalized linear quantile regression of y_log on standardized X, on a random row sample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs, ys = X[idx].astype(np.float64), y_log[idx]
    mean, scale = Xs.mean(0), Xs.std(0)
    scale[scale == 0] = 1.0
    reg = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit((Xs - mean) / scale, ys)
    return LinearQuantile(mean, scale, reg.coef_, float(reg.intercept_))


# ---- XGBoost ------------------------------------------------------------------------------------

def fit_xgb(X_train: np.ndarray, y_train_log: np.ndarray, X_val: np.ndarray, y_val_log: np.ndarray,
            columns: Sequence[str], params: dict | None = None, rounds: int = XGB_ROUNDS,
            early_stop: int = XGB_EARLY_STOP) -> xgb.Booster:
    """Quantile XGBoost on log length, early-stopped on validation pinball loss. Keeps the best round."""
    dtrain = xgb.QuantileDMatrix(X_train, y_train_log, feature_names=list(columns))
    dval = xgb.QuantileDMatrix(X_val, y_val_log, ref=dtrain, feature_names=list(columns))
    return xgb.train({**XGB_PARAMS, **(params or {})}, dtrain, rounds, evals=[(dval, SPLIT)],
                     early_stopping_rounds=early_stop, verbose_eval=250)


def predict_xgb(booster: xgb.Booster, X: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=list(columns))
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


def feature_group(column: str) -> str:
    """The importance group of a column: the one-hot and the embedding each count as one feature."""
    if column.startswith("model="):
        return "target model (one-hot)"
    if column.startswith("minilm"):
        return f"MiniLM-{column.removeprefix('minilm').split('_')[0]} embedding"
    return column


def grouped_importance(booster: xgb.Booster) -> dict[str, float]:
    """Share of total split gain per feature group, largest first. Shares sum to 1."""
    gain = booster.get_score(importance_type="total_gain")
    groups: dict[str, float] = {}
    for col, g in gain.items():
        groups[feature_group(col)] = groups.get(feature_group(col), 0.0) + g
    total = sum(groups.values())
    return dict(sorted(((k, v / total) for k, v in groups.items()), key=lambda kv: -kv[1]))


# ---- Evaluation ---------------------------------------------------------------------------------

def evaluate_predictions(pred_log: np.ndarray, y: np.ndarray, models: np.ndarray, targets: Sequence[float] = SWEEP,
                         target: float = TARGET) -> dict:
    """Tune the shift to `target` coverage, then sweep the curve; coverage per target model at the tuned shift."""
    bounds = shifted_bounds(pred_log)
    s = score(bounds, y, tune(bounds, y, target, *SHIFT))
    covered = y <= bounds(s.theta)
    by_model = pd.Series(covered).groupby(models).mean()
    return {**s.as_dict(), "curve": [c.as_dict() for c in sweep(bounds, y, targets, *SHIFT)],
            "coverage_by_model": {str(k): float(v) for k, v in by_model.items()}}


def add_gap(results: dict, bar: float, oracle: float) -> None:
    for r in results.values():
        r["vs_bar"] = r["ratio"] / bar - 1
        r["gap_closed"] = oracle_gap_closed(bar, r["ratio"], oracle)


def markdown_table(results: dict, baselines: dict) -> str:
    lines = [
        "| Method | Shift θ | Coverage (sanitized) | Coverage (original) | Reservation ratio | vs. the bar "
        "| Oracle gap closed |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in (BAR, "per_model_input", ORACLE):
        b = baselines[name]
        lines.append(f"| {b['label']} | — | {b['coverage']:.2%} | {b['coverage_original']:.2%} | {b['ratio']:.3f} "
                     f"| {b['vs_bar']:+.1%} | {b['gap_closed']:.1%} |")
    for r in results.values():
        lines.append(f"| {r['label']} | {r['theta']:+.3f} | {r['coverage']:.2%} | {r['coverage_original']:.2%} "
                     f"| {r['ratio']:.3f} | {r['vs_bar']:+.1%} | {r['gap_closed']:.1%} |")
    return "\n".join(lines)


def plot_curve(results: dict, baselines: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    series = [(baselines[n], COLORS[n]) for n in (BAR, "per_model_input", ORACLE)]
    series += [(results[n], c) for n, c in MODEL_COLORS.items()]
    for r, color in series:
        cov = [c["coverage"] * 100 for c in r["curve"]]
        ratio = [c["ratio"] for c in r["curve"]]
        dashes = (0, (4, 2)) if not r.get("deployable", True) else "-"
        ax.plot(cov, ratio, color=color, linewidth=2, linestyle=dashes, label=r["label"])
        ax.plot(r["coverage"] * 100, r["ratio"], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.axvline(TARGET * 100, color=INK_3, linewidth=1, linestyle=":")
    ax.text(TARGET * 100, ax.get_ylim()[1], " matched coverage, 95%", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlim(79.5, 99.5)
    ax.set_xlabel("coverage on validation (%): share of completions within their bound")
    ax.set_ylabel("reservation ratio: reserved ÷ used tokens")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    hidden = [r for n, r in results.items() if n not in MODEL_COLORS]
    note = "".join(f"\nNot shown: {r['label']}, ratio {r['ratio']:.3f} at 95%." for r in hidden)
    add_titles(fig, "Models vs. baselines, validation split (sanitized): lower is better",
               "Each model's log-space shift tuned to 80–99% coverage; dots: tuned to 95%. "
               f"Dashed: the oracle, not deployable.{note}", top=0.87)
    fig.savefig(path)
    plt.close(fig)


def plot_importance(importances: dict[str, dict[str, float]], labels: dict[str, str], path, top: int = 12) -> None:
    fig, axes = plt.subplots(1, len(importances), figsize=(5.2 * len(importances), 5.6))
    for ax, (name, imp) in zip(np.atleast_1d(axes), importances.items()):
        items = list(imp.items())[:top]
        names = [k for k, _ in items][::-1]
        shares = [v * 100 for _, v in items][::-1]
        y = np.arange(len(names))
        ax.barh(y, shares, height=0.7, color=BLUE, edgecolor=SURFACE, linewidth=2)
        for yi, s in zip(y, shares):
            ax.text(s + 0.6, yi, f"{s:.1f}%", va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, names)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(shares) * 1.18)
        ax.set_xlabel("% of total split gain")
        ax.set_title(labels[name], loc="left")
    add_titles(fig, "What the XGBoost models split on",
               f"Share of total gain, top {top} feature groups. The 13-model one-hot and the {PCA_DIMS} embedding "
               "components each count as one group.", top=0.88)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    baselines = json.loads((FIG_DIR / "baselines_validate.json").read_text())["methods"]
    bar, oracle = baselines[BAR]["ratio"], baselines[ORACLE]["ratio"]

    Xtr, ytr, mtr = load_features("train", "sanitized")
    Xva, yva, mva = load_features(SPLIT, "sanitized")
    Xor, yor, mor = load_features(SPLIT, "original")
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    models_va = mva.model.to_numpy()

    results, importances, manifest = {}, {}, {"train_quantile": TRAIN_QUANTILE, "models": {}}

    def record(name: str, label: str, pred_va: np.ndarray, pred_or: np.ndarray, extra: dict) -> None:
        r = {"label": label, "deployable": True, **evaluate_predictions(pred_va, yva, models_va)}
        r["coverage_original"] = score(shifted_bounds(pred_or), yor, r["theta"]).coverage
        r["pinball_validate"] = pinball(yva_log, pred_va, TRAIN_QUANTILE)
        results[name] = {**r, **extra}
        print(f"{label:40} θ {r['theta']:+.3f}  coverage {r['coverage']:.4f}  ratio {r['ratio']:.3f}")

    # Linear quantile regression, hand-built features.
    A_tr, cols = design(Xtr, mtr, None)
    t = time.time()
    lin = fit_linear(A_tr, ytr_log)
    fit_s = time.time() - t
    record("linear", "Linear quantile regression (hand-built)", lin.predict(design(Xva, mva, None)[0]),
           lin.predict(design(Xor, mor, None)[0]),
           {"n_features": len(cols), "train_rows": min(LINEAR_SAMPLE, len(A_tr)), "fit_seconds": fit_s})
    (MODELS_DIR / "linear.json").write_text(json.dumps({"columns": cols, **lin.as_dict()}) + "\n")
    manifest["models"]["linear"] = {"file": "linear.json", "embed_length": None, "columns": cols,
                                    "shift": results["linear"]["theta"]}

    feature_sets = [
        FeatureSet("xgb_hand", "XGBoost (hand-built)"),
        FeatureSet(f"xgb_minilm{SERVING_LENGTH}_raw", f"XGBoost (hand-built + raw MiniLM-{SERVING_LENGTH})",
                   SERVING_LENGTH),
        *(FeatureSet(f"xgb_minilm{n}", f"XGBoost (hand-built + MiniLM-{n}, PCA-{PCA_DIMS})", n, PCA_DIMS)
          for n in EMBED_LENGTHS),
    ]
    train_rows = mtr.prompt_row.to_numpy()
    for fs in feature_sets:
        block = None
        if fs.embed_length is not None:
            block = prompt_block(load_embeddings(fs.embed_length), fs.embed_length, train_rows, fs.pca_dims)
        A_tr, cols = design(Xtr, mtr, block)
        A_va, _ = design(Xva, mva, block)
        t = time.time()
        booster = fit_xgb(A_tr, ytr_log, A_va, yva_log, cols)
        fit_s = time.time() - t
        del A_tr
        pred_or = predict_xgb(booster, design(Xor, mor, block)[0], cols)
        record(fs.name, fs.label, predict_xgb(booster, A_va, cols), pred_or,
               {"n_features": len(cols), "train_rows": len(ytr), "best_iteration": booster.best_iteration,
                "fit_seconds": fit_s})
        importances[fs.name] = grouped_importance(booster)
        results[fs.name]["importance"] = importances[fs.name]
        booster.save_model(MODELS_DIR / f"{fs.name}.ubj")
        manifest["models"][fs.name] = {"file": f"{fs.name}.ubj", "embed_length": fs.embed_length, "columns": cols,
                                       "shift": results[fs.name]["theta"], "best_iteration": booster.best_iteration}
        if block is not None and block.pca is not None:
            pca_file = f"pca_minilm{fs.embed_length}_{fs.pca_dims}.npz"
            np.savez(MODELS_DIR / pca_file, mean=block.pca.mean_, components=block.pca.components_)
            manifest["models"][fs.name]["pca_file"] = pca_file
        del A_va

    add_gap(results, bar, oracle)
    summary = {"split": SPLIT, "target": TARGET, "train_quantile": TRAIN_QUANTILE, "pca_dims": PCA_DIMS,
               "linear_sample": LINEAR_SAMPLE, "xgb_params": XGB_PARAMS,
               "bar_ratio": bar, "oracle_ratio": oracle, "methods": results}
    (FIG_DIR / "models_validate.json").write_text(json.dumps(summary, indent=2) + "\n")
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_curve(results, baselines, FIG_DIR / "models_curve.png")
    plot_importance({n: importances[n] for n in ("xgb_hand", SERVING)},
                    {fs.name: fs.label for fs in feature_sets}, FIG_DIR / "feature_importance.png")
    print(markdown_table(results, baselines))
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/ and {MODELS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
