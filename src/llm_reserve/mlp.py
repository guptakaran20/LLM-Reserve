"""Day-6 neural network: a PyTorch MLP with the pinball loss, then the winning model is picked.

The MLP predicts the TRAIN_QUANTILE (95%) quantile of log1p(output tokens), like the Day-5 models,
from the same request-time features: the hand-built prompt features, log input size, the 13-model
one-hot and the MiniLM-128 embedding. Two hidden layers (256 and 64 units, ReLU, dropout), one
output. The one-hot enters the first layer with the prompt features, so the hidden units can let a
prompt feature mean different things for different models (findings 2 and 14), which a per-model
output offset can't.

Training is done at the sample level. All completions of one (prompt, model) sample share their
features, so the matrix is collapsed to one row per sample with its (up to 10) log lengths, and
the loss is the mean pinball loss over completions: the same objective as one row per completion,
10× cheaper. Each train prompt still repeats across 12 models, and train has only ~8k prompts, so
the network memorizes prompts within a few epochs. Dropout and weight decay (AdamW) barely help;
dropout on the inputs, with the embedding projected onto the Day-5 PCA, is what works. Early
stopping on validation pinball loss keeps the best epoch.

Standardization is baked into the module (buffers), so Day 11 exports a single graph.

The matched-coverage protocol is unchanged: bound = ceil(expm1(pred + θ)), θ tuned on validation
to the smallest shift reaching 95% coverage. Every configuration tried is listed in CONFIGS and
reported. The chosen configuration is retrained with SEEDS to show how much of a difference is
seed noise. The winner is picked from the Day-5 XGBoost, the MLP and their average in log space,
by validation ratio. Test is never loaded here.

Writes docs/figures/mlp_validate.json and mlp_curve.png, data/models/mlp.pt, and the MLP entry plus
the winner to data/models/manifest.json. Needs `make models` first. Run with `make mlp`.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from llm_reserve.baselines import BAR, COLORS, ORACLE
from llm_reserve.data import ROOT
from llm_reserve.evaluate import TARGET, score
from llm_reserve.explore import INK_2, INK_3, SURFACE, add_titles
from llm_reserve.features import SERVING_LENGTH, load_embeddings, load_features
from llm_reserve.models import (
    FIG_DIR, MODELS_DIR, PCA_DIMS, SERVING, SPLIT, TRAIN_QUANTILE, VALIDATION_PREDICTIONS, PromptBlock, add_gap,
    design, evaluate_predictions, pinball, prompt_block, shifted_bounds,
)

SEED = 0
THREADS = 4  # leave the rest of the machine usable; training takes about 3 minutes
SEEDS = (0, 1, 2)  # the chosen configuration is retrained with each, to measure seed noise


@dataclass(frozen=True)
class MLPConfig:
    name: str
    label: str
    embedding: str | None = "raw"  # "raw" 384 dims, "pca" (PCA_DIMS components), or None
    hidden: tuple[int, ...] = (256, 64)
    dropout: float = 0.2
    input_dropout: float = 0.0  # dropout on the standardized inputs
    weight_decay: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 512  # samples, ~10 completions each
    max_epochs: int = 200
    patience: int = 12  # epochs without a better validation loss
    lr_patience: int = 4  # halve the learning rate after this many


# ---- Sample-level data ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Samples:
    """One row per (prompt, model) sample: features, log lengths (NaN-padded) and the row -> sample map."""
    X: np.ndarray  # (n_samples, n_features) float32
    Y: np.ndarray  # (n_samples, max_completions) float32 log1p(output tokens), NaN where missing
    codes: np.ndarray  # (n_rows,) sample index of each completion row
    columns: list[str]


def collapse(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, block: PromptBlock | None) -> Samples:
    """Collapse completion rows to samples. Features are constant within a sample (tests/test_features.py)."""
    codes, _ = pd.factorize(meta.sample_id)
    first = np.unique(codes, return_index=True)[1]
    A, cols = design(X.iloc[first].reset_index(drop=True), meta.iloc[first].reset_index(drop=True), block)
    pos = pd.Series(codes).groupby(codes).cumcount().to_numpy()
    Y = np.full((len(first), pos.max() + 1), np.nan, dtype=np.float32)
    Y[codes, pos] = np.log1p(y)
    return Samples(A, Y, codes, cols)


# ---- Model and loss -------------------------------------------------------------------------------

class QuantileMLP(nn.Module):
    """Standardize, then Linear-ReLU-Dropout blocks, then one output: the log1p quantile."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, hidden=(256, 64), dropout: float = 0.2,
                 input_dropout: float = 0.0):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        layers, d = ([nn.Dropout(input_dropout)] if input_dropout else []), len(mean)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale).squeeze(-1)


def pinball_loss(pred: torch.Tensor, Y: torch.Tensor, q: float) -> torch.Tensor:
    """Mean pinball loss over the observed completions: pred is (n,), Y is (n, k) with NaN padding."""
    mask = ~torch.isnan(Y)
    diff = torch.where(mask, Y - pred[:, None], torch.zeros_like(Y))
    return torch.maximum(q * diff, (q - 1) * diff).sum() / mask.sum()


def standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = X.mean(0), X.std(0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


@torch.no_grad()
def predict_mlp(model: QuantileMLP, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    xs = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([model(xs[i:i + batch]) for i in range(0, len(xs), batch)]).numpy()


@torch.no_grad()
def eval_loss(model: QuantileMLP, X: torch.Tensor, Y: torch.Tensor, q: float, batch: int = 8192) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(X), batch):
        m = int((~torch.isnan(Y[i:i + batch])).sum())
        total += float(pinball_loss(model(X[i:i + batch]), Y[i:i + batch], q)) * m
        n += m
    return total / n


def fit_mlp(train: Samples, val: Samples, cfg: MLPConfig, seed: int = SEED, q: float = TRAIN_QUANTILE,
            verbose: bool = True) -> tuple[QuantileMLP, list[dict]]:
    """Mini-batch AdamW on the pinball loss, early-stopped on validation loss. Returns the best epoch's model."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = QuantileMLP(*standardizer(train.X), cfg.hidden, cfg.dropout, cfg.input_dropout)
    with torch.no_grad():  # start at the train quantile, so early epochs learn the shape, not the level
        model.net[-1].bias.fill_(float(np.nanquantile(train.Y, q)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg.lr_patience)
    Xtr, Ytr = torch.from_numpy(train.X.copy()), torch.from_numpy(train.Y.copy())
    Xva, Yva = torch.from_numpy(val.X.copy()), torch.from_numpy(val.Y.copy())

    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = pinball_loss(model(Xtr[idx]), Ytr[idx], q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            n += len(idx)
        val_loss = eval_loss(model, Xva, Yva, q)
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "lr": opt.param_groups[0]["lr"]})
        if val_loss < best - 1e-6:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  train {total / n:.5f}  val {val_loss:.5f}  lr {opt.param_groups[0]['lr']:.1e}")
        if wait >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# ---- Configurations, all reported -----------------------------------------------------------------

def label(cfg: MLPConfig) -> str:
    """'MLP (hand-built + MiniLM-128 PCA-32; input dropout 0.2)': the inputs, then every knob off its default."""
    inputs = {None: "hand-built", "raw": f"hand-built + raw MiniLM-{SERVING_LENGTH}",
              "pca": f"hand-built + MiniLM-{SERVING_LENGTH} PCA-{PCA_DIMS}"}[cfg.embedding]
    default = MLPConfig("", "")
    names = {"input_dropout": "input dropout", "dropout": "dropout", "weight_decay": "weight decay", "lr": "lr",
             "batch_size": "batch"}
    knobs = [f"{text} {getattr(cfg, k):g}" for k, text in names.items() if getattr(cfg, k) != getattr(default, k)]
    return f"MLP ({inputs}{'; ' + ', '.join(knobs) if knobs else ''})"


def config(name: str, **kw) -> MLPConfig:
    cfg = MLPConfig(name, "", **kw)
    return replace(cfg, label=label(cfg))


# In the order they were tried on validation (seed 0). Defaults: dropout 0.2 after each hidden layer,
# weight decay 1e-4, lr 1e-3, batch 512 samples. docs/findings.md (Day 6) discusses them.
CONFIGS = (
    config("mlp_hand", embedding=None),
    config("mlp_raw"),
    config("mlp_raw_d5", dropout=0.5),
    config("mlp_raw_wd2", weight_decay=1e-2),
    config("mlp_raw_wd1", weight_decay=0.1),
    config("mlp_raw_lr4", lr=1e-4),
    config("mlp_raw_b4096", batch_size=4096),
    config("mlp_pca", embedding="pca"),
    config("mlp_pca_d5", embedding="pca", dropout=0.5),
    config("mlp_raw_d7", dropout=0.7),
    config("mlp_raw_in3_d5", input_dropout=0.3, dropout=0.5),
    config("mlp_pca_d3_wd2", embedding="pca", dropout=0.3, weight_decay=1e-2),
    config("mlp_pca_in2", embedding="pca", input_dropout=0.2),
    config("mlp_pca_in1", embedding="pca", input_dropout=0.1),
    config("mlp_pca_in3", embedding="pca", input_dropout=0.3),
    config("mlp_pca_in4", embedding="pca", input_dropout=0.4),
    config("mlp_pca_in2_d0", embedding="pca", input_dropout=0.2, dropout=0.0),
    config("mlp_raw_in5", input_dropout=0.5),
    config("mlp_hand_in2", embedding=None, input_dropout=0.2),
)
# Frozen after the sweep: the lowest validation pinball loss (tied with input dropout 0.1, which has a
# worse ratio), in the middle of the 0.1-0.3 plateau, and stable across seeds.
CHOSEN = "mlp_pca_in2"
SEED_STUDY = (CHOSEN, "mlp_pca")  # retrained with every seed in SEEDS: the best model with and without input dropout
SEED_AVG = "mlp_seed_avg"
ENSEMBLE = "mlp_xgb_avg"


def xgb_pca() -> dict:
    """The frozen Day-5 projection (mean, components), shared by XGBoost and the MLP."""
    m = json.loads((MODELS_DIR / "manifest.json").read_text())["models"][SERVING]
    p = np.load(MODELS_DIR / m["pca_file"])
    return {"file": m["pca_file"], "mean": p["mean"], "components": p["components"]}


def plot_curve(results: dict, baselines: dict, xgb_row: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    series = [(baselines[n], COLORS[n]) for n in (BAR, ORACLE)]
    series += [(xgb_row, "#e34948"), (results["mlp_raw"], "#eda100"), (results[CHOSEN], "#6250d6"),
               (results[ENSEMBLE], "#008300")]
    for r, color in series:
        cov = [c["coverage"] * 100 for c in r["curve"]]
        ratio = [c["ratio"] for c in r["curve"]]
        ax.plot(cov, ratio, color=color, linewidth=2, linestyle="-" if r.get("deployable", True) else (0, (4, 2)),
                label=r["label"])
        ax.plot(r["coverage"] * 100, r["ratio"], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.axvline(TARGET * 100, color=INK_3, linewidth=1, linestyle=":")
    ax.text(TARGET * 100, ax.get_ylim()[1], " matched coverage, 95%", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlim(79.5, 99.5)
    ax.set_xlabel("coverage on validation (%): share of completions within their bound")
    ax.set_ylabel("reservation ratio: reserved ÷ used tokens")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    add_titles(fig, "MLP vs. XGBoost, validation split (sanitized): lower is better",
               "Each model's log-space shift tuned to 80–99% coverage; dots: tuned to 95%. "
               "Dashed: the oracle, not deployable.", top=0.87)
    fig.savefig(path)
    plt.close(fig)


def markdown_table(results: dict, baselines: dict, xgb_row: dict) -> str:
    lines = ["| Method | Shift θ | Coverage (original) | Validation pinball | Best epoch | Reservation ratio "
             "| vs. the bar | Oracle gap closed |", "|---|---|---|---|---|---|---|---|"]
    b = baselines[BAR]
    lines.append(f"| {b['label']} | — | {b['coverage_original']:.2%} | — | — | {b['ratio']:.3f} | — | 0% |")
    for r in (xgb_row, *results.values()):
        epoch = r.get("best_epoch")
        lines.append(f"| {r['label']} | {r['theta']:+.3f} | {r['coverage_original']:.2%} "
                     f"| {r['pinball_validate']:.5f} | {'—' if epoch is None else epoch + 1} | {r['ratio']:.3f} "
                     f"| {r['vs_bar']:+.1%} | {r['gap_closed']:.1%} |")
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(THREADS)
    baselines = json.loads((FIG_DIR / "baselines_validate.json").read_text())["methods"]
    day5 = json.loads((FIG_DIR / "models_validate.json").read_text())["methods"]
    bar, oracle = baselines[BAR]["ratio"], baselines[ORACLE]["ratio"]

    Xtr, ytr, mtr = load_features("train", "sanitized")
    Xva, yva, mva = load_features(SPLIT, "sanitized")
    Xor, yor, mor = load_features(SPLIT, "original")
    yva_log, models_va = np.log1p(yva), mva.model.to_numpy()
    emb = load_embeddings(SERVING_LENGTH)
    pca = xgb_pca()
    blocks = {None: None, "raw": prompt_block(emb, SERVING_LENGTH, mtr.prompt_row.to_numpy(), None),
              "pca": prompt_block(emb, SERVING_LENGTH, mtr.prompt_row.to_numpy(), PCA_DIMS)}
    # The MLP's projection is the one XGBoost was trained with (same train prompts, same seed), so serving needs one.
    assert np.allclose(blocks["pca"].pca.components_, pca["components"], atol=1e-6)
    data = {kind: tuple(collapse(X, y, m, block) for X, y, m in ((Xtr, ytr, mtr), (Xva, yva, mva), (Xor, yor, mor)))
            for kind, block in blocks.items()}

    results, preds, fitted = {}, {}, {}

    def record(name: str, label: str, pred_va: np.ndarray, pred_or: np.ndarray, extra: dict) -> dict:
        r = {"label": label, "deployable": True, **evaluate_predictions(pred_va, yva, models_va)}
        r["coverage_original"] = score(shifted_bounds(pred_or), yor, r["theta"]).coverage
        r["pinball_validate"] = pinball(yva_log, pred_va, TRAIN_QUANTILE)
        preds[name] = (pred_va, pred_or)
        print(f"{label:70} θ {r['theta']:+.3f}  pinball {r['pinball_validate']:.5f}  ratio {r['ratio']:.3f}")
        return {**r, **extra}

    def run(cfg: MLPConfig, seed: int) -> tuple[QuantileMLP, dict, np.ndarray, np.ndarray]:
        tr, va, orig = data[cfg.embedding]
        t = time.time()
        model, history = fit_mlp(tr, va, cfg, seed, verbose=False)
        best = int(np.argmin([h["val_loss"] for h in history]))
        extra = {"config": {k: v for k, v in asdict(cfg).items() if k not in ("name", "label")}, "seed": seed,
                 "n_features": tr.X.shape[1], "train_samples": len(tr.X), "best_epoch": best,
                 "epochs_run": len(history), "fit_seconds": time.time() - t, "history": history}
        return model, extra, predict_mlp(model, va.X)[va.codes], predict_mlp(model, orig.X)[orig.codes]

    # Day-5 XGBoost, rescored from the validation predictions `make models` saved. xgboost is not imported
    # here: next to torch on macOS, its OpenMP runtime segfaults or deadlocks (docs/findings.md, Day 6).
    saved = np.load(MODELS_DIR / VALIDATION_PREDICTIONS)
    xgb_va, xgb_or = saved["sanitized"], saved["original"]
    assert len(xgb_va) == len(yva) and len(xgb_or) == len(yor), "stale predictions: rerun `make models`"
    xgb_row = record(SERVING, day5[SERVING]["label"], xgb_va, xgb_or, {})
    assert abs(xgb_row["ratio"] - day5[SERVING]["ratio"]) < 1e-9, "the saved XGBoost does not reproduce Day 5"

    for cfg in CONFIGS:
        model, extra, pred_va, pred_or = run(cfg, SEED)
        results[cfg.name] = record(cfg.name, cfg.label, pred_va, pred_or, extra)
        fitted[cfg.name] = model

    # Seed noise: retrain with every seed. Seed 0 is the frozen model; the seed average is reported, not chosen.
    seed_study = {}
    for name in SEED_STUDY:
        cfg = next(c for c in CONFIGS if c.name == name)
        runs = {SEED: preds[name]}
        for seed in SEEDS:
            if seed not in runs:
                runs[seed] = run(cfg, seed)[2:]
        rows = []
        for seed, (pred_va, _) in runs.items():
            r = evaluate_predictions(pred_va, yva, models_va, targets=(TARGET,))
            rows.append({"seed": seed, "ratio": r["ratio"], "theta": r["theta"],
                         "pinball_validate": pinball(yva_log, pred_va, TRAIN_QUANTILE)})
        seed_study[name] = rows
        print(f"  {name} seeds: " + ", ".join(f"{r['ratio']:.3f}" for r in rows))
        if name == CHOSEN:
            avg_va, avg_or = (np.mean([p[i] for p in runs.values()], axis=0) for i in (0, 1))
            results[SEED_AVG] = record(SEED_AVG, f"Average of {len(runs)} seeds of {results[CHOSEN]['label']}",
                                       avg_va, avg_or, {"members": [f"{CHOSEN} seed {s}" for s in SEEDS]})

    # The equal-weight average of the MLP and XGBoost, in log space.
    mlp_va, mlp_or = preds[CHOSEN]
    results[ENSEMBLE] = record(ENSEMBLE, f"Average of the MLP (PCA-{PCA_DIMS}; input dropout 0.2) and XGBoost, "
                               "log space", (mlp_va + xgb_va) / 2, (mlp_or + xgb_or) / 2,
                               {"members": [CHOSEN, SERVING]})

    add_gap(results, bar, oracle)
    add_gap({SERVING: xgb_row}, bar, oracle)
    # The winner: the lowest validation ratio among single models and the MLP + XGBoost average. The seed
    # average is reported but not a candidate (seed 1 and 2 models are not saved).
    candidates = {SERVING: xgb_row["ratio"], CHOSEN: results[CHOSEN]["ratio"], ENSEMBLE: results[ENSEMBLE]["ratio"]}
    winner = {SERVING: SERVING, CHOSEN: "mlp", ENSEMBLE: ENSEMBLE}[min(candidates, key=candidates.get)]

    columns = data["pca"][0].columns
    torch.save({"state_dict": fitted[CHOSEN].state_dict(), "columns": columns,
                "config": results[CHOSEN]["config"]}, MODELS_DIR / "mlp.pt")
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    manifest["models"]["mlp"] = {"file": "mlp.pt", "embed_length": SERVING_LENGTH, "pca_file": pca["file"],
                                 "columns": columns, "shift": results[CHOSEN]["theta"], "config": CHOSEN,
                                 "best_epoch": results[CHOSEN]["best_epoch"]}
    manifest["models"][ENSEMBLE] = {"members": ["mlp", SERVING], "combine": "mean of log1p predictions",
                                    "shift": results[ENSEMBLE]["theta"]}
    manifest["winner"] = winner
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {"split": SPLIT, "target": TARGET, "train_quantile": TRAIN_QUANTILE, "bar_ratio": bar,
               "oracle_ratio": oracle, "chosen": CHOSEN, "winner": winner, "seeds": seed_study,
               "xgboost": xgb_row, "methods": results}
    (FIG_DIR / "mlp_validate.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_curve(results, baselines, xgb_row, FIG_DIR / "mlp_curve.png")
    print(markdown_table(results, baselines, xgb_row))
    print(f"winner: {winner}")
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/ and {MODELS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
