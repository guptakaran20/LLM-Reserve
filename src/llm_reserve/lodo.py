"""Day-7 leave-one-dataset-out: did the model learn what prompts ask for, or which benchmark they come from?

The embedding names a prompt's source dataset 77% of the time from CASTILLO's templates (finding 10),
so part of the test gain could be benchmark recognition, which real gateway traffic would not reward.
For each of the 7 source datasets D, the frozen design is retrained with D removed everywhere it
could leak in:

- the MiniLM-128 PCA is fitted on train prompts outside D,
- XGBoost and the MLP (frozen hyperparameters, seed 0) train on train rows outside D,
- early stopping uses validation rows outside D.

Each run saves log predictions for every validate and test row (original config). llm_reserve.holdout
tunes the shift on validation rows outside D and scores the test rows inside D: unseen prompts from
a benchmark the model has never seen. Nothing here is chosen from the result; the design is the one
frozen on Day 6.

Holding a dataset out also removes 9–19% of the train prompts. A control run (CONTROL) removes the
same share, one seventh, at random from every dataset, and early-stops on all of validation, so the
cost of less data can be told apart from the cost of an unseen benchmark.

As in llm_reserve.predict, the two families run in separate processes (torch and xgboost can't share
one on macOS):

    python -m llm_reserve.lodo xgb
    python -m llm_reserve.lodo mlp

Writes data/lodo/{family}_{dataset}.npz. Run with `make eval`.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from llm_reserve.data import ROOT, load_prompts
from llm_reserve.features import SERVING_LENGTH, load_embeddings, load_features

LODO_DIR = ROOT / "data" / "lodo"
FAMILIES = ("xgb", "mlp")
SEED = 0
CONTROL = "random"  # not a dataset: one seventh of the train prompts, drawn at random across datasets
# (split, config) of each part: train and early stopping use sanitized; predictions cover every original row.
PARTS = {"train": ("train", "sanitized"), "early_stop": ("validate", "sanitized"),
         "validate": ("validate", "original"), "test": ("test", "original")}


def datasets() -> list[str]:
    return sorted(load_prompts().dataset.unique())


def control_prompts(prompts: pd.DataFrame, seed: int = SEED) -> set[str]:
    """The control's held-out prompts: as many train prompts as an average dataset has, drawn at random."""
    train = prompts[prompts.split == "train"]
    return set(train.prompt_id.sample(round(len(train) / train.dataset.nunique()), random_state=seed))


def dataset_of(meta: pd.DataFrame, prompts: pd.DataFrame) -> np.ndarray:
    """The source dataset of each row. Only to split rows here; it never becomes a feature."""
    return meta.prompt_id.map(prompts.set_index("prompt_id").dataset).to_numpy()


def path(family: str, dataset: str):
    return LODO_DIR / f"{family}_{dataset}.npz"


def load(family: str, dataset: str) -> dict[str, np.ndarray]:
    """{"validate": log predictions, "test": log predictions, plus run info}, rows as load_features(split, "original")."""
    with np.load(path(family, dataset)) as f:
        return {k: f[k] for k in f.files}


def load_parts() -> dict[str, tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]]:
    """Each part as (X, y, meta, source dataset per row, whether the row is one of the control's prompts)."""
    prompts = load_prompts()
    control = control_prompts(prompts)
    out = {}
    for part, (split, config) in PARTS.items():
        X, y, meta = load_features(split, config)
        out[part] = (X, y, meta, dataset_of(meta, prompts), meta.prompt_id.isin(control).to_numpy())
    return out


def without(part: tuple, held_out: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """The rows of a part outside `held_out`: a source dataset, or the control's prompts (CONTROL)."""
    X, y, meta, ds, is_control = part
    keep = ~is_control if held_out == CONTROL else ds != held_out
    return X[keep].reset_index(drop=True), y[keep], meta[keep].reset_index(drop=True)


def run_xgb(parts: dict, dataset: str) -> tuple[dict[str, np.ndarray], dict]:
    from llm_reserve.models import PCA_DIMS, design, fit_xgb, predict_xgb, prompt_block

    Xtr, ytr, mtr = without(parts["train"], dataset)
    Xes, yes, mes = without(parts["early_stop"], dataset)
    block = prompt_block(load_embeddings(SERVING_LENGTH), SERVING_LENGTH, mtr.prompt_row.to_numpy(), PCA_DIMS)
    A_tr, cols = design(Xtr, mtr, block)
    t = time.time()
    booster = fit_xgb(A_tr, np.log1p(ytr), design(Xes, mes, block)[0], np.log1p(yes), cols)
    info = {"best_round": booster.best_iteration, "train_rows": len(ytr), "fit_seconds": time.time() - t}
    preds = {split: predict_xgb(booster, design(parts[split][0], parts[split][2], block)[0], cols)
             for split in ("validate", "test")}
    return preds, info


def run_mlp(parts: dict, dataset: str) -> tuple[dict[str, np.ndarray], dict]:
    from llm_reserve.mlp import CHOSEN, CONFIGS, collapse, fit_mlp, predict_mlp
    from llm_reserve.models import PCA_DIMS, prompt_block

    cfg = next(c for c in CONFIGS if c.name == CHOSEN)
    Xtr, ytr, mtr = without(parts["train"], dataset)
    block = prompt_block(load_embeddings(SERVING_LENGTH), SERVING_LENGTH, mtr.prompt_row.to_numpy(), PCA_DIMS)
    train = collapse(Xtr, ytr, mtr, block)
    t = time.time()
    model, history = fit_mlp(train, collapse(*without(parts["early_stop"], dataset), block), cfg, SEED, verbose=False)
    info = {"best_round": int(np.argmin([h["val_loss"] for h in history])), "train_rows": int(len(ytr)),
            "fit_seconds": time.time() - t}
    preds = {}
    for split in ("validate", "test"):
        X, y, meta = parts[split][:3]
        s = collapse(X, y, meta, block)
        preds[split] = predict_mlp(model, s.X)[s.codes]
    return preds, info


def main(argv: list[str]) -> None:
    if len(argv) != 1 or argv[0] not in FAMILIES:
        raise SystemExit(f"usage: python -m llm_reserve.lodo {{{','.join(FAMILIES)}}}")
    family = argv[0]
    if family == "mlp":  # torch before anything that imports xgboost (docs/findings.md, Day 6)
        import torch

        from llm_reserve.mlp import THREADS

        torch.set_num_threads(THREADS)
    run = {"xgb": run_xgb, "mlp": run_mlp}[family]
    LODO_DIR.mkdir(parents=True, exist_ok=True)
    parts = load_parts()
    for held_out in [*datasets(), CONTROL]:
        preds, info = run(parts, held_out)
        np.savez(path(family, held_out), **preds, **info)
        print(f"{family} without {held_out:13} best round {info['best_round']:4d}  "
              f"{info['train_rows']:,} train rows  {info['fit_seconds']:.0f} s")
    print(f"wrote {LODO_DIR.relative_to(ROOT)}/{family}_*.npz")


if __name__ == "__main__":
    main(sys.argv[1:])
