"""Day-3 baselines: the bar every model must beat, at matched coverage on validation.

Deployable baselines use only what a gateway knows at request time (the model and the prompt
length) and are fitted on train. Two reference rows are not deployable: per-model × source dataset
(the dataset is unknown at serving time) and the oracle (each prompt's own completions). The oracle
is the ceiling that a model's gain is read against, not a competitor.

Bounds are whole tokens (a gateway reserves integers), so every bound is rounded up.

Writes docs/figures/baselines_validate.json and docs/figures/coverage_curve.png. Run with
`make baselines`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.evaluate import SWEEP, TARGET, BoundFn, oracle_gap_closed, score, sweep, tune
from llm_reserve.explore import BLUE, INK_2, INK_3, ORANGE, SURFACE, add_titles

FIG_DIR = ROOT / "docs" / "figures"
SPLIT = "validate"
MAX_TOKENS = 4096
BAR = "per_model"  # the fixed per-model percentile: the bar in the success metrics
ORACLE = "oracle"


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    deployable: bool
    bounds: BoundFn
    # Range of the knob θ: a quantile in [0, 1], or the oracle's scale k. None means not tunable.
    knob: tuple[float, float] | None = (0.0, 1.0)


def group_key(df: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """One string per row naming its group; a single group when `keys` is empty."""
    if not keys:
        return np.full(len(df), "all", dtype=object)
    out = df[keys[0]].astype(str)
    for k in keys[1:]:
        out = out + "|" + df[k].astype(str)
    return out.to_numpy()


def group_quantile(train: pd.DataFrame, rows: pd.DataFrame, keys: Sequence[str]) -> BoundFn:
    """θ -> the θ-quantile of train lengths in each row's group. Groups unseen in train are an error."""
    lengths = pd.Series(train.output_tokens.to_numpy())
    by_group = dict(iter(lengths.groupby(pd.Series(group_key(train, keys)))))
    names = list(by_group)
    sorted_lengths = [np.sort(by_group[g].to_numpy()) for g in names]
    pos = pd.Index(names).get_indexer(group_key(rows, keys))
    if (pos < 0).any():
        missing = sorted(set(group_key(rows, keys)[pos < 0]))
        raise ValueError(f"groups not in train: {missing[:5]}")

    def bounds(q: float) -> np.ndarray:
        return np.ceil([np.quantile(a, q) for a in sorted_lengths])[pos]

    return bounds


def input_quintile_edges(train: pd.DataFrame) -> np.ndarray:
    """The 4 inner edges of input_size quintiles over train completions."""
    return np.quantile(train.input_size, [0.2, 0.4, 0.6, 0.8])


def with_input_bucket(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Quintile 0-4 by input_size; prompts outside the train range fall into the end buckets."""
    return df.assign(input_bucket=np.searchsorted(edges, df.input_size, side="right"))


def oracle_bounds(rows: pd.DataFrame) -> BoundFn:
    """k -> k × own mean + own std, over each (prompt, model) sample's completions in `rows`.

    It reads the lengths it bounds, so it is a ceiling, never a method. The std is the sample std
    (ddof=1); a sample with one completion has a std of 0.
    """
    g = rows.groupby("sample_id").output_tokens
    mean = g.transform("mean").to_numpy()
    std = g.transform("std").fillna(0).to_numpy()
    return lambda k: np.ceil(k * mean + std)


def make_methods(train: pd.DataFrame, rows: pd.DataFrame, prompts: pd.DataFrame) -> list[Method]:
    """Every baseline and reference row, fitted on `train`, producing bounds for `rows`."""
    datasets = prompts.set_index("prompt_id").dataset
    train = train.assign(dataset=train.prompt_id.map(datasets))
    rows = rows.assign(dataset=rows.prompt_id.map(datasets))
    edges = input_quintile_edges(train)
    train, rows = with_input_bucket(train, edges), with_input_bucket(rows, edges)
    constant = np.full(len(rows), float(MAX_TOKENS))
    return [
        Method("max_tokens", f"max_tokens = {MAX_TOKENS:,}", True, lambda _: constant, knob=None),
        Method("global", "Global percentile", True, group_quantile(train, rows, [])),
        Method(BAR, "Per-model percentile (the bar)", True, group_quantile(train, rows, ["model"])),
        Method("per_model_input", "Per-model × input-length quintile", True,
               group_quantile(train, rows, ["model", "input_bucket"])),
        Method("per_model_dataset", "Per-model × source dataset", False,
               group_quantile(train, rows, ["model", "dataset"])),
        Method(ORACLE, "Oracle: k × own mean + own std", False, oracle_bounds(rows), knob=(0.0, 10.0)),
    ]


def evaluate_methods(methods: list[Method], y: np.ndarray, target: float = TARGET,
                     targets: Sequence[float] = SWEEP) -> dict:
    """Tune each method to `target`, then sweep the curve. Gap closed is read against the bar and the oracle."""
    results = {}
    for m in methods:
        if m.knob is None:
            s, curve = score(m.bounds, y, 0.0), []
        else:
            s = score(m.bounds, y, tune(m.bounds, y, target, *m.knob))
            curve = [c.as_dict() for c in sweep(m.bounds, y, targets, *m.knob)]
        results[m.name] = {"label": m.label, "deployable": m.deployable, **s.as_dict(), "curve": curve}
    bar, oracle = results[BAR]["ratio"], results[ORACLE]["ratio"]
    for r in results.values():
        r["vs_bar"] = r["ratio"] / bar - 1
        r["gap_closed"] = oracle_gap_closed(bar, r["ratio"], oracle)
    return results


def markdown_table(results: dict) -> str:
    lines = [
        "| Method | Deployable | Knob θ | Coverage (sanitized) | Coverage (original) | Reservation ratio "
        "| vs. the bar | Oracle gap closed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results.values():
        theta = "—" if not r["curve"] else f"{r['theta']:.4f}"
        lines.append(
            f"| {r['label']} | {'yes' if r['deployable'] else 'no'} | {theta} | {r['coverage']:.2%} "
            f"| {r['coverage_original']:.2%} | {r['ratio']:.3f} | {r['vs_bar']:+.1%} | {r['gap_closed']:.1%} |"
        )
    return "\n".join(lines)


# Fixed categorical order (reference palette slots 1-5), so a method keeps its color in later plots.
COLORS = {"global": BLUE, BAR: ORANGE, "per_model_input": "#1baf7a", "per_model_dataset": "#eda100",
          ORACLE: "#e87ba4"}


def plot_curve(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, color in COLORS.items():
        r = results[name]
        cov = [c["coverage"] * 100 for c in r["curve"]]
        ratio = [c["ratio"] for c in r["curve"]]
        dashes = (0, (4, 2)) if not r["deployable"] else "-"
        ax.plot(cov, ratio, color=color, linewidth=2, linestyle=dashes, label=r["label"])
        ax.plot(r["coverage"] * 100, r["ratio"], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.axvline(TARGET * 100, color=INK_3, linewidth=1, linestyle=":")
    ax.text(TARGET * 100, ax.get_ylim()[1], " matched coverage, 95%", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlim(79.5, 99.5)
    ax.set_xlabel("coverage on validation (%): share of completions within their bound")
    ax.set_ylabel("reservation ratio: reserved ÷ used tokens")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    mt = results["max_tokens"]
    add_titles(fig, "Coverage vs. reservation, validation split (sanitized): lower is better",
               f"Each method tuned to 80–99% coverage; percentiles fitted on train. Dots: tuned to 95%. "
               f"Dashed: not deployable.\nNot shown: max_tokens = {MAX_TOKENS:,} "
               f"({mt['coverage']:.2%} coverage, ratio {mt['ratio']:.2f}).", top=0.87)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = load_completions("train", "sanitized")
    prompts = load_prompts()
    rows = load_completions(SPLIT, "sanitized")
    original = load_completions(SPLIT, "original")

    results = evaluate_methods(make_methods(train, rows, prompts), rows.output_tokens.to_numpy())
    # The same frozen knobs, scored on the original config, where the degenerate tail is.
    # The oracle recomputes its mean and std from the original completions.
    y_original = original.output_tokens.to_numpy()
    for m in make_methods(train, original, prompts):
        r = results[m.name]
        r["coverage_original"] = score(m.bounds, y_original, r["theta"]).coverage

    summary = {
        "split": SPLIT, "target": TARGET, "train_completions": len(train),
        "completions_sanitized": len(rows), "completions_original": len(original),
        "mean_length": float(rows.output_tokens.mean()),
        "input_quintile_edges": input_quintile_edges(train).tolist(),
        "methods": results,
    }
    (FIG_DIR / "baselines_validate.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_curve(results, FIG_DIR / "coverage_curve.png")
    print(markdown_table(results))
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
