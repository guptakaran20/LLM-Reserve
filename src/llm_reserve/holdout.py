"""Day 7: score the test split once, with every choice frozen, then analyse the errors.

Every knob was frozen on validation (Days 3, 5 and 6) and is read back here, never re-tuned:

- each baseline's percentile θ, fitted on train and tuned on validation (baselines_validate.json),
- each model's log-space shift (models_validate.json, mlp_validate.json, data/models/manifest.json),
- the winner: the equal-weight average of the MLP and XGBoost log predictions (Day 6).

The frozen protocol is the headline: each method keeps its validation knob, so its test coverage lands
near 95% but not on it. Two diagnostics sit next to it and choose nothing: the same comparison with
every knob re-tuned to exactly 95% on test ("matched on test"), so a gain can't come from covering
less, and a prompt-level bootstrap of the frozen numbers. The oracle (k × own mean + own std) reads
the lengths it bounds, so its k is tuned on test, as it was on validation: it is the ceiling, not a
method.

Then the PRD's breakdowns and checks, all at the frozen knobs:

- coverage and reservation ratio per target model, source dataset and input-length quintile;
- the 100 worst under-predictions (by (y - bound) / bound, one per sample) put into causes by a fixed
  rule list (CAUSES), and where the original config's misses and overshoot come from;
- leave-one-dataset-out (llm_reserve.lodo): for each source dataset, the winner retrained without it,
  its shift tuned on validation outside it, scored on its test prompts.

Reads the predictions written by llm_reserve.predict and llm_reserve.lodo: this process loads neither
torch nor runs xgboost. Writes docs/figures/test_results.json, test_curve.png, test_by_model.png and
lodo.png. Run with `make eval`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_reserve import lodo, predict
from llm_reserve.baselines import (
    BAR, MAX_TOKENS, ORACLE, group_quantile, make_methods, oracle_bounds, with_input_bucket,
)
from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.evaluate import SWEEP, TARGET, BoundFn, coverage, oracle_gap_closed, reservation_ratio, score, sweep, tune
from llm_reserve.explore import BLUE, GRID, INK_2, INK_3, ORANGE, SURFACE, add_titles, short_model
from llm_reserve.features import LENGTH_HINTS, SERVING_LENGTH, load_features
from llm_reserve.models import SHIFT, shifted_bounds

FIG_DIR = ROOT / "docs" / "figures"
MODELS_DIR = ROOT / "data" / "models"
SPLIT = "test"
SERVING = f"xgb_minilm{SERVING_LENGTH}"  # the Day-5 XGBoost, one half of the winner
MLP = "mlp"
WINNER = "mlp_xgb_avg"  # Day 6's pick: the average of the MLP and XGBoost log predictions
MODEL_ROWS = ("linear", "xgb_hand", SERVING, MLP, WINNER)
BREAKDOWN = (BAR, SERVING, MLP, WINNER)  # methods broken down per model, dataset and input quintile
GOAL = -0.20  # the PRD target: at least 20% fewer reserved tokens than the bar, at the same coverage
N_WORST = 100
N_BOOT = 2000
SEED = 0

# Rules for the worst under-predictions, applied in this order: the first that matches is the cause.
CAUSES = {
    "runaway": "Runaway: the completion is past max_tokens = 4,096, and the sanitized config kept it",
    "lone_long_run": "Lone long run: the sample's median completion is within the bound",
    "prompt_misread": "Prompt misread: the median is over the bound, and so is it for at least half "
                      "of the other LLMs on this prompt",
    "llm_specific": "This LLM, not the prompt: the median is over the bound, but most other LLMs stayed "
                    "within theirs",
}
# Day-7 figures: the winner takes slot 1 (blue). Day 6 drew it green, but green beside the bar's orange
# fails the protanopia check (ΔE 3.2); blue, orange and aqua pass every pair. References are gray.
WINNER_COLOR, AQUA, HELD_OUT = BLUE, "#1baf7a", "#e34948"


@dataclass(frozen=True)
class Frozen:
    """One method as it was frozen on validation."""
    name: str
    label: str
    deployable: bool
    theta: float | None  # the frozen knob; None for max_tokens (no knob) and the oracle (tuned on test)
    curve: tuple[float, ...]  # the validation-tuned knob for each SWEEP target
    knob: tuple[float, float] | None  # the knob's range, for the matched-on-test diagnostic


def frozen_methods() -> list[Frozen]:
    """Every baseline and model with the knobs it was frozen with, checked against the model manifest."""
    base = json.loads((FIG_DIR / "baselines_validate.json").read_text())["methods"]
    day5 = json.loads((FIG_DIR / "models_validate.json").read_text())["methods"]
    day6 = json.loads((FIG_DIR / "mlp_validate.json").read_text())
    manifest = predict.manifest()
    if manifest["winner"] != WINNER:
        raise AssertionError(f"Day 6 froze {manifest['winner']}, not {WINNER}")
    out = []
    for name, r in base.items():
        knob = None if name == "max_tokens" else (0.0, 10.0) if name == ORACLE else (0.0, 1.0)
        tuned = name not in ("max_tokens", ORACLE)
        out.append(Frozen(name, r["label"], r["deployable"], r["theta"] if tuned else None,
                          tuple(c["theta"] for c in r["curve"]) if tuned else (), knob))
    rows = {**{n: day5[n] for n in MODEL_ROWS[:3]}, MLP: day6["methods"][day6["chosen"]],
            WINNER: day6["methods"][WINNER]}
    labels = {SERVING: "XGBoost (hand-built + MiniLM-128 PCA-32)", MLP: "MLP (hand-built + MiniLM-128 PCA-32)",
              WINNER: "Average of the MLP and XGBoost (the winner)"}
    for name, r in rows.items():
        if manifest["models"][name]["shift"] != r["theta"]:
            raise AssertionError(f"{name}: the manifest's shift is not the one validation chose")
        out.append(Frozen(name, labels.get(name, r["label"]), True, r["theta"], tuple(c["theta"] for c in r["curve"]),
                          SHIFT))
    return out


def winner_log_pred(split: str) -> np.ndarray:
    """The winner's log predictions, averaged in float32 exactly as Day 6 did."""
    return (predict.load(MLP, split) + predict.load(SERVING, split)) / 2


def bound_functions(train: pd.DataFrame, rows: pd.DataFrame, prompts: pd.DataFrame,
                    log_preds: dict[str, np.ndarray]) -> dict[str, BoundFn]:
    """θ -> bounds on `rows`, for every baseline (fitted on train) and every model (from its log predictions)."""
    fns = {m.name: m.bounds for m in make_methods(train, rows, prompts)}
    return fns | {name: shifted_bounds(p) for name, p in log_preds.items()}


def score_frozen(methods: Sequence[Frozen], sanitized: dict[str, BoundFn], original: dict[str, BoundFn],
                 y: np.ndarray, y_original: np.ndarray, targets: Sequence[float] = SWEEP) -> dict:
    """Each method at its frozen knob on test (both configs), its frozen curve, and the matched-on-test diagnostic.

    The curve applies each validation-tuned knob to test, so its points sit near, not on, the targets.
    `vs_bar` and `gap_closed` compare frozen numbers; `matched` re-tunes every knob to `TARGET` on test.
    """
    results = {}
    for m in methods:
        fs, fo = sanitized[m.name], original[m.name]
        if m.name == ORACLE:  # the ceiling reads the lengths it bounds: k is tuned on the split it scores
            theta = tune(fs, y, TARGET, *m.knob)
            curve = [{"target": t, **c.as_dict()} for t, c in zip(targets, sweep(fs, y, targets, *m.knob))]
        elif m.theta is None:
            theta, curve = 0.0, []
        else:
            theta = m.theta
            curve = [{"target": t, **score(fs, y, th).as_dict()} for t, th in zip(targets, m.curve)]
        s, o = score(fs, y, theta), score(fo, y_original, theta)
        r = {"label": m.label, "deployable": m.deployable, "theta": theta, "coverage": s.coverage, "ratio": s.ratio,
             "coverage_original": o.coverage, "ratio_original": o.ratio, "curve": curve}
        if m.knob is not None:
            r["matched"] = score(fs, y, tune(fs, y, TARGET, *m.knob)).as_dict()
        results[m.name] = r
    bar, oracle = results[BAR], results[ORACLE]
    for r in results.values():
        r["vs_bar"] = r["ratio"] / bar["ratio"] - 1
        r["gap_closed"] = oracle_gap_closed(bar["ratio"], r["ratio"], oracle["ratio"])
        if "matched" in r:
            r["matched"]["vs_bar"] = r["matched"]["ratio"] / bar["matched"]["ratio"] - 1
            r["matched"]["gap_closed"] = oracle_gap_closed(bar["matched"]["ratio"], r["matched"]["ratio"],
                                                           oracle["matched"]["ratio"])
    return results


# ---- Breakdowns and the bootstrap -----------------------------------------------------------------

def breakdown(y: np.ndarray, bound: np.ndarray, groups: np.ndarray, order: Sequence | None = None) -> dict:
    """Rows, coverage and reservation ratio per group."""
    df = pd.DataFrame({"y": y, "b": bound, "covered": y <= bound, "g": groups})
    agg = df.groupby("g", sort=True).agg(rows=("y", "size"), covered=("covered", "mean"), y=("y", "sum"),
                                         b=("b", "sum"))
    keys = list(agg.index) if order is None else [k for k in order if k in agg.index]
    return {str(k): {"rows": int(agg.at[k, "rows"]), "coverage": float(agg.at[k, "covered"]),
                     "ratio": float(agg.at[k, "b"] / agg.at[k, "y"])} for k in keys}


def bucket_labels(edges: Sequence[float]) -> list[str]:
    """Labels of the input-length quintiles that `with_input_bucket` assigns (bucket i: edges[i-1] <= x < edges[i])."""
    e = [f"{x:,.0f}" for x in edges]
    return [f"< {e[0]}", *(f"{a}–{b}" for a, b in zip(e, e[1:])), f"≥ {e[-1]}"]


def interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(values, a)), float(np.quantile(values, 1 - a))]


def prompt_bootstrap(prompt_ids: np.ndarray, y: np.ndarray, bounds: dict[str, np.ndarray], reference: str,
                     n: int = N_BOOT, seed: int = SEED, goal: float = GOAL) -> dict:
    """95% intervals for coverage, ratio and the change vs. `reference`, resampling whole prompts.

    Every completion of a prompt, across all target models, is drawn together: they share the prompt's
    text, so resampling rows would understate the noise. Knobs stay frozen.
    """
    codes, uniques = pd.factorize(prompt_ids)
    p = len(uniques)
    w = np.random.default_rng(seed).multinomial(p, np.full(p, 1 / p), size=n).astype(float)
    used = w @ np.bincount(codes, weights=y, minlength=p)
    rows = w @ np.bincount(codes, minlength=p)
    ratios, out = {}, {}
    for name, b in bounds.items():
        ratios[name] = (w @ np.bincount(codes, weights=b, minlength=p)) / used
        cov = (w @ np.bincount(codes, weights=(y <= b).astype(float), minlength=p)) / rows
        out[name] = {"coverage": interval(cov), "ratio": interval(ratios[name])}
    for name in bounds:
        change = ratios[name] / ratios[reference] - 1
        out[name]["vs_reference"] = interval(change)
        out[name]["share_reaching_goal"] = float(np.mean(change <= goal))
    return {"resamples": n, "prompts": p, "reference": reference, "methods": out}


# ---- The worst under-predictions ------------------------------------------------------------------

def sample_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (prompt, model) sample: its bound, completions and worst relative miss.

    `rows` holds one completion per row with sample_id, prompt_id, model, dataset, has_hint, output_tokens
    and bound. The bound is constant within a sample (features are), so the sample's median can be read
    against it. `others_over` is the share of the prompt's other LLMs whose median is over their own
    bound (NaN when no other LLM answered the prompt).
    """
    rows = rows.assign(over=rows.output_tokens > rows.bound)
    g = rows.groupby("sample_id", sort=False)
    s = g.agg(prompt_id=("prompt_id", "first"), model=("model", "first"), dataset=("dataset", "first"),
              has_hint=("has_hint", "first"), bound=("bound", "first"), bound_max=("bound", "max"),
              worst=("output_tokens", "max"), median=("output_tokens", "median"), mean=("output_tokens", "mean"),
              std=("output_tokens", "std"), n=("output_tokens", "size"), n_over=("over", "sum"))
    if (s.bound != s.bound_max).any():
        raise ValueError("a sample's bound varies across its completions")
    s = s.drop(columns="bound_max")
    s["miss"] = (s.worst - s.bound) / s.bound
    s["cv"] = (s["std"] / s["mean"]).fillna(0.0)
    s["median_over"] = s["median"] > s.bound
    per_prompt = s.groupby("prompt_id").median_over.agg(["sum", "size"])
    total, count = s.prompt_id.map(per_prompt["sum"]), s.prompt_id.map(per_prompt["size"])
    s["others_over"] = ((total - s.median_over) / (count - 1)).where(count > 1)
    return s


def cause(samples: pd.DataFrame) -> pd.Series:
    """The first rule in CAUSES that each sample matches."""
    rules = [samples.worst > MAX_TOKENS, ~samples.median_over, samples.others_over.fillna(0) >= 0.5]
    return pd.Series(np.select(rules, list(CAUSES)[:3], default="llm_specific"), index=samples.index)


def worst_misses(samples: pd.DataFrame, n: int = N_WORST) -> pd.DataFrame:
    """The n samples with the largest relative under-prediction of their worst completion, with causes."""
    top = samples[samples.miss > 0].nlargest(n, "miss")
    return top.assign(cause=cause(top))


def error_summary(top: pd.DataFrame, samples: pd.DataFrame) -> dict:
    """Counts per cause, and where the worst misses sit against the share of all samples."""
    def shares(col: str) -> dict:
        base = samples[col].astype(str).value_counts(normalize=True)
        worst = top[col].astype(str).value_counts()
        return {k: {"worst": int(worst.get(k, 0)), "share_of_samples": float(base[k])}
                for k in worst.index.union(base.index)}

    causes = {}
    for key, text in CAUSES.items():
        t = top[top.cause == key]
        causes[key] = {"label": text, "count": len(t), "prompts": int(t.prompt_id.nunique()),
                       "median_miss": float(t.miss.median()) if len(t) else None,
                       "median_cv": float(t.cv.median()) if len(t) else None,
                       "with_length_hint": int(t.has_hint.sum()),
                       "whole_sample_over_max_tokens": int((t["median"] > MAX_TOKENS).sum()),
                       "datasets": t.dataset.astype(str).value_counts().to_dict(),
                       "models": t.model.astype(str).map(short_model).value_counts().to_dict()}
    return {"n": len(top), "prompts": int(top.prompt_id.nunique()), "causes": causes,
            "median_cv_worst": float(top.cv.median()), "median_cv_all": float(samples.cv.median()),
            "worst_miss": float(top.miss.max()), "smallest_miss_in_top": float(top.miss.min()),
            "by_dataset": shares("dataset"),
            "by_model": {short_model(k): v for k, v in shares("model").items()}}


def tail_accounting(y: np.ndarray, bound: np.ndarray, degenerate: np.ndarray, max_tokens: int = MAX_TOKENS) -> dict:
    """Where the original config's misses and overshoot come from, raw and as a gateway sees them.

    A gateway reserves min(bound, max_tokens) and is billed min(y, max_tokens), so overshoot past a
    reservation is clipped at max_tokens too.
    """
    over = np.maximum(y - bound, 0.0)
    clipped = np.maximum(np.minimum(y, max_tokens) - np.minimum(bound, max_tokens), 0.0)
    missed = y > bound
    return {"completions": len(y), "missed": int(missed.sum()), "missed_degenerate": int((missed & degenerate).sum()),
            "over_tokens": float(over.sum()), "over_share_degenerate": float(over[degenerate].sum() / over.sum()),
            "over_per_used": float(over.sum() / y.sum()),
            "over_tokens_clipped": float(clipped.sum()),
            "clipped_share_degenerate": float(clipped[degenerate].sum() / clipped.sum()),
            "clipped_over_per_used": float(clipped.sum() / np.minimum(y, max_tokens).sum())}


# ---- Leave one dataset out ------------------------------------------------------------------------

def lodo_scores(datasets: Sequence[str], train: pd.DataFrame, rows_va: pd.DataFrame, y_va: np.ndarray,
                rows_te: pd.DataFrame, y_te: np.ndarray, held_out: dict, full: dict[str, tuple[np.ndarray, float]],
                bar_theta: float) -> dict:
    """Per held-out dataset D, and pooled: the winner trained with D vs. without it, on D's test rows.

    Rows are sanitized and carry `dataset`; train does too. `held_out[D][family]` holds the LODO log
    predictions {"validate", "test"} row-aligned with rows_va and rows_te; `full[family]` the frozen
    model's test log predictions and shift. "Frozen" uses the knob tuned on validation (outside D for the
    LODO model and its bar); "matched" re-tunes every method to 95% on D's test rows, to compare how well
    they rank D's prompts at equal coverage. The LODO winner is read against a bar fitted without D.
    """
    ds_va, ds_te = rows_va.dataset.to_numpy(), rows_te.dataset.to_numpy()
    per, pooled = {}, {"lodo": [], "bar": [], "y": []}
    for d in datasets:
        va, te = ds_va != d, ds_te == d
        yv, yt = y_va[va], y_te[te]
        r = {"test_prompts": int(rows_te[te].prompt_id.nunique()), "test_rows": int(te.sum())}

        def matched(fn: BoundFn, knob: tuple[float, float]) -> float:
            return score(fn, yt, tune(fn, yt, TARGET, *knob)).ratio

        for family, preds in held_out[d].items():
            pv, pt = preds["validate"][va], preds["test"][te]
            shift = tune(shifted_bounds(pv), yv, TARGET, *SHIFT)
            fn = shifted_bounds(pt)
            r[f"lodo_{family}"] = {"shift": shift, "coverage": coverage(yt, fn(shift)),
                                   "ratio": reservation_ratio(yt, fn(shift)), "matched_ratio": matched(fn, SHIFT)}
            if family == "winner":
                pooled["lodo"].append((pt, shift))
        for family, (pred, shift) in full.items():
            fn = shifted_bounds(pred[te])
            r[f"full_{family}"] = {"shift": shift, "coverage": coverage(yt, fn(shift)),
                                   "ratio": reservation_ratio(yt, fn(shift)), "matched_ratio": matched(fn, SHIFT)}
        train_without = train[train.dataset != d]
        bar_full = group_quantile(train, rows_te[te], ["model"])
        theta = tune(group_quantile(train_without, rows_va[va], ["model"]), yv, TARGET)
        bar_without = group_quantile(train_without, rows_te[te], ["model"])
        pooled["bar"].append((bar_without, theta))
        pooled["y"].append(yt)
        r["bar_full"] = {"theta": bar_theta, "coverage": coverage(yt, bar_full(bar_theta)),
                         "ratio": reservation_ratio(yt, bar_full(bar_theta)), "matched_ratio": matched(bar_full, (0, 1))}
        r["bar_without"] = {"theta": theta, "coverage": coverage(yt, bar_without(theta)),
                            "ratio": reservation_ratio(yt, bar_without(theta)),
                            "matched_ratio": matched(bar_without, (0, 1))}
        r["oracle_ratio"] = matched(oracle_bounds(rows_te[te]), (0.0, 10.0))
        r["gap_closed_full"] = oracle_gap_closed(r["bar_full"]["matched_ratio"], r["full_winner"]["matched_ratio"],
                                                 r["oracle_ratio"])
        r["gap_closed_lodo"] = oracle_gap_closed(r["bar_without"]["matched_ratio"],
                                                 r["lodo_winner"]["matched_ratio"], r["oracle_ratio"])
        r["vs_bar_full"] = r["full_winner"]["matched_ratio"] / r["bar_full"]["matched_ratio"] - 1
        r["vs_bar_lodo"] = r["lodo_winner"]["matched_ratio"] / r["bar_without"]["matched_ratio"] - 1
        per[d] = r

    # Pooled: every test row scored by the model that never saw its dataset. Frozen keeps each D's knob;
    # matched adds one common offset (a shift for the model, a percentile for the bar) to reach 95% overall.
    y_all = np.concatenate(pooled["y"])
    lodo_fn = lambda delta: np.concatenate([shifted_bounds(p)(s + delta) for p, s in pooled["lodo"]])  # noqa: E731
    bar_frozen = np.concatenate([fn(t) for fn, t in pooled["bar"]])
    bar_fn = lambda q: np.concatenate([fn(q) for fn, _ in pooled["bar"]])  # noqa: E731
    lodo_m = score(lodo_fn, y_all, tune(lodo_fn, y_all, TARGET, -2.0, 2.0))
    bar_m = score(bar_fn, y_all, tune(bar_fn, y_all, TARGET))
    summary = {"coverage": coverage(y_all, lodo_fn(0.0)), "ratio": reservation_ratio(y_all, lodo_fn(0.0)),
               "bar_coverage": coverage(y_all, bar_frozen), "bar_ratio": reservation_ratio(y_all, bar_frozen),
               "matched_ratio": lodo_m.ratio, "bar_matched_ratio": bar_m.ratio}
    summary["vs_bar"] = summary["ratio"] / summary["bar_ratio"] - 1
    summary["matched_vs_bar"] = lodo_m.ratio / bar_m.ratio - 1
    return {"datasets": per, "pooled": summary}


def control_score(pred_va: np.ndarray, y_va: np.ndarray, pred_te: np.ndarray, y_te: np.ndarray,
                  bar_matched: float, oracle: float) -> dict:
    """The control run (1/7 of train prompts dropped at random) on all test rows: its shift tuned on all of validation."""
    shift = tune(shifted_bounds(pred_va), y_va, TARGET, *SHIFT)
    fn = shifted_bounds(pred_te)
    m = score(fn, y_te, tune(fn, y_te, TARGET, *SHIFT))
    return {"shift": shift, "coverage": coverage(y_te, fn(shift)), "ratio": reservation_ratio(y_te, fn(shift)),
            "matched_ratio": m.ratio, "matched_vs_bar": m.ratio / bar_matched - 1,
            "gap_closed": oracle_gap_closed(bar_matched, m.ratio, oracle)}


# ---- Figures --------------------------------------------------------------------------------------

def plot_curve(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    series = [(WINNER, WINNER_COLOR, "-"), (BAR, ORANGE, "-"), ("per_model_input", AQUA, "-"),
              ("per_model_dataset", INK_3, ":"), (ORACLE, INK_3, (0, (4, 2)))]
    for name, color, style in series:
        r = results[name]
        ax.plot([c["coverage"] * 100 for c in r["curve"]], [c["ratio"] for c in r["curve"]], color=color,
                linewidth=2, linestyle=style, label=r["label"])
        ax.plot(r["coverage"] * 100, r["ratio"], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.axvline(TARGET * 100, color=INK_3, linewidth=1, linestyle=":")
    ax.text(TARGET * 100, ax.get_ylim()[1], " 95%", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlim(79.5, 99.8)
    ax.set_xlabel("coverage on test (%): share of completions within their bound")
    ax.set_ylabel("reservation ratio: reserved ÷ used tokens")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    note = f"XGBoost {results[SERVING]['ratio']:.3f} and the MLP {results[MLP]['ratio']:.3f}"
    add_titles(fig, "Coverage vs. reservation on the test split (sanitized), scored once: lower is better",
               "Every knob frozen on validation for 80–99% coverage; x is the coverage it reached on test. "
               "Dots: the 95% knob.\nGray: not deployable; the oracle is tuned on test itself, as the ceiling.\n"
               f"Not shown: {note} at their 95% knobs.", top=0.83)
    fig.savefig(path)
    plt.close(fig)


def plot_by_model(by_model: dict, path) -> None:
    """Coverage and reservation ratio per target model, the bar vs. the winner, at their frozen knobs."""
    bar, win = by_model[BAR], by_model[WINNER]
    models = sorted(bar, key=lambda m: bar[m]["ratio"])
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
    for ax, key, scale in ((axes[0], "coverage", 100), (axes[1], "ratio", 1)):
        b = [bar[m][key] * scale for m in models]
        w = [win[m][key] * scale for m in models]
        ax.hlines(y, np.minimum(b, w), np.maximum(b, w), color=GRID, linewidth=2)
        ax.plot(b, y, "o", color=ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label="Per-model percentile (the bar)")
        ax.plot(w, y, "D", color=WINNER_COLOR, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                label="Average of the MLP and XGBoost (the winner)")
        ax.grid(axis="y", visible=False)
    axes[0].axvline(TARGET * 100, color=INK_3, linewidth=1, linestyle=":")
    axes[0].set_xlabel("coverage on test (%)")
    axes[1].set_xlabel("reservation ratio: reserved ÷ used tokens")
    axes[0].set_yticks(y, models)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    add_titles(fig, "Per target model on the test split (sanitized), at the knobs frozen on validation",
               "Models sorted by the bar's reservation ratio. Dotted: 95% coverage.", top=0.9)
    fig.savefig(path)
    plt.close(fig)


def plot_lodo(result: dict, path) -> None:
    per = result["datasets"]
    names = sorted(per, key=lambda d: per[d]["gap_closed_full"])
    y = np.arange(len(names))
    h = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    ax = axes[0]
    for offset, key, color, label in ((h / 2, "gap_closed_full", WINNER_COLOR, "Trained with the dataset (frozen winner)"),
                                      (-h / 2, "gap_closed_lodo", HELD_OUT, "Dataset held out of training")):
        vals = [per[d][key] * 100 for d in names]
        ax.barh(y + offset, vals, height=h, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for yi, v in zip(y + offset, vals):
            ax.text(v + (1 if v >= 0 else -1), yi, f"{v:.0f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_3, linewidth=1)
    low = min(0.0, min(per[d][k] for d in names for k in ("gap_closed_full", "gap_closed_lodo"))) * 100
    ax.set_xlim(low - 14, 92)
    ax.set_xlabel("oracle gap closed on its test prompts (%),\nevery method tuned to 95% there")
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax = axes[1]
    for key, color, marker in (("full_winner", WINNER_COLOR, "D"), ("lodo_winner", HELD_OUT, "o")):
        ax.plot([per[d][key]["coverage"] * 100 for d in names], y, marker, color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
    ax.axvline(TARGET * 100, color=INK_3, linewidth=1, linestyle=":")
    ax.set_xlabel("coverage on its test prompts (%),\nshift frozen on validation. Dotted: 95%")
    ax.grid(axis="y", visible=False)
    add_titles(fig, "Leave one source dataset out: does the gain survive a benchmark the model never saw?",
               "For each dataset, the winner retrained without it (PCA, training and early stopping), its shift "
               "tuned on validation outside it.\nLeft: ranking quality at equal coverage. Right: whether the "
               "frozen 95% holds.", top=0.86)
    fig.savefig(path)
    plt.close(fig)


# ---- Tables ---------------------------------------------------------------------------------------

def markdown_results(results: dict) -> str:
    lines = ["| Method | Deployable | Frozen knob θ | Coverage (sanitized) | Coverage (original) | Reservation ratio "
             "| vs. the bar | Oracle gap closed | Ratio matched on test | vs. the bar, matched |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        theta = "—" if name == "max_tokens" else ("tuned on test" if name == ORACLE else
                                                  f"{r['theta']:+.3f}" if name in MODEL_ROWS else f"{r['theta']:.4f}")
        m = r.get("matched")
        gap = "—" if name == "max_tokens" else f"{r['gap_closed']:.1%}"
        lines.append(f"| {r['label']} | {'yes' if r['deployable'] else 'no'} | {theta} | {r['coverage']:.2%} "
                     f"| {r['coverage_original']:.2%} | {r['ratio']:.3f} | {r['vs_bar']:+.1%} | {gap} "
                     f"| {'—' if m is None else format(m['ratio'], '.3f')} "
                     f"| {'—' if m is None else format(m['vs_bar'], '+.1%')} |")
    return "\n".join(lines)


def markdown_breakdown(tables: dict, labels: dict) -> str:
    groups = list(next(iter(tables.values())))
    head = " | ".join(f"{labels[n]} coverage | {labels[n]} ratio" for n in tables)
    lines = [f"| Group | Rows | {head} | Change vs. the bar |", "|---" * (3 + 2 * len(tables)) + "|"]
    for g in groups:
        cells = " | ".join(f"{t[g]['coverage']:.2%} | {t[g]['ratio']:.3f}" for t in tables.values())
        change = tables[WINNER][g]["ratio"] / tables[BAR][g]["ratio"] - 1
        lines.append(f"| {g} | {tables[BAR][g]['rows']:,} | {cells} | {change:+.1%} |")
    return "\n".join(lines)


def markdown_lodo(result: dict) -> str:
    lines = ["| Held-out dataset | Test prompts | Coverage, frozen: bar / full / held out | Ratio at 95% on the "
             "dataset: bar / full / held out / oracle | vs. the bar: full / held out | Gap closed: full / held out |",
             "|---|---|---|---|---|---|"]
    for d, r in sorted(result["datasets"].items()):
        lines.append(
            f"| {d} | {r['test_prompts']} | {r['bar_without']['coverage']:.1%} / {r['full_winner']['coverage']:.1%} / "
            f"{r['lodo_winner']['coverage']:.1%} | {r['bar_without']['matched_ratio']:.3f} / "
            f"{r['full_winner']['matched_ratio']:.3f} / {r['lodo_winner']['matched_ratio']:.3f} / {r['oracle_ratio']:.3f} "
            f"| {r['vs_bar_full']:+.1%} / {r['vs_bar_lodo']:+.1%} | {r['gap_closed_full']:.0%} / {r['gap_closed_lodo']:.0%} |")
    return "\n".join(lines)


def fingerprints(files: Sequence[str]) -> dict[str, str]:
    """sha256 of the frozen artifacts, so later stages can check they serve what test scored."""
    return {f: hashlib.sha256((MODELS_DIR / f).read_bytes()).hexdigest() for f in files}


def main() -> None:
    methods = frozen_methods()
    manifest = predict.manifest()
    base = json.loads((FIG_DIR / "baselines_validate.json").read_text())
    day6 = json.loads((FIG_DIR / "mlp_validate.json").read_text())
    prompts = load_prompts()
    dataset = prompts.set_index("prompt_id").dataset
    train = load_completions("train", "sanitized")
    train = train.assign(dataset=train.prompt_id.map(dataset))

    # The winner, rebuilt from its saved halves, must reproduce its Day-6 validation ratio before test is read.
    _, y_va_o, meta_va = load_features("validate", "original")
    deg_va = meta_va.is_degenerate.to_numpy()
    val_ratio = score(shifted_bounds(winner_log_pred("validate")[~deg_va]), y_va_o[~deg_va],
                      manifest["models"][WINNER]["shift"]).ratio
    if abs(val_ratio - day6["methods"][WINNER]["ratio"]) > predict.TOL:
        raise AssertionError(f"the winner gives validation ratio {val_ratio:.6f}, Day 6 froze "
                             f"{day6['methods'][WINNER]['ratio']:.6f}")

    rows_o = load_completions(SPLIT, "original")
    X_o, y_o, meta_o = load_features(SPLIT, "original")
    if not (rows_o.sample_id.equals(meta_o.sample_id) and rows_o.completion_idx.equals(meta_o.completion_idx)):
        raise AssertionError("completions and features are not row-aligned")
    deg = rows_o.is_degenerate.to_numpy()
    rows_o = rows_o.assign(dataset=rows_o.prompt_id.map(dataset).to_numpy(),
                           has_hint=(X_o[list(LENGTH_HINTS)] > 0).any(axis=1).to_numpy())
    rows_s, y_s = rows_o[~deg].reset_index(drop=True), y_o[~deg]
    preds_o = {name: predict.load(name, SPLIT) for name in MODEL_ROWS[:4]} | {WINNER: winner_log_pred(SPLIT)}
    fns_s = bound_functions(train, rows_s, prompts, {k: v[~deg] for k, v in preds_o.items()})
    fns_o = bound_functions(train, rows_o, prompts, preds_o)
    results = score_frozen(methods, fns_s, fns_o, y_s, y_o)
    frozen = {m.name: m for m in methods}
    bounds_s = {n: fns_s[n](results[n]["theta"]) for n in results}

    boot = prompt_bootstrap(rows_s.prompt_id.to_numpy(), y_s, {n: bounds_s[n] for n in (BAR, *MODEL_ROWS[1:])}, BAR)

    edges = base["input_quintile_edges"]
    buckets = with_input_bucket(rows_s, np.asarray(edges)).input_bucket.to_numpy()
    labels = bucket_labels(edges)
    groups = {"by_model": rows_s.model.astype(str).map(short_model).to_numpy(),
              "by_dataset": rows_s.dataset.to_numpy(), "by_input_size": np.asarray(labels, dtype=object)[buckets]}
    by = {key: {n: breakdown(y_s, bounds_s[n], g, labels if key == "by_input_size" else None) for n in BREAKDOWN}
          for key, g in groups.items()}

    samples = sample_table(rows_s.assign(bound=bounds_s[WINNER]))
    top = worst_misses(samples)
    errors = error_summary(top, samples)
    worst = [{"sample_id": sid, "model": short_model(str(r.model)), "dataset": r.dataset, "bound": float(r.bound),
              "worst": int(r.worst), "median": float(r["median"]), "completions": int(r.n), "over": int(r.n_over),
              "miss": float(r.miss), "cv": float(r.cv), "others_over": None if pd.isna(r.others_over)
              else float(r.others_over), "cause": r.cause} for sid, r in top.iterrows()]
    tail = {n: tail_accounting(y_o, fns_o[n](results[n]["theta"]), deg) for n in (BAR, WINNER)}

    # Leave one dataset out, scored on sanitized rows.
    _, y_va, meta_va_s = load_features("validate", "sanitized")
    rows_va = load_completions("validate", "sanitized").assign(dataset=lambda d: d.prompt_id.map(dataset).to_numpy())
    if not rows_va.sample_id.equals(meta_va_s.sample_id):
        raise AssertionError("validation completions and features are not row-aligned")
    names = lodo.datasets()
    held_out = {}
    for d in names:
        runs = {f: lodo.load(f, d) for f in lodo.FAMILIES}
        fam = {"xgb": runs["xgb"], "mlp": runs["mlp"],
               "winner": {s: (runs["mlp"][s] + runs["xgb"][s]) / 2 for s in ("validate", "test")}}
        held_out[d] = {f: {"validate": p["validate"][~deg_va], "test": p["test"][~deg]} for f, p in fam.items()}
    full = {"winner": (preds_o[WINNER][~deg], frozen[WINNER].theta), "xgb": (preds_o[SERVING][~deg], frozen[SERVING].theta),
            "mlp": (preds_o[MLP][~deg], frozen[MLP].theta)}
    loo = lodo_scores(names, train, rows_va, y_va, rows_s, y_s, held_out, full, frozen[BAR].theta)
    loo["pooled"]["oracle_ratio"] = results[ORACLE]["ratio"]
    loo["pooled"]["gap_closed"] = oracle_gap_closed(loo["pooled"]["bar_matched_ratio"], loo["pooled"]["matched_ratio"],
                                                    results[ORACLE]["ratio"])
    ctrl = {f: lodo.load(f, lodo.CONTROL) for f in lodo.FAMILIES}
    ctrl["winner"] = {s: (ctrl["mlp"][s] + ctrl["xgb"][s]) / 2 for s in ("validate", "test")}
    loo["control"] = {f: control_score(ctrl[f]["validate"][~deg_va], y_va, ctrl[f]["test"][~deg], y_s,
                                       results[BAR]["matched"]["ratio"], results[ORACLE]["ratio"])
                      for f in ("winner", "xgb", "mlp")}
    loo["control"]["full_matched_ratio"] = results[WINNER]["matched"]["ratio"]
    loo["runs"] = {d: {f: {k: float(v) for k, v in lodo.load(f, d).items() if v.ndim == 0} for f in lodo.FAMILIES}
                   for d in [*names, lodo.CONTROL]}

    w = results[WINNER]
    goals = {"coverage": w["coverage"], "coverage_within_1pt": abs(w["coverage"] - TARGET) <= 0.01,
             "vs_bar": w["vs_bar"], "vs_bar_matched": w["matched"]["vs_bar"],
             "reaches_20pct": w["vs_bar"] <= GOAL and w["matched"]["vs_bar"] <= GOAL,
             "gap_closed": w["gap_closed"], "gap_closed_matched": w["matched"]["gap_closed"],
             "vs_max_tokens": w["ratio"] / results["max_tokens"]["ratio"]}
    files = [manifest["models"][n]["file"] for n in (SERVING, MLP)] + [manifest["models"][SERVING]["pca_file"]]
    summary = {
        "split": SPLIT, "target": TARGET, "goal_vs_bar": GOAL,
        "frozen": {"winner": WINNER, "members": manifest["models"][WINNER]["members"],
                   "shifts": {n: manifest["models"][n]["shift"] for n in (SERVING, MLP, WINNER)},
                   "winner_validate_ratio": val_ratio, "sha256": fingerprints(files)},
        "prompts": int(rows_o.prompt_id.nunique()), "completions_sanitized": len(rows_s),
        "completions_original": len(rows_o), "mean_length": float(y_s.mean()),
        "goals": goals, "methods": results, "bootstrap": boot,
        "breakdown": {"input_quintile_edges": edges, **by},
        "errors": {**errors, "worst": worst}, "tail_original": tail, "lodo": loo,
    }
    (FIG_DIR / "test_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_curve(results, FIG_DIR / "test_curve.png")
    plot_by_model(by["by_model"], FIG_DIR / "test_by_model.png")
    plot_lodo(loo, FIG_DIR / "lodo.png")

    short = {BAR: "Bar", SERVING: "XGBoost", MLP: "MLP", WINNER: "Winner"}
    print(markdown_results(results))
    print(json.dumps(goals, indent=2))
    print(json.dumps({n: boot["methods"][n] for n in (BAR, WINNER)}, indent=2))
    for key in by:
        print(f"\n{key}\n" + markdown_breakdown({n: by[key][n] for n in (BAR, WINNER)}, short))
    print("\n" + json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("datasets", "models")}
                             for k, v in errors["causes"].items()}, indent=2))
    print(json.dumps(tail, indent=2))
    print("\n" + markdown_lodo(loo))
    print(json.dumps({"pooled": loo["pooled"], "control": loo["control"]}, indent=2))
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/test_results.json, test_curve.png, test_by_model.png, lodo.png")


if __name__ == "__main__":
    main()
