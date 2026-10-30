"""Day-2 exploration: the four plots and the numbers behind docs/findings.md.

Uses the train split only. Validate is for tuning and test is scored once, so neither is
explored here.

Writes docs/figures/*.png and docs/figures/explore_stats.json. Run with `make explore`.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_reserve.data import ROOT, load_completions, load_prompts

FIG_DIR = ROOT / "docs" / "figures"
SPLIT = "train"

# Reference palette (light mode): two categorical slots plus ink and grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 10, "figure.titlesize": 13, "figure.dpi": 110,
})


def add_titles(fig, title: str, subtitle: str, top: float = 0.9) -> None:
    """Figure-level title and subtitle above the plot area, then lay out beneath them."""
    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, top + 0.005, subtitle, color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, top))


def short_model(name: str) -> str:
    return name.split("--", 1)[-1]


def within_prompt_spread(completions: pd.DataFrame) -> pd.DataFrame:
    """Per sample (one prompt, one model): mean, std (ddof=1), CV and max / median of its completions.

    Samples with fewer than 2 completions have no spread and are dropped.
    """
    g = completions.groupby("sample_id").output_tokens.agg(["size", "mean", "std", "median", "max"])
    g = g[g["size"] >= 2]
    return g.assign(cv=g["std"] / g["mean"], max_over_median=g["max"] / g["median"])


def degenerate_share(original: pd.DataFrame) -> pd.DataFrame:
    """Per model, in the original config: share of completions flagged degenerate, and share of
    samples with at least one (sanitized has none of either, by construction)."""
    per_sample = original.groupby(["model", "sample_id"], observed=True).is_degenerate.agg(["any", "all"])
    out = pd.DataFrame({
        "completions": original.groupby("model", observed=True).is_degenerate.mean(),
        "samples_any": per_sample["any"].groupby("model", observed=True).mean(),
        "samples_all": per_sample["all"].groupby("model", observed=True).mean(),
        "samples": per_sample.groupby("model", observed=True).size(),
    })
    return out.sort_values("samples_any")


def log_variance_explained(completions: pd.DataFrame, keys: list[str]) -> float:
    """R² of log(1 + length) explained by the group mean over `keys`: an upper bound for any
    predictor that only sees those keys (in-sample)."""
    y = np.log1p(completions.output_tokens.to_numpy(dtype=float))
    fitted = pd.Series(y, index=completions.index).groupby(
        [completions[k] for k in keys], observed=True,
    ).transform("mean").to_numpy()
    return float(1 - np.mean((y - fitted) ** 2) / np.var(y))


def plot_length_by_model(s: pd.DataFrame, path) -> dict:
    g = s.groupby("model", observed=True).output_tokens
    stats = pd.DataFrame({"n": g.size(), "p50": g.median(), "p95": g.quantile(0.95), "p99": g.quantile(0.99)})
    stats = stats.sort_values("p95")
    bins = np.logspace(0, np.log10(15_000), 61)

    fig, axes = plt.subplots(4, 4, figsize=(13, 9.5), sharex=True, sharey=True)
    for ax, (model, row) in zip(axes.flat, stats.iterrows()):
        x = s.loc[s.model == model, "output_tokens"]
        ax.hist(x, bins=bins, weights=np.full(len(x), 100 / len(x)), color=BLUE, edgecolor=SURFACE, linewidth=0.4)
        ax.axvline(row.p50, color=INK_3, linewidth=1.2, linestyle=":")
        ax.axvline(row.p95, color=ORANGE, linewidth=2)
        ax.set_title(short_model(model), loc="left", color=INK)
        ax.text(0.03, 0.95, f"p50 {row.p50:,.0f}\np95 {row.p95:,.0f}", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK_2)
        ax.set_xscale("log")
    for ax in axes.flat[len(stats):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("output tokens (log scale)")
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of completions")
    add_titles(fig, "Output length per model, train split (sanitized), sorted by p95",
               "Dotted gray line: median.  Orange line: 95th percentile.", top=0.94)
    fig.savefig(path)
    plt.close(fig)
    return {short_model(m): {k: round(float(v), 1) for k, v in r.items()} for m, r in stats.iterrows()}


def plot_length_vs_input(s: pd.DataFrame, path) -> dict:
    edges = np.unique(np.quantile(s.input_size, np.linspace(0, 1, 21)))
    bucket = pd.cut(s.input_size, edges, include_lowest=True)
    q = s.groupby(bucket, observed=True).agg(
        mid=("input_size", "median"),
        p50=("output_tokens", "median"),
        p95=("output_tokens", lambda x: x.quantile(0.95)),
    )
    rho = float(s[["input_size", "output_tokens"]].corr("spearman").iloc[0, 1])

    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(s.input_size, s.output_tokens.clip(lower=1), xscale="log", yscale="log", gridsize=70,
                   bins="log", cmap="Greys", mincnt=1, linewidths=0)
    ax.plot(q.mid, q.p95, color=ORANGE, linewidth=2, marker="o", markersize=5)
    ax.plot(q.mid, q.p50, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.annotate("p95", (q.mid.iloc[-1], q.p95.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                color=INK, va="center", fontweight="bold")
    ax.annotate("p50", (q.mid.iloc[-1], q.p50.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                color=INK, va="center", fontweight="bold")
    ax.set_xlabel("input_size (prompt tokens, log scale)")
    ax.set_ylabel("output tokens (log scale)")
    fig.colorbar(hb, ax=ax, label="completions per cell", pad=0.08)
    add_titles(fig, f"Output length vs. prompt length, train split (sanitized). Spearman ρ = {rho:.2f}",
               "Gray: completion density (log count). Lines: p50 and p95 within 20 equal-count input-length bins.")
    fig.savefig(path)
    plt.close(fig)
    return {"spearman": round(rho, 3),
            "bins": [{"input_median": float(r.mid), "p50": float(r.p50), "p95": float(r.p95)}
                     for r in q.itertuples()]}


def plot_within_prompt_spread(spread: pd.DataFrame, path) -> dict:
    cv = spread.cv
    qs = cv.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    ratio = spread.max_over_median.quantile([0.5, 0.9, 0.99])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, 1.5, 76)
    ax.hist(cv.clip(upper=1.5), bins=bins, weights=np.full(len(cv), 100 / len(cv)), color=BLUE,
            edgecolor=SURFACE, linewidth=0.4)
    for p, style in ((0.5, "-"), (0.9, "--")):
        ax.axvline(qs[p], color=INK, linewidth=1.2, linestyle=style)
        ax.annotate(f"{'median' if p == 0.5 else 'p90'} {qs[p]:.2f}", (qs[p], ax.get_ylim()[1] * 0.92),
                    xytext=(5, 0), textcoords="offset points", color=INK)
    ax.set_xlabel("std ÷ mean of one prompt's completions from one model (values above 1.5 shown at 1.5)")
    ax.set_ylabel("% of (prompt, model) samples")
    add_titles(fig, "Within-prompt spread: same prompt, same model, up to 10 completions. Train split (sanitized)",
               f"{len(cv):,} samples. The longest completion is {ratio[0.5]:.2f}× the median at p50, "
               f"{ratio[0.9]:.2f}× at p90 and {ratio[0.99]:.1f}× at p99.")
    fig.savefig(path)
    plt.close(fig)
    return {"samples": int(len(cv)), "cv_quantiles": {str(k): round(float(v), 3) for k, v in qs.items()},
            "max_over_median": {str(k): round(float(v), 2) for k, v in ratio.items()}}


def plot_degenerate(share: pd.DataFrame, path) -> dict:
    names = [short_model(m) for m in share.index]
    y = np.arange(len(share))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(y + h / 2, share.samples_any * 100, height=h, color=BLUE, edgecolor=SURFACE, linewidth=1,
            label="samples with at least 1 degenerate completion")
    ax.barh(y - h / 2, share.completions * 100, height=h, color=ORANGE, edgecolor=SURFACE, linewidth=1,
            label="completions that are degenerate")
    for yi, a, c in zip(y, share.samples_any * 100, share.completions * 100):
        ax.text(a + 0.08, yi + h / 2, f"{a:.2f}%", va="center", fontsize=8.5, color=INK_2)
        ax.text(c + 0.08, yi - h / 2, f"{c:.2f}%", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% (original config)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, share.samples_any.max() * 100 * 1.15)
    add_titles(fig, "Degenerate output per model, train split: original config",
               "Sanitized removes exactly these completions, so it is 0% for every model.")
    fig.savefig(path)
    plt.close(fig)
    return {short_model(m): {"completions": round(float(r.completions), 5),
                             "samples_any": round(float(r.samples_any), 5),
                             "samples_all": round(float(r.samples_all), 5), "samples": int(r.samples)}
            for m, r in share.iterrows()}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    original = load_completions(SPLIT, "original")
    s = original[~original.is_degenerate].reset_index(drop=True)
    prompts = load_prompts()

    stats: dict = {"split": SPLIT, "completions_sanitized": len(s), "completions_original": len(original)}
    stats["overall"] = {f"p{int(q * 1000) / 10:g}": float(s.output_tokens.quantile(q))
                        for q in (0.5, 0.9, 0.95, 0.99, 0.999)}
    stats["tail"] = {
        "original_over_4096": float((original.output_tokens > 4096).mean()),
        "sanitized_over_4096": float((s.output_tokens > 4096).mean()),
        "original_at_15000_cap": int((original.output_tokens >= 15_000).sum()),
    }
    stats["per_model"] = plot_length_by_model(s, FIG_DIR / "length_by_model.png")
    stats["input_size"] = plot_length_vs_input(s, FIG_DIR / "length_vs_input.png")
    stats["within_prompt"] = plot_within_prompt_spread(within_prompt_spread(s), FIG_DIR / "within_prompt_spread.png")
    stats["degenerate"] = plot_degenerate(degenerate_share(original), FIG_DIR / "degenerate_by_model.png")

    with_dataset = s.merge(prompts[["prompt_id", "dataset"]], on="prompt_id", validate="many_to_one")
    stats["log_r2"] = {
        "+".join(keys): round(log_variance_explained(with_dataset, keys), 4)
        for keys in (["model"], ["dataset"], ["model", "dataset"], ["prompt_id"], ["sample_id"])
    }
    per_prompt = np.log1p(s.output_tokens).groupby([s.prompt_id, s.model], observed=True).mean().unstack()
    corr = per_prompt.corr().to_numpy()
    stats["cross_model_corr_median"] = round(float(np.median(corr[np.triu_indices_from(corr, 1)])), 3)
    g = with_dataset.groupby("dataset")
    stats["per_dataset"] = {
        d: {"input_median": float(x.input_size.median()), "p50": float(x.output_tokens.median()),
            "p95": float(x.output_tokens.quantile(0.95))} for d, x in g
    }

    (FIG_DIR / "explore_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in ("overall", "tail", "log_r2", "cross_model_corr_median")}, indent=2))
    print(f"wrote {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
