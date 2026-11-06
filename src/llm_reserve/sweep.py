"""Day 9: every reservation policy in the gateway simulator, swept over arrival rate, budget and decode speed.

The policies reserve an output bound per request, before the `max_tokens` cap:

- `max_tokens`: 4,096 for everyone, today's gateways (Day 8);
- `global`: one fixed percentile of train lengths for every request;
- `per_model`: a fixed percentile per target model, the bar of the success metrics;
- `mlp_xgb_avg`: the frozen winner, the average of the MLP and XGBoost log predictions, shifted;
- `oracle`: k × the sample's own mean + std. It reads the lengths it bounds, so it is the ceiling, not a policy.

Nothing is tuned here. Every bound is computed at the knob Day 7 scored the test split with (read from
docs/figures/test_results.json, checked against the knobs frozen on validation), on the test split's
`original` config, and must reproduce Day 7's coverage and reservation ratio there before any
simulation runs. The winner's log predictions are the ones llm_reserve.predict saved, so this process
never loads torch or runs xgboost.

All policies run on the same traces (same arrivals, requests and real lengths per seed), so every
comparison is paired. The grid is arrival rate × budget at the default decode speed, plus arrival
rate × decode speed at the default budget (finding 26: decode speed moves admission as much as the
budget does), 3 seeds each. Next to admitted requests it reports wrongful rejections (429s whose real
cost would have fit), overshoot, real spend, the mean real cost of an admitted request, and admission per
source dataset and per target model, at the default setting and at an overloaded one (8 requests/s).

Writes docs/figures/sim_policies.json and sim_policies.png. Run with `make sweep`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_reserve import holdout
from llm_reserve.baselines import BAR, ORACLE
from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.evaluate import coverage, reservation_ratio
from llm_reserve.explore import BLUE, INK_2, INK_3, ORANGE, add_titles
from llm_reserve.features import load_features
from llm_reserve.simulate import SEEDS, Config, Policy, bounds_policy, load_rows, mean_over_seeds, run

FIG_DIR = ROOT / "docs" / "figures"
SPLIT = "test"
WINNER = holdout.WINNER
POLICIES = ("max_tokens", "global", BAR, WINNER, ORACLE)
LABELS = {"max_tokens": "max_tokens = 4,096", "global": "Global p95", BAR: "Per-model p95 (the bar)",
          WINNER: "Learned bound (the winner)", ORACLE: "Oracle (not deployable)"}
DEPLOYABLE = {p: p != ORACLE for p in POLICIES}
RATES = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)  # requests per second
BUDGETS = (100_000, 200_000, 400_000)  # tokens per minute
SPEEDS = (25.0, 50.0, 100.0)  # decode tokens per second
STRESS_RATE = 8.0  # an overloaded setting (demand 1.5x the default budget) for the per-group tables
TOL = 1e-9  # Day-7 coverage and ratio must reproduce to this
GOAL_VS_MAX_TOKENS = 1.5  # the PRD target: at least 1.5x the requests max_tokens admits
POLICY_COLORS = {"max_tokens": holdout.AQUA, BAR: ORANGE, WINNER: BLUE, ORACLE: INK_3}
SPEED_COLORS = {25.0: "#9cc3f0", 50.0: BLUE, 100.0: "#173f73"}  # one hue, light to dark: slow to fast


def frozen_bounds(split: str = SPLIT) -> tuple[dict[str, np.ndarray], dict]:
    """Each policy's output bound per row of load_features(split, "original"), at the knob Day 7 scored.

    Returns the bounds and, per policy, its knob with the coverage and reservation ratio it reaches on
    the original config, which must match Day 7's.
    """
    day7 = json.loads((FIG_DIR / "test_results.json").read_text())["methods"]
    frozen = {m.name: m for m in holdout.frozen_methods()}
    rows = load_completions(split, "original")
    _, y, meta = load_features(split, "original")
    if not (rows.sample_id.equals(meta.sample_id) and rows.completion_idx.equals(meta.completion_idx)):
        raise AssertionError("completions and features are not row-aligned")
    fns = holdout.bound_functions(load_completions("train", "sanitized"), rows, load_prompts(),
                                  {WINNER: holdout.winner_log_pred(split)})
    bounds, check = {}, {}
    for name in POLICIES:
        theta = day7[name]["theta"]
        if frozen[name].theta is not None and theta != frozen[name].theta:
            raise AssertionError(f"{name}: Day 7 scored θ = {theta}, validation froze {frozen[name].theta}")
        b = fns[name](theta)
        cov, ratio = coverage(y, b), reservation_ratio(y, b)
        if abs(cov - day7[name]["coverage_original"]) > TOL or abs(ratio - day7[name]["ratio_original"]) > TOL:
            raise AssertionError(f"{name}: coverage {cov:.6f} and ratio {ratio:.6f} on the original config "
                                 f"differ from Day 7's")
        bounds[name] = b
        check[name] = {"theta": theta, "coverage_original": cov, "ratio_original": ratio,
                       "mean_bound": float(b.mean())}
    return bounds, check


def grid(rates: Sequence[float] = RATES, budgets: Sequence[int] = BUDGETS, speeds: Sequence[float] = SPEEDS,
         base: Config = Config()) -> list[Config]:
    """Rate × budget at the default speed, then rate × speed at the default budget; no setting twice."""
    out = [Config(**(base.as_dict() | {"rate": r, "budget": b})) for b in budgets for r in rates]
    out += [Config(**(base.as_dict() | {"rate": r, "tokens_per_second": s})) for s in speeds for r in rates
            if s != base.tokens_per_second]
    return out


def paired(seeds: dict[str, list[dict]], name: str, reference: str) -> dict:
    """`name`'s admitted ÷ `reference`'s, per seed (same trace), then the mean and the range."""
    x = [a["admitted"] / b["admitted"] for a, b in zip(seeds[name], seeds[reference])]
    return {"mean": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def mean_groups(tables: list[dict]) -> dict:
    """Per-group tables (one per seed) averaged over seeds; admit rates recomputed from the summed counts."""
    out = {}
    for g in tables[0]:
        rows = [t[g] for t in tables]
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        m["admit_rate"] = sum(r["admitted"] for r in rows) / sum(r["requests"] for r in rows)
        out[g] = m
    return out


def setting(cfg: Config) -> dict:
    return {"rate": cfg.rate, "budget": cfg.budget, "tokens_per_second": cfg.tokens_per_second}


def run_setting(rows: pd.DataFrame, policies: dict[str, Policy], cfg: Config, seeds: Sequence[int] = SEEDS,
                groups: bool = False) -> dict:
    """Every policy at one setting: means over seeds, admitted relative to max_tokens and to the bar."""
    r = run(rows, policies, cfg, seeds)
    seeds_by = {name: r[name]["seeds"] for name in policies}
    out = {"setting": setting(cfg), "policies": {}}
    for name in policies:
        p = mean_over_seeds(seeds_by[name])
        p["admitted_sd"] = float(np.std([s["admitted"] for s in seeds_by[name]], ddof=1)) if len(seeds) > 1 else 0.0
        p["vs_max_tokens"] = paired(seeds_by, name, "max_tokens")
        if BAR in policies:
            p["vs_bar"] = paired(seeds_by, name, BAR)
        if groups:
            p["seeds"] = seeds_by[name]
            p["by_model"] = mean_groups(r[name]["by_model"])
            p["by_dataset"] = mean_groups(r[name]["by_dataset"])
        out["policies"][name] = p
    return out


def find(results: list[dict], **where) -> dict:
    hits = [r for r in results if all(r["setting"][k] == v for k, v in where.items())]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} settings match {where}")
    return hits[0]


def series(results: list[dict], policy: str, key: str, **where) -> tuple[np.ndarray, np.ndarray]:
    """(rates, value) for one policy along the rate axis at a fixed budget and speed."""
    hits = sorted((r for r in results if all(r["setting"][k] == v for k, v in where.items())),
                  key=lambda r: r["setting"]["rate"])
    v = [r["policies"][policy][key] for r in hits]
    return np.array([r["setting"]["rate"] for r in hits]), np.array([x["mean"] if isinstance(x, dict) else x for x in v])


def plot(results: list[dict], base: Config, path) -> None:
    """Admitted per minute vs. arrival rate for each policy, then the winner's gain vs. max_tokens and the bar."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": (1.3, 1, 1)})
    at = {"budget": base.budget, "tokens_per_second": base.tokens_per_second}

    ax = axes[0]
    rates = np.array(RATES)
    ax.plot(rates, rates * 60, color=INK_3, linewidth=1, linestyle=":", label="Every request (arrivals)")
    for name in ("max_tokens", BAR, WINNER, ORACLE):
        x, y = series(results, name, "admitted_per_minute", **at)
        ax.plot(x, y, color=POLICY_COLORS[name], linewidth=2, linestyle=(0, (4, 2)) if name == ORACLE else "-",
                marker="o", markersize=4, label=LABELS[name])
    ax.set_xlabel("arrival rate (requests per second)")
    ax.set_ylabel("admitted requests per minute")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    for ax, ref, label in ((axes[1], "vs_max_tokens", "max_tokens"), (axes[2], "vs_bar", "per-model p95")):
        for s in SPEEDS:
            x, y = series(results, WINNER, ref, budget=base.budget, tokens_per_second=s)
            ax.plot(x, y, color=SPEED_COLORS[s], linewidth=2, marker="o", markersize=4,
                    label=f"{s:g} tokens/s" + (" (default)" if s == base.tokens_per_second else ""))
        ax.axhline(1, color=INK_3, linewidth=1)
        if ref == "vs_max_tokens":
            ax.axhline(GOAL_VS_MAX_TOKENS, color=INK_3, linewidth=1, linestyle=":")
            ax.text(RATES[0], GOAL_VS_MAX_TOKENS, " 1.5× target", va="bottom", fontsize=8.5, color=INK_2)
        ax.set_xlabel("arrival rate (requests per second)")
        ax.set_ylabel(f"admitted requests: learned bound ÷ {label}")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for a in axes:
        a.tick_params(colors=INK_2)
    add_titles(fig, f"Admitted requests vs. arrival rate, {base.budget:,} tokens per minute, means of {len(SEEDS)} seeds",
               "Left: at 50 tokens/s. Middle and right: the learned bound's admitted requests divided by "
               "max_tokens's and by the per-model p95's, on the same traces, at three decode speeds.\n"
               "At 50 tokens/s every policy admits every request up to 2 requests/s, the learned bounds up to 4. "
               "Global p95 is in the table, close to the per-model p95.", top=0.84)
    fig.savefig(path)
    plt.close(fig)


def markdown_table(res: dict) -> str:
    lines = ["| Policy | Admitted | Admit rate | ÷ max_tokens | ÷ per-model p95 | Wrongful 429s | "
             "Windows over budget | Largest overshoot | Under-reserved tokens | Real spend ÷ budget | Reserved ÷ used | "
             "Mean admitted cost |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, p in res["policies"].items():
        lines.append(f"| {LABELS[name]} | {p['admitted']:,.0f} | {p['admit_rate']:.1%} | "
                     f"{p['vs_max_tokens']['mean']:.3f}× | {p['vs_bar']['mean']:.3f}× | {p['wrongful_rejections']:,.0f} | "
                     f"{p['overshoot_windows']:.1f} | {p['overshoot_max_share']:.2%} | "
                     f"{p['under_reserved_tokens']:,.0f} | {p['utilization']:.1%} | {p['reserved_per_used']:.2f} | "
                     f"{p['mean_admitted_cost']:.0f} |")
    return "\n".join(lines)


def markdown_sweep(results: list[dict], key: str, values: Sequence, fixed: dict) -> str:
    """Admit rate per policy and the winner's multiples, along the rate axis, for each value of `key`."""
    lines = [f"| {key} | Rate (req/s) | Demand ÷ budget | " + " | ".join(LABELS[p] for p in POLICIES)
             + " | Winner ÷ max_tokens | Winner ÷ bar | Winner overshoot windows |",
             "|---|---|---|" + "---|" * len(POLICIES) + "---|---|---|"]
    for v in values:
        for r in RATES:
            p = find(results, rate=r, **(fixed | {key: v}))["policies"]
            w = p[WINNER]
            lines.append(f"| {v:,g} | {r:g} | {w['demand_per_budget']:.0%} | "
                         + " | ".join(f"{p[n]['admit_rate']:.1%}" for n in POLICIES)
                         + f" | {w['vs_max_tokens']['mean']:.2f}× | {w['vs_bar']['mean']:.3f}× "
                         f"| {w['overshoot_windows']:.1f} |")
    return "\n".join(lines)


def main() -> None:
    base = Config()
    bounds, check = frozen_bounds(SPLIT)
    rows = load_rows(SPLIT)
    policies = {name: bounds_policy(b) for name, b in bounds.items()}

    default = run_setting(rows, policies, base, groups=True)
    stress = run_setting(rows, policies, Config(**(base.as_dict() | {"rate": STRESS_RATE})), groups=True)
    results = []
    for cfg in grid(base=base):
        results.append(run_setting(rows, policies, cfg))
        w = results[-1]["policies"][WINNER]
        print(f"rate {cfg.rate:>4g}/s, budget {cfg.budget:>7,}, {cfg.tokens_per_second:>3g} tok/s: winner admits "
              f"{w['admit_rate']:.1%}, {w['vs_max_tokens']['mean']:.2f}× max_tokens, {w['vs_bar']['mean']:.3f}× bar")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot(results, base, FIG_DIR / "sim_policies.png")
    out = {"split": SPLIT, "config": "original", "seeds": list(SEEDS), "default": base.as_dict(),
           "policies": {p: {"label": LABELS[p], "deployable": DEPLOYABLE[p], **check[p]} for p in POLICIES},
           "grid": {"rates": list(RATES), "budgets": list(BUDGETS), "tokens_per_second": list(SPEEDS)},
           "default_result": default, "stress_result": stress,
           "sweep": results}
    (FIG_DIR / "sim_policies.json").write_text(json.dumps(out, indent=2) + "\n")

    print()
    print(markdown_table(default))
    print()
    print(markdown_table(stress))
    print()
    print(markdown_sweep(results, "budget", BUDGETS, {"tokens_per_second": base.tokens_per_second}))
    print()
    print(markdown_sweep(results, "tokens_per_second", SPEEDS, {"budget": base.budget}))
    print(f"wrote {(FIG_DIR / 'sim_policies.json').relative_to(ROOT)} and sim_policies.png")


if __name__ == "__main__":
    main()
