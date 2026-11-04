"""Day 10: what the gateway does when a request outruns its bound (PRD FR6, two misprediction strategies).

Day 9 found that once the budget binds, every policy reserving less than `max_tokens` overshoots its
windows by up to 9% (finding 30), against a target of at most 1%. Two strategies, applied to the bar
(per-model p95) and to the winner alike, so the comparison between them stays paired and fair:

- **A, cap.** Each request is sent with max_tokens = ceil(bound × margin) and reserves exactly that.
  It can never cost more than it reserved, so overshoot is zero by construction (asserted on every
  run); the price is truncation. Margins 1, 1.25, 1.5, 2 and 3; the default margin is the smallest
  that truncates at most 1% of real answers on the **validation** split (the mirror of the 1%
  overshoot target), computed offline from the validation completions.
- **B, overdraft buffer.** A share of every window is kept out of admission; requests that outrun
  their reservation spend into it. Overshoot is still measured against the whole budget. The share is
  sized on the **validation** split: the smallest, on a 0.5% grid, whose largest overshoot across the
  whole Day-9 grid (35 settings, 3 seeds each) is at most 1%. Test is then run once at that share.

Truncation is split into degenerate completions (runaways and repetition loops, which a cap cuts at
no loss) and real answers. Nothing about the policies is tuned here: the bounds are Day 9's frozen
ones, which reproduce Day 7's test coverage, and on validation they must reproduce the coverage and
reservation ratio each method was frozen with.

Writes docs/figures/mispredict.json and mispredict.png. Run with `make mispredict` (about 3 minutes,
one core).
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
from llm_reserve.baselines import BAR
from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.evaluate import coverage, reservation_ratio
from llm_reserve.explore import BLUE, INK_2, INK_3, ORANGE, add_titles
from llm_reserve.features import load_features
from llm_reserve.simulate import SEEDS, Config, Policy, bounds_policy, load_rows, run, with_strategy
from llm_reserve.sweep import STRESS_RATE, WINNER, find, frozen_bounds, grid, paired, run_setting

FIG_DIR = ROOT / "docs" / "figures"
BASES = (BAR, WINNER)  # the policies both strategies are applied to
LABELS = {BAR: "Per-model p95 (the bar)", WINNER: "Learned bound (the winner)"}
COLORS = {BAR: ORANGE, WINNER: BLUE}
MARGINS = (1.0, 1.25, 1.5, 2.0, 3.0)
SHARES = tuple(round(0.005 * i, 3) for i in range(41))  # holdback search on validation: 0-20%, 0.5% steps
CURVE_SHARES = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)  # extra holdbacks run on test, for the trade-off curve
GOAL_OVERSHOOT = 0.01  # the PRD target: at most 1% of the window budget, the largest across simulator runs
GOAL_TRUNCATION = 0.01  # strategy A's default margin: at most 1% of real (non-degenerate) answers cut, on validation
VALIDATION_SEEDS = (10, 11, 12)  # not the test seeds, so validation and test share no arrival stream
TOL = 1e-6


def cap_name(base: str, margin: float) -> str:
    return f"{base}+cap{margin:g}"


def hold_name(base: str, share: float) -> str:
    return f"{base}+hold{share:g}"


def capped(bounds: np.ndarray, margin: float, max_tokens: int = Config().max_tokens) -> np.ndarray:
    """The max_tokens strategy A sends with each request: ceil(bound × margin), never above max_tokens."""
    return np.minimum(np.ceil(np.asarray(bounds, dtype=float) * margin), max_tokens)


def truncation_rates(y: np.ndarray, degenerate: np.ndarray, bounds: np.ndarray,
                     margins: Sequence[float] = MARGINS) -> dict[str, dict]:
    """Per margin, the share of completions strategy A would cut: real answers, and all of them."""
    out = {}
    for m in margins:
        cut = np.asarray(y) > capped(bounds, m)
        out[f"{m:g}"] = {"clean": float((cut & ~degenerate).mean()), "all": float(cut.mean())}
    return out


def choose_margin(rates: dict[str, dict], goal: float = GOAL_TRUNCATION) -> float:
    """The smallest margin whose real-answer truncation is at most `goal`."""
    ok = [float(m) for m, r in rates.items() if r["clean"] <= goal]
    if not ok:
        raise RuntimeError(f"no margin truncates at most {goal:.0%} of real answers")
    return min(ok)


def validation_bounds() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """The bar's and the winner's bounds per row of load_features("validate", "original"), at their frozen knobs.

    On the sanitized rows they must reproduce the coverage and ratio each was frozen with (Day 3, Day 6).
    Also returns each row's real length and whether it is degenerate.
    """
    frozen = {m.name: m for m in holdout.frozen_methods()}
    expected = {BAR: json.loads((FIG_DIR / "baselines_validate.json").read_text())["methods"][BAR],
                WINNER: json.loads((FIG_DIR / "mlp_validate.json").read_text())["methods"][WINNER]}
    rows = load_completions("validate", "original")
    _, y, meta = load_features("validate", "original")
    if not (rows.sample_id.equals(meta.sample_id) and rows.completion_idx.equals(meta.completion_idx)):
        raise AssertionError("completions and features are not row-aligned")
    fns = holdout.bound_functions(load_completions("train", "sanitized"), rows, load_prompts(),
                                  {WINNER: holdout.winner_log_pred("validate")})
    clean = ~rows.is_degenerate.to_numpy(bool)
    out = {}
    for name in BASES:
        b = fns[name](frozen[name].theta)
        cov, ratio = coverage(y[clean], b[clean]), reservation_ratio(y[clean], b[clean])
        if abs(cov - expected[name]["coverage"]) > TOL or abs(ratio - expected[name]["ratio"]) > TOL:
            raise AssertionError(f"{name}: validation coverage {cov:.6f} and ratio {ratio:.6f} differ from the frozen ones")
        out[name] = b
    return out, y, ~clean


def worst_overshoot(rows: pd.DataFrame, policy: Policy, cfg: Config, seeds: Sequence[int]) -> float:
    """The largest overshoot (share of the budget) over the seeds at one setting."""
    r = run(rows, {"p": policy}, cfg, seeds)["p"]["seeds"]
    return max(s["overshoot_max_share"] for s in r)


def size_holdback(rows: pd.DataFrame, bounds: np.ndarray, configs: Sequence[Config], seeds: Sequence[int],
                  shares: Sequence[float] = SHARES, goal: float = GOAL_OVERSHOOT) -> dict:
    """The smallest holdback share whose largest overshoot over every setting and seed is at most `goal`.

    Settings are tried worst-first (by their overshoot with no holdback), and a share is dropped at the
    first setting that breaks the goal, so a failing share's `worst` is a lower bound.
    """
    unprotected = [worst_overshoot(rows, bounds_policy(bounds), c, seeds) for c in configs]
    order = np.argsort(unprotected)[::-1]
    tried = []
    for share in shares:
        policy = with_strategy(bounds_policy(bounds), holdback=share)
        worst = 0.0
        for k in order:
            worst = max(worst, worst_overshoot(rows, policy, configs[k], seeds) if share else unprotected[k])
            if worst > goal:
                break
        tried.append({"share": share, "worst_overshoot": worst, "passed": worst <= goal})
        if worst <= goal:
            return {"share": share, "tried": tried, "unprotected_worst": float(max(unprotected)),
                    "unprotected_worst_setting": configs[order[0]].as_dict()}
    raise RuntimeError(f"no holdback up to {shares[-1]:.1%} keeps overshoot at or under {goal:.0%}")


def strategy_policies(bounds: dict[str, np.ndarray], chosen: dict[str, float]) -> dict[str, Policy]:
    """max_tokens, the two bases unprotected, each under strategy A at every margin and under B at every share."""
    out = {"max_tokens": bounds_policy(bounds["max_tokens"])}
    for base in BASES:
        out[base] = bounds_policy(bounds[base])
    for base in BASES:
        for m in MARGINS:
            out[cap_name(base, m)] = with_strategy(bounds_policy(capped(bounds[base], m)), cap=True)
        for s in sorted(set(CURVE_SHARES) | {chosen[base]}):
            out[hold_name(base, s)] = with_strategy(bounds_policy(bounds[base]), holdback=s)
    return out


def grid_extremes(results: list[dict], names: Sequence[str]) -> dict:
    """Per policy, the worst over every setting (means over seeds) and the largest single-run overshoot."""
    out = {}
    for name in names:
        ps = [r["policies"][name] for r in results]
        out[name] = {
            "overshoot_max_share": max(p["overshoot_max_share_worst_seed"] for p in ps),
            "truncated_rate_max": max(p["truncated_rate"] for p in ps),
            "truncated_clean_rate_max": max(p["truncated_clean_rate"] for p in ps),
            "vs_max_tokens_min": min(p["vs_max_tokens"]["mean"] for p in ps),
        }
    return out


def add_worst_seed(res: dict, raw: dict) -> None:
    for name, p in res["policies"].items():
        p["overshoot_max_share_worst_seed"] = max(s["overshoot_max_share"] for s in raw[name])


def run_all(rows: pd.DataFrame, policies: dict[str, Policy], cfg: Config, groups: bool = False) -> dict:
    """sweep.run_setting, plus the worst seed's overshoot (the PRD measures the largest across runs)."""
    res = run_setting(rows, policies, cfg, groups=True)
    add_worst_seed(res, {n: p["seeds"] for n, p in res["policies"].items()})
    if not groups:
        for p in res["policies"].values():
            for k in ("seeds", "by_model", "by_dataset"):
                p.pop(k)
    return res


def same_strategy_edge(res: dict) -> dict:
    """The winner's admitted ÷ the bar's when both carry the same strategy, per seed (same trace)."""
    seeds = {n: p["seeds"] for n, p in res["policies"].items()}
    out = {"none": paired(seeds, WINNER, BAR)}
    for m in MARGINS:
        out[f"cap{m:g}"] = paired(seeds, cap_name(WINNER, m), cap_name(BAR, m))
    for s in CURVE_SHARES:
        out[f"hold{s:g}"] = paired(seeds, hold_name(WINNER, s), hold_name(BAR, s))
    return out


def check_cap_never_overshoots(results: list[dict]) -> None:
    for r in results:
        for name, p in r["policies"].items():
            if "+cap" in name and (p["overshoot_max_share_worst_seed"] > 0 or p["under_reserved_tokens"] > 0):
                raise AssertionError(f"{name} overshot at {r['setting']}: a capped request cannot outrun its reservation")


def plot(stress: dict, extremes: dict, chosen: dict[str, float], margin: dict[str, float], path) -> None:
    """Admitted at the overloaded setting vs. the price paid: truncated answers (A), largest overshoot (B)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    p = stress["policies"]
    ref = p["max_tokens"]["vs_max_tokens"]["mean"]

    ax = axes[0]
    for base in BASES:
        names = [cap_name(base, m) for m in MARGINS]
        x = [p[n]["truncated_clean_rate"] * 100 for n in names]
        y = [p[n]["vs_max_tokens"]["mean"] for n in names]
        ax.plot(x, y, color=COLORS[base], linewidth=2, marker="o", markersize=5, label=LABELS[base])
        for m, xi, yi in zip(MARGINS, x, y):
            if m == margin[base]:
                ax.scatter([xi], [yi], s=120, facecolors="none", edgecolors=COLORS[base], linewidths=1.6, zorder=3)
            ax.annotate(f"×{m:g}", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8, color=INK_2)
    ax.scatter([p["max_tokens"]["truncated_clean_rate"] * 100], [ref], color=INK_3, zorder=3)
    ax.annotate("max_tokens", (p["max_tokens"]["truncated_clean_rate"] * 100, ref), textcoords="offset points",
                xytext=(5, 4), fontsize=8, color=INK_2)
    ax.set_xlabel("real answers truncated (% of admitted, degenerate excluded)")
    ax.set_ylabel(f"admitted at {STRESS_RATE:g} requests/s ÷ max_tokens")
    ax.set_title("A: cap max_tokens at bound × margin (zero overshoot)", fontsize=10, color=INK_2, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    ax = axes[1]
    for base in BASES:
        names = [base] + [hold_name(base, s) for s in sorted(set(CURVE_SHARES) | {chosen[base]})]
        shares = [0.0] + sorted(set(CURVE_SHARES) | {chosen[base]})
        x = [extremes[n]["overshoot_max_share"] * 100 for n in names]
        y = [p[n]["vs_max_tokens"]["mean"] for n in names]
        ax.plot(x, y, color=COLORS[base], linewidth=2, marker="o", markersize=5, label=LABELS[base])
        for s, xi, yi in zip(shares, x, y):
            if s == chosen[base]:
                ax.scatter([xi], [yi], s=120, facecolors="none", edgecolors=COLORS[base], linewidths=1.6, zorder=3)
            elif abs(s - chosen[base]) < 0.01:
                continue  # its label would sit on the chosen share's
            ax.annotate(f"{s:.1%}", (xi, yi), textcoords="offset points", xytext=(5, 4) if base == WINNER else (5, -11),
                        fontsize=8, color=INK_2)
    ax.axvline(GOAL_OVERSHOOT * 100, color=INK_3, linewidth=1, linestyle=":")
    ax.text(GOAL_OVERSHOOT * 100, ax.get_ylim()[0], " 1% target", fontsize=8.5, color=INK_2, va="bottom")
    ax.set_xlabel("largest overshoot over the whole grid (% of a window's budget)")
    ax.set_title("B: hold back a share of each window (no truncation)", fontsize=10, color=INK_2, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for a in axes:
        a.tick_params(colors=INK_2)
    add_titles(fig, f"What each misprediction strategy costs, test split, means of {len(SEEDS)} seeds",
               f"y: requests admitted at {STRESS_RATE:g} requests/s and 200,000 tokens per minute, ÷ max_tokens on the "
               "same traces. Left: labels are the margin. Right: labels are the share held back.\n"
               "Circled: the margin and the share chosen on validation. Overshoot is the largest of any window in any of the "
               "35 settings × 3 seeds.", top=0.83)
    fig.savefig(path)
    plt.close(fig)


def decision_rows(chosen: dict[str, float]) -> list[tuple[str, str]]:
    rows = [("max_tokens", "max_tokens = 4,096")]
    for base in BASES:
        rows.append((base, f"{LABELS[base]}, unprotected"))
        rows += [(cap_name(base, m), f"{LABELS[base]}, A ×{m:g}") for m in MARGINS]
        rows.append((hold_name(base, chosen[base]), f"{LABELS[base]}, B {chosen[base]:.1%} held back"))
    return rows


def markdown_table(default: dict, stress: dict, extremes: dict, chosen: dict[str, float]) -> str:
    lines = ["| Policy and strategy | Admitted, 8/s | ÷ max_tokens | ÷ bar unprotected | Real answers truncated | "
             "All truncated | Largest overshoot, 8/s | Largest overshoot, whole grid | Admit rate, 4/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, label in decision_rows(chosen):
        s, d, e = stress["policies"][name], default["policies"][name], extremes[name]
        lines.append(f"| {label} | {s['admitted']:,.0f} | {s['vs_max_tokens']['mean']:.3f}× | {s['vs_bar']['mean']:.3f}× | "
                     f"{s['truncated_clean_rate']:.2%} | {s['truncated_rate']:.2%} | "
                     f"{s['overshoot_max_share_worst_seed']:.2%} | {e['overshoot_max_share']:.2%} | {d['admit_rate']:.1%} |")
    return "\n".join(lines)


def main() -> None:
    base_cfg = Config()
    configs = grid(base=base_cfg)

    # B's holdback, sized on validation only.
    va_rows = load_rows("validate")
    va_bounds, va_y, va_degenerate = validation_bounds()
    margins = {}
    for base in BASES:
        rates = truncation_rates(va_y, va_degenerate, va_bounds[base])
        margins[base] = {"margin": choose_margin(rates), "validation_truncation": rates}
        print(f"{base}: validation real answers cut per margin "
              + ", ".join(f"×{m} {r['clean']:.2%}" for m, r in rates.items()) + f"; margin {margins[base]['margin']:g}")
    sizing = {}
    for base in BASES:
        sizing[base] = size_holdback(va_rows, va_bounds[base], configs, VALIDATION_SEEDS)
        print(f"{base}: validation overshoot unprotected {sizing[base]['unprotected_worst']:.2%}, "
              f"holdback sized at {sizing[base]['share']:.1%}")
    chosen = {base: sizing[base]["share"] for base in BASES}
    margin = {base: margins[base]["margin"] for base in BASES}

    # Test, run once with everything frozen.
    bounds, check = frozen_bounds("test")
    rows = load_rows("test")
    policies = strategy_policies(bounds, chosen)
    default = run_all(rows, policies, base_cfg, groups=True)
    stress = run_all(rows, policies, Config(**(base_cfg.as_dict() | {"rate": STRESS_RATE})), groups=True)
    results = []
    for cfg in configs:
        results.append(run_all(rows, policies, cfg))
        print(f"rate {cfg.rate:>4g}/s, budget {cfg.budget:>7,}, {cfg.tokens_per_second:>3g} tok/s done")
    check_cap_never_overshoots(results + [default, stress])
    extremes = grid_extremes(results, list(policies))
    edge = {"default": same_strategy_edge(default), "stress": same_strategy_edge(stress)}
    for r in (default, stress):
        for p in r["policies"].values():
            p.pop("seeds")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot(stress, extremes, chosen, margin, FIG_DIR / "mispredict.png")
    out = {"split": "test", "config": "original", "seeds": list(SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
           "default": base_cfg.as_dict(), "stress_rate": STRESS_RATE, "goal_overshoot": GOAL_OVERSHOOT,
           "margins": list(MARGINS), "curve_shares": list(CURVE_SHARES), "bounds_check": {b: check[b] for b in BASES},
           "margin_choice": margins, "chosen_margin": margin, "holdback_sizing": sizing, "chosen_holdback": chosen, "grid_extremes": extremes, "winner_vs_bar": edge,
           "default_result": default, "stress_result": stress, "sweep": results}
    (FIG_DIR / "mispredict.json").write_text(json.dumps(out, indent=2) + "\n")

    print()
    print(markdown_table(default, stress, extremes, chosen))
    print()
    print("winner ÷ bar under the same strategy, 8 requests/s:",
          ", ".join(f"{k} {v['mean']:.3f}×" for k, v in edge["stress"].items()))
    print(f"wrote {(FIG_DIR / 'mispredict.json').relative_to(ROOT)} and mispredict.png")


if __name__ == "__main__":
    main()
