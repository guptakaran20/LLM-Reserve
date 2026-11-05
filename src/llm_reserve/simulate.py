"""Day 8: a discrete-event simulator of a budget gateway (PRD FR6).

A gateway holds a token budget per fixed window (200,000 tokens per minute by default). Requests
arrive as a Poisson process. Each one asks to reserve its input tokens plus an output bound chosen by
a policy, capped at `max_tokens`; if that fits in what is left of the current window it is admitted,
otherwise it gets a 429 and is dropped (no retries). An admitted request runs for
output tokens ÷ tokens per second, then settles: its window is charged the real cost instead of the
reservation. That is the Redis design of Day 12: reserve with INCRBY on the window's key, settle with
INCRBY by (actual - reserved) on the same key, so a request is always billed to the window that
admitted it, even when it finishes in the next one.

Actual lengths come from the test split's `original` config, degenerate completions included, and
the provider stops at `max_tokens`, so the real cost is input + min(y, max_tokens) (PRD rev 2). A
request is one (prompt, model) sample drawn uniformly, with one of its 10 completions drawn uniformly
as what it actually produces. Every policy runs on the same trace (same arrivals, same requests,
same actual lengths), so the comparison between policies is paired.

Overshoot is what a window really spent past its budget once everything it admitted has settled. It
can only happen when a request uses more than it reserved. Under-reserved tokens, (actual -
reserved)+, are also summed per target model and per source dataset, where Day 7 found coverage
differs most (91.3% on ShareGPT to 98.3% on Dolly).

A policy is a function from the trace to one output reservation per request. Day 8 runs the
`max_tokens` policy end to end; Day 9 adds the fixed-percentile and model policies from precomputed
predictions, and Day 10 the misprediction strategies.

Writes docs/figures/sim_max_tokens.json and sim_windows.png. Run with `make sim`.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_reserve.baselines import MAX_TOKENS
from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.explore import BLUE, GRID, INK_2, INK_3, ORANGE, add_titles, short_model

FIG_DIR = ROOT / "docs" / "figures"
SPLIT = "test"
SEEDS = (0, 1, 2)
# One knob at a time around the default, for the max_tokens policy: decode speed sets how soon an
# admitted request gives its unused reservation back; Day 9 sweeps the rate and budget properly.
SENSITIVITY = {"tokens_per_second": (25.0, 50.0, 100.0), "rate": (2.0, 4.0, 8.0)}

# Event kinds. A completion sorts before an arrival at the same instant, so budget it frees is usable.
COMPLETION, ARRIVAL = 0, 1


@dataclass(frozen=True)
class Config:
    budget: int = 200_000  # tokens per window, input and output together
    window: float = 60.0  # seconds; fixed windows, each with its own budget
    rate: float = 4.0  # Poisson arrivals per second
    duration: float = 3600.0  # seconds of arrivals; the run then drains every admitted request
    tokens_per_second: float = 50.0  # decode speed: a request runs for output tokens ÷ this
    max_tokens: int = MAX_TOKENS  # every request carries it; the provider stops there
    seed: int = 0

    @property
    def n_windows(self) -> int:
        return int(np.ceil(self.duration / self.window))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Trace:
    """One arrival stream, shared by every policy. Arrays are aligned, one entry per request."""

    arrival: np.ndarray  # seconds, nondecreasing
    row: np.ndarray  # row of load_completions(split, "original"), i.e. of load_features(split, "original")
    input_tokens: np.ndarray
    output_tokens: np.ndarray  # the completion's real length, before `max_tokens` cuts it
    model: np.ndarray
    dataset: np.ndarray

    def __len__(self) -> int:
        return len(self.arrival)


Policy = Callable[[Trace], np.ndarray]  # output tokens to reserve for each request, before the max_tokens cap


def max_tokens_policy(max_tokens: int = MAX_TOKENS) -> Policy:
    """Today's gateways: reserve the worst case for every request."""
    return lambda trace: np.full(len(trace), max_tokens, dtype=np.int64)


def poisson_arrivals(rng: np.random.Generator, rate: float, duration: float) -> np.ndarray:
    """Arrival times in [0, duration) of a Poisson process: cumulative exponential gaps."""
    n = int(rate * duration + 10 * np.sqrt(rate * duration) + 10)  # enough gaps to pass `duration`
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    while t[-1] < duration:  # vanishingly rare; keep going rather than truncate the process
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(1.0 / rate, n))])
    return t[t < duration]


def make_trace(rows: pd.DataFrame, cfg: Config) -> Trace:
    """Poisson arrivals; each request is a uniformly drawn sample and one of its completions, uniformly.

    `rows` holds one row per completion with sample_id, input_size, output_tokens, model and dataset.
    """
    rng = np.random.default_rng(cfg.seed)
    arrival = poisson_arrivals(rng, cfg.rate, cfg.duration)
    codes, _ = pd.factorize(rows.sample_id)
    order = np.argsort(codes, kind="stable")  # row positions grouped by sample
    counts = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sample = rng.integers(0, len(counts), len(arrival))
    row = order[starts[sample] + rng.integers(0, counts[sample])]
    return Trace(
        arrival=arrival,
        row=row,
        input_tokens=rows.input_size.to_numpy(np.int64)[row],
        output_tokens=rows.output_tokens.to_numpy(np.int64)[row],
        model=rows.model.astype(str).to_numpy()[row],
        dataset=rows.dataset.astype(str).to_numpy()[row],
    )


@dataclass
class Result:
    """Per-request outcomes and per-window books of one run."""

    admitted: np.ndarray  # bool per request
    reserved: np.ndarray  # tokens reserved per request (input + capped output bound), admitted or not
    actual: np.ndarray  # tokens the request costs if admitted: input + min(y, max_tokens)
    window: np.ndarray  # window each request arrived in
    spend: np.ndarray  # per window: tokens actually spent once everything it admitted has settled
    peak_in_flight: int
    timeline: list[tuple[float, int, float]] | None  # (time, window, committed spend) after each event


def simulate(trace: Trace, reserved_output: np.ndarray, cfg: Config, record: bool = False) -> Result:
    """Run the gateway over `trace`, reserving input + min(reserved_output, max_tokens) per request."""
    reserved = trace.input_tokens + np.minimum(np.asarray(reserved_output, dtype=np.int64), cfg.max_tokens)
    produced = np.minimum(trace.output_tokens, cfg.max_tokens)
    actual = trace.input_tokens + produced
    runtime = produced / cfg.tokens_per_second
    window = (trace.arrival // cfg.window).astype(np.int64)
    if len(trace) and window.max() >= cfg.n_windows:
        raise ValueError("an arrival falls after the last window")

    committed = np.zeros(cfg.n_windows, dtype=np.int64)  # reserved in flight + actual of settled, per window
    admitted = np.zeros(len(trace), dtype=bool)
    events = [(float(t), ARRIVAL, i) for i, t in enumerate(trace.arrival)]
    heapq.heapify(events)
    in_flight = peak = 0
    timeline = [] if record else None
    while events:
        t, kind, i = heapq.heappop(events)
        w = window[i]
        if kind == ARRIVAL:
            if committed[w] + reserved[i] > cfg.budget:
                continue  # 429
            committed[w] += reserved[i]
            admitted[i] = True
            in_flight += 1
            peak = max(peak, in_flight)
            heapq.heappush(events, (t + runtime[i], COMPLETION, i))
        else:  # settle: the window that admitted it is charged the real cost
            committed[w] += actual[i] - reserved[i]
            in_flight -= 1
        if record:
            timeline.append((t, int(w), float(committed[w])))
    return Result(admitted, reserved, actual, window, committed, peak, timeline)


def group_table(groups: np.ndarray, trace: Trace, res: Result) -> dict:
    """Admission and under-reservation per group (target model or source dataset)."""
    under = np.where(res.admitted, np.maximum(res.actual - res.reserved, 0), 0)
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        adm = res.admitted[m]
        out[str(g)] = {
            "requests": int(m.sum()),
            "admitted": int(adm.sum()),
            "admit_rate": float(adm.mean()),
            "under_reserved_requests": int((under[m] > 0).sum()),
            "under_reserved_tokens": int(under[m].sum()),
            "used_tokens": int(res.actual[m][adm].sum()),
        }
    return out


def summarize(trace: Trace, res: Result, cfg: Config) -> dict:
    adm = res.admitted
    n = len(trace)
    over = np.maximum(res.spend - cfg.budget, 0)
    under = np.where(adm, np.maximum(res.actual - res.reserved, 0), 0)
    if res.spend.sum() != res.actual[adm].sum():
        raise AssertionError("window books do not add up to the admitted requests' real cost")
    return {
        "requests": n,
        "admitted": int(adm.sum()),
        "rejected": int(n - adm.sum()),
        "admit_rate": float(adm.mean()) if n else 0.0,
        "admitted_per_minute": float(adm.sum() / (cfg.duration / 60)),
        "truncated": int((adm & (trace.output_tokens > cfg.max_tokens)).sum()),  # cut off at max_tokens
        "overshoot_windows": int((over > 0).sum()),
        "overshoot_max_tokens": int(over.max()),
        "overshoot_max_share": float(over.max() / cfg.budget),
        "under_reserved_requests": int((under > 0).sum()),
        "under_reserved_tokens": int(under.sum()),
        "utilization": float(res.spend.mean() / cfg.budget),  # real spend ÷ budget, averaged over windows
        "reserved_per_used": float(res.reserved[adm].sum() / res.actual[adm].sum()),
        "demand_per_budget": float(res.actual.sum() / (cfg.n_windows * cfg.budget)),  # if every request got in
        "peak_in_flight": res.peak_in_flight,
    }


def load_rows(split: str = SPLIT) -> pd.DataFrame:
    """The split's completions in the `original` config (degenerate included), with each source dataset."""
    dataset = load_prompts().set_index("prompt_id").dataset
    rows = load_completions(split, "original")
    return rows.assign(dataset=rows.prompt_id.map(dataset).to_numpy())


def run(rows: pd.DataFrame, policies: dict[str, Policy], cfg: Config, seeds: Sequence[int] = SEEDS) -> dict:
    """Every policy on the same trace for each seed: summaries, plus per-model and per-dataset tables."""
    out = {name: {"seeds": [], "by_model": [], "by_dataset": []} for name in policies}
    for seed in seeds:
        c = Config(**(cfg.as_dict() | {"seed": seed}))
        trace = make_trace(rows, c)
        models = np.array([short_model(m) for m in trace.model])
        for name, policy in policies.items():
            res = simulate(trace, policy(trace), c)
            out[name]["seeds"].append(summarize(trace, res, c) | {"seed": seed})
            out[name]["by_model"].append(group_table(models, trace, res))
            out[name]["by_dataset"].append(group_table(trace.dataset, trace, res))
    return out


def mean_over_seeds(summaries: list[dict]) -> dict:
    keys = [k for k in summaries[0] if k != "seed"]
    return {k: float(np.mean([s[k] for s in summaries])) for k in keys}


def plot_windows(trace: Trace, res: Result, cfg: Config, path, minutes: int = 5) -> None:
    """Committed spend through the first few windows, and real spend per window for the whole run."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": (1.25, 1)})
    ax = axes[0]
    t, w, c = map(np.asarray, zip(*res.timeline))
    for k in range(minutes):
        m = (w == k) & (t < (k + 1) * cfg.window)  # late settles land after the window closed: not drawn
        ax.step(t[m] / 60, c[m] / 1000, where="post", color=ORANGE, linewidth=1.2,
                label="Committed: reserved in flight + settled" if k == 0 else None)
        ax.hlines(res.spend[k] / 1000, k, k + 1, color=BLUE, linewidth=2.2,
                  label="Really spent, once all settled" if k == 0 else None)
    ax.axhline(cfg.budget / 1000, color=INK_3, linewidth=1, linestyle=":")
    ax.set_xlim(0, minutes)
    ax.set_ylim(0, cfg.budget / 1000 * 1.18)
    ax.set_xlabel("minute")
    ax.set_ylabel("tokens in the window (thousands)")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.bar(np.arange(cfg.n_windows), res.spend / cfg.budget * 100, width=0.8, color=BLUE)
    ax.axhline(100, color=INK_3, linewidth=1, linestyle=":")
    ax.set_xlabel("window (minute)")
    ax.set_ylabel("real spend (% of budget)")
    ax.set_ylim(0, 110)
    ax.grid(axis="x", visible=False)
    for a in axes:
        a.tick_params(colors=INK_2)
    add_titles(fig, f"The max_tokens policy at {cfg.rate:g} requests/s and {cfg.budget:,} tokens per minute",
               f"Reservations fill each window up to the budget (left), but once its requests settle only "
               f"{res.spend.mean() / cfg.budget:.0%} of it is really spent (right). Dotted: the budget.", top=0.86)
    fig.savefig(path)
    plt.close(fig)


def markdown_summary(results: dict) -> str:
    lines = ["| Policy | Requests | Admitted | Rejected (429) | Admit rate | Truncated at max_tokens | "
             "Windows over budget | Largest overshoot | Real spend ÷ budget | Reserved ÷ used |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s = r["mean"]
        lines.append(f"| `{name}` | {s['requests']:,.0f} | {s['admitted']:,.0f} | {s['rejected']:,.0f} | "
                     f"{s['admit_rate']:.1%} | {s['truncated']:.1f} | {s['overshoot_windows']:.0f} | "
                     f"{s['overshoot_max_share']:.2%} | {s['utilization']:.1%} | {s['reserved_per_used']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    cfg = Config()
    rows = load_rows(SPLIT)
    policies = {"max_tokens": max_tokens_policy(cfg.max_tokens)}
    results = run(rows, policies, cfg)
    for r in results.values():
        r["mean"] = mean_over_seeds(r["seeds"])

    sensitivity = {
        knob: {str(v): mean_over_seeds(run(rows, policies, Config(**(cfg.as_dict() | {knob: v})))["max_tokens"]["seeds"])
               for v in values}
        for knob, values in SENSITIVITY.items()
    }

    trace = make_trace(rows, cfg)
    res = simulate(trace, policies["max_tokens"](trace), cfg, record=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_windows(trace, res, cfg, FIG_DIR / "sim_windows.png")

    out = {"split": SPLIT, "config": "original", "seeds": list(SEEDS), "settings": cfg.as_dict(), "policies": results,
           "sensitivity": sensitivity}
    (FIG_DIR / "sim_max_tokens.json").write_text(json.dumps(out, indent=2) + "\n")

    for name, r in results.items():
        for s in r["seeds"]:
            print(f"{name} seed {s['seed']}: {s['requests']:,} requests, {s['admitted']:,} admitted, "
                  f"{s['rejected']:,} rejected, overshoot {s['overshoot_max_tokens']:,} tokens in "
                  f"{s['overshoot_windows']} windows, real spend {s['utilization']:.1%} of budget")
    print()
    print(markdown_summary(results))
    print()
    for knob, by in sensitivity.items():
        for v, m in by.items():
            print(f"max_tokens at {knob} = {v}: admit rate {m['admit_rate']:.1%}, real spend {m['utilization']:.1%} "
                  f"of budget, demand {m['demand_per_budget']:.1%} of budget")
    print(f"wrote {(FIG_DIR / 'sim_max_tokens.json').relative_to(ROOT)} and sim_windows.png")


if __name__ == "__main__":
    main()
