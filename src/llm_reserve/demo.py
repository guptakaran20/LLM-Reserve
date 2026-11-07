"""Day-15 demo logic: what the Streamlit app computes, kept out of the UI so it can be tested without a browser.

Everything reads the self-contained `app/` folder that `demo_assets` writes (ONNX bundle with the int8 encoder,
the per-model bar, the simulator grid, the headline numbers), so it runs wherever onnxruntime does, with no
dataset, no Redis and no network. The reservation rule is the service's: a request is sent with
`max_tokens` = min(ceil(margin x bound), the client's `max_tokens`) and reserves exactly that.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[2] / "app"
MARGIN = 1.5  # the Day-10 default (findings 35)
MAX_TOKENS = 4096
POLICIES = ("max_tokens", "bar", "learned")
POLICY_LABELS = {"max_tokens": "Reserve max_tokens", "bar": "Fixed per-model p95",
                 "learned": "Learned bound (llm-reserve)"}

# Short prompts of different kinds, for the demo's picker and its request queue.
EXAMPLES: dict[str, str] = {
    "Quick fact": "What is the capital of Australia?",
    "Two sentences": "Explain in two sentences why the sky is blue.",
    "Summarize": "Summarize the plot of Romeo and Juliet in a short paragraph.",
    "List": "List 10 tips for a first-time visitor to Tokyo, with one line of explanation each.",
    "Code, short": "Write a Python function that returns True if a string is a palindrome.",
    "Code, long": ("Write a complete Python implementation of a thread-safe LRU cache with a TTL per entry, "
                   "including type hints, docstrings, and a small set of unit tests using pytest."),
    "Essay": "Write a 800-word essay on the causes and consequences of the Industrial Revolution.",
    "Chat": "hey, can you help me figure out what to cook tonight? i have rice, eggs and some spinach",
    "Rewrite": "Rewrite this sentence to sound more formal: 'we gotta fix the bug before the demo tomorrow'.",
    "Explain a concept": "Explain how gradient descent works to someone who has never studied calculus.",
    "Compare": "Compare PostgreSQL and MongoDB for a small e-commerce application and recommend one.",
    "Math": "Solve for x: 3x + 7 = 22. Show your steps.",
}


def short_name(model: str) -> str:
    return model.split("--", 1)[-1]


def estimate_input_tokens(text: str) -> int:
    """About 4 characters a token; the real count depends on the target model's tokenizer, and a gateway knows it."""
    return max(1, round(len(text) / 4))


def reserve_tokens(bound: float, max_tokens: int = MAX_TOKENS, margin: float = MARGIN) -> int:
    """What the gateway holds, and sends as `max_tokens`: min(ceil(margin x bound), the client's max_tokens)."""
    return int(min(math.ceil(margin * bound), max_tokens))


def fill_budget(reservations: list[int], budget: int) -> dict:
    """Admit requests in arrival order while each one's reservation still fits in the window (as /reserve does)."""
    spent, admitted = 0, []
    for r in reservations:
        if spent + r <= budget:
            spent += r
            admitted.append(True)
        else:
            admitted.append(False)
    return {"admitted": admitted, "count": sum(admitted), "reserved": spent, "remaining": budget - spent}


@dataclass
class Demo:
    predictor: object  # llm_reserve.serving.Predictor
    bar: dict[str, int]
    grid: dict
    headline: dict

    @classmethod
    def load(cls, app_dir: Path = APP_DIR, threads: int = 2) -> Demo:
        from llm_reserve.serving import Predictor

        predictor = Predictor(app_dir / "bundle", threads=threads, verify=True, encoder="int8")
        bar = json.loads((app_dir / "data" / "bar.json").read_text())["bounds"]
        grid = json.loads((app_dir / "data" / "sim_grid.json").read_text())
        headline = json.loads((app_dir / "data" / "headline.json").read_text())
        missing = sorted(set(predictor.models) - set(bar))
        if missing:
            raise ValueError(f"no per-model bar for {missing}")
        return cls(predictor, bar, grid, headline)

    @property
    def models(self) -> list[str]:
        return list(self.predictor.models)

    def compare(self, text: str, input_tokens: int | None = None, max_tokens: int = MAX_TOKENS,
                models: list[str] | None = None) -> pd.DataFrame:
        """One row per target model: the learned bound against the fixed p95 and `max_tokens`, and what each reserves."""
        models = models or self.models
        n_in = input_tokens or estimate_input_tokens(text)
        out = self.predictor.predict_rows(text, models, [n_in] * len(models))
        rows = []
        for m, bound in zip(models, out["bound"], strict=True):
            bound = int(bound)
            bar = self.bar[m]
            rows.append({
                "model": short_name(m), "key": m, "bound": bound, "bar": bar,
                "reserve_learned": reserve_tokens(bound, max_tokens),
                "reserve_bar": reserve_tokens(bar, max_tokens),
                "reserve_max": max_tokens,
            })
        df = pd.DataFrame(rows)
        df["saved_vs_max"] = 1 - df.reserve_learned / df.reserve_max
        df["saved_vs_bar"] = 1 - df.reserve_learned / df.reserve_bar
        return df

    def queue(self, prompts: list[str], model: str, max_tokens: int = MAX_TOKENS) -> pd.DataFrame:
        """The reservation each policy would make for each prompt, sent to one target model."""
        rows = []
        for text in prompts:
            c = self.compare(text, max_tokens=max_tokens, models=[model]).iloc[0]
            rows.append({"prompt": text, "max_tokens": int(c.reserve_max), "bar": int(c.reserve_bar),
                         "learned": int(c.reserve_learned), "bound": int(c.bound)})
        return pd.DataFrame(rows)

    def scenarios(self) -> dict[str, dict]:
        """The simulator grid's (budget, decode speed) settings that have a full arrival-rate sweep."""
        g = self.grid["grid"]
        base_budget, base_speed = self.grid["default"]["budget"], self.grid["default"]["tokens_per_second"]
        found: dict[str, dict] = {}
        for budget in g["budgets"]:
            found[f"{budget // 1000}k tokens/min, {base_speed:g} tokens/s"] = {"budget": budget, "speed": base_speed}
        for speed in g["tokens_per_second"]:
            if speed != base_speed:
                found[f"{base_budget // 1000}k tokens/min, {speed:g} tokens/s"] = {"budget": base_budget, "speed": speed}
        return found

    def sweep(self, budget: int, speed: float) -> pd.DataFrame:
        """Simulator results over arrival rate for one scenario: one row per (rate, policy)."""
        rows = []
        for r in self.grid["rows"]:
            if r["budget"] == budget and r["tokens_per_second"] == speed:
                base = r["policies"]["max_tokens"]["admitted"]
                for name, v in r["policies"].items():
                    rows.append({"rate": r["rate"], "policy": name, **v, "vs_max_tokens": v["admitted"] / base})
        return pd.DataFrame(rows)

    def headline_ratio(self, rate: float = 8.0) -> float:
        """Learned bound with the cap, admitted requests over `max_tokens`, at the Day-10 stress load."""
        d = self.grid["default"]
        s = self.sweep(d["budget"], d["tokens_per_second"])
        s = s[s.rate == rate].set_index("policy")
        return float(s.loc["winner_capped", "admitted"] / s.loc["max_tokens", "admitted"])


def reserved_per_policy(q: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Per policy: how many of the queued requests fit in `budget`, and how much of it they hold."""
    rows = []
    for p in POLICIES:
        f = fill_budget(q[p].tolist(), budget)
        rows.append({"policy": POLICY_LABELS[p], "admitted": f["count"], "of": len(q),
                     "reserved": f["reserved"], "share": f["reserved"] / budget})
    return pd.DataFrame(rows)


__all__ = ["Demo", "EXAMPLES", "POLICIES", "POLICY_LABELS", "estimate_input_tokens", "fill_budget",
           "reserve_tokens", "reserved_per_policy", "short_name"]
