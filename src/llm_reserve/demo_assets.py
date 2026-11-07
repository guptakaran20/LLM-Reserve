"""Day-15 demo assets: everything the Streamlit app needs, in one self-contained folder (`app/`).

The app must run on a free host that has neither the 600 MB dataset, nor the training stack, nor Redis. So this
stage copies in only what serving needs and nothing else:

* `app/bundle/`: the ONNX bundle with the **int8 encoder only** (the Day-13 default, 63 MB instead of 150 MB)
  and a `serving.json` that lists just the files it ships, with their sha256, so `Predictor` still refuses a
  bundle whose files do not match.
* `app/data/bar.json`: the per-model 95th percentile, the bar. It is fitted on train at the frozen θ from
  `baselines_validate.json`, exactly as `baselines.group_quantile` does, so the app can show the baseline
  next to the learned bound without loading the dataset.
* `app/data/sim_grid.json`: the Day-9/10 simulator results for the 35-setting grid, cut down to what the app
  plots (admitted requests, rejections, overshoot, truncation) for `max_tokens`, the bar and the winner,
  unprotected and under the chosen 1.5x cap.
* `app/data/headline.json`: the test-split numbers quoted in the app.

Nothing here is re-computed from a model: the simulator and test numbers are read from `docs/figures/`, so the
app shows what `make sweep`, `make mispredict` and `make eval` produced. Run with `make demo-assets`
(needs `make data` and `make onnx`; about 10 seconds).
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from llm_reserve.data import ROOT, load_completions
from llm_reserve.serving import BUNDLE_DIR, SPEC, sha256

APP_DIR = ROOT / "app"
OUT_BUNDLE = APP_DIR / "bundle"
OUT_DATA = APP_DIR / "data"
FIG_DIR = ROOT / "docs" / "figures"
MAX_TOKENS = 4096

# The simulator's policies as the app names them, and the keys they have in sim_policies.json / mispredict.json.
SIM_KEYS = {
    "max_tokens": ("max_tokens", "sim"),
    "bar": ("per_model", "sim"),
    "winner": ("mlp_xgb_avg", "sim"),
    "bar_capped": ("per_model+cap1.5", "mispredict"),
    "winner_capped": ("mlp_xgb_avg+cap1.5", "mispredict"),
}
SIM_FIELDS = ("admitted", "admit_rate", "rejected", "wrongful_rejections", "overshoot_max_share",
              "utilization", "truncated_clean_rate", "demand_per_budget")


def trimmed_bundle() -> dict:
    """The int8-only serving.json: the same spec, without the fp32 encoder file or its hash."""
    spec = json.loads((BUNDLE_DIR / SPEC).read_text())
    if "int8" not in spec["encoder"]:
        raise SystemExit("the bundle has no int8 encoder: run `make onnx` first")
    keep = [f for f in spec["sha256"] if f != spec["encoder"]["file"]]
    spec["sha256"] = {f: sha256(BUNDLE_DIR / f) for f in keep}
    return spec


def write_bundle() -> list[str]:
    spec = trimmed_bundle()
    shutil.rmtree(OUT_BUNDLE, ignore_errors=True)
    OUT_BUNDLE.mkdir(parents=True)
    for name in spec["sha256"]:
        shutil.copy2(BUNDLE_DIR / name, OUT_BUNDLE / name)
    (OUT_BUNDLE / SPEC).write_text(json.dumps(spec, indent=2) + "\n")
    return sorted(spec["sha256"])


def per_model_bar() -> dict:
    """Per-model percentile of train output tokens at the frozen θ, rounded up (Day 3's bar)."""
    theta = json.loads((FIG_DIR / "baselines_validate.json").read_text())["methods"]["per_model"]["theta"]
    train = load_completions("train")
    bounds = {m.replace("/", "--"): int(np.ceil(np.quantile(g.output_tokens.to_numpy(), theta)))
              for m, g in train.groupby("model")}
    return {"theta": theta, "split": "train", "config": "sanitized", "bounds": dict(sorted(bounds.items()))}


def sim_grid() -> dict:
    sim = json.loads((FIG_DIR / "sim_policies.json").read_text())
    mis = json.loads((FIG_DIR / "mispredict.json").read_text())
    by_source = {"sim": {json.dumps(s["setting"], sort_keys=True): s["policies"] for s in sim["sweep"]},
                 "mispredict": {json.dumps(s["setting"], sort_keys=True): s["policies"] for s in mis["sweep"]}}
    rows = []
    for s in sim["sweep"]:
        key = json.dumps(s["setting"], sort_keys=True)
        row = {**s["setting"], "policies": {}}
        for name, (policy, source) in SIM_KEYS.items():
            r = by_source[source][key][policy]
            row["policies"][name] = {f: r[f] for f in SIM_FIELDS}
        rows.append(row)
    return {"seeds": sim["seeds"], "default": sim["default"], "grid": sim["grid"], "margin": 1.5, "rows": rows}


def headline() -> dict:
    t = json.loads((FIG_DIR / "test_results.json").read_text())
    m, g, pooled = t["methods"], t["goals"], t["lodo"]["pooled"]
    return {
        "split": "test", "prompts": t["prompts"], "completions": t["completions_sanitized"],
        "winner": {"coverage": m["mlp_xgb_avg"]["coverage"], "ratio": m["mlp_xgb_avg"]["ratio"]},
        "bar": {"coverage": m["per_model"]["coverage"], "ratio": m["per_model"]["ratio"]},
        "oracle_ratio": pooled["oracle_ratio"],
        "vs_bar": g["vs_bar"], "vs_bar_matched": g["vs_bar_matched"],
        "gap_closed": g["gap_closed"], "vs_max_tokens": g["vs_max_tokens"],
        "unseen_dataset": {"coverage": pooled["coverage"], "matched_vs_bar": pooled["matched_vs_bar"]},
    }


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    files = write_bundle()
    print(f"bundle: {len(files)} files, {sum((OUT_BUNDLE / f).stat().st_size for f in files) / 1e6:.1f} MB")
    for name, payload in [("bar.json", per_model_bar()), ("sim_grid.json", sim_grid()), ("headline.json", headline())]:
        (OUT_DATA / name).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"data/{name}: {(OUT_DATA / name).stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
