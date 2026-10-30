"""Data pipeline: CASTILLO -> one row per completion, with leak-free splits.

Builds data/processed/ from the pinned CASTILLO parquet export:

- completions_{split}.parquet: one row per completion of the `original` config, with
  `is_degenerate` marking the completions that the `sanitized` config removed. Both
  configs are views of this one table (see `load_completions`).
- prompts.parquet: one row per prompt with its text, source dataset, category and split.
- manifest.json: row counts per split and what the overlap fix moved.

Only safe columns are ever read. The response texts and summary statistics
(LEAKY_COLUMNS) are derived from the label, so they never leave the raw files.

Run with `make data` (or `python -m llm_reserve.data`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "danfperam/castillo"
# refs/convert/parquet, generated from main@fb93f412. Pinned so `make data` is reproducible.
REVISION = "92db54f9a0227857b233f903929179e2f40b453e"

# Order matters: a prompt found in several splits is kept in the earliest one, so test stays unseen.
SPLITS = ("train", "validate", "test")
CONFIGS = ("sanitized", "original")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

RAW_COLUMNS = (
    "sample_id", "prompt_id", "model", "dataset", "category", "prompt_text", "input_size", "output_sizes",
)
# In CASTILLO but computed from the responses themselves: never read, never a feature.
LEAKY_COLUMNS = (
    "longest_response", "shortest_response", "output_mean", "output_std", "output_percentiles", "gen_time",
)


def read_raw(config: str, split: str, columns: Sequence[str] = RAW_COLUMNS) -> pd.DataFrame:
    """Read one CASTILLO split, downloading it to data/raw on first use."""
    leaked = set(columns) & set(LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"refusing to read label-derived columns: {sorted(leaked)}")
    path = hf_hub_download(
        REPO_ID, f"{config}/{split}/0000.parquet", repo_type="dataset", revision=REVISION, local_dir=RAW_DIR,
    )
    return pd.read_parquet(path, columns=list(columns))


def assign_splits(prompts_by_split: dict[str, set[str]]) -> dict[str, str]:
    """Map each prompt_id to the earliest split, in SPLITS order, that contains it."""
    owner: dict[str, str] = {}
    for split in SPLITS:
        for prompt_id in prompts_by_split.get(split, ()):
            owner.setdefault(prompt_id, split)
    return owner


def degenerate_flags(original: Sequence[int], sanitized: Sequence[int] | None) -> list[bool]:
    """Flag the completions in `original` that the sanitized config dropped.

    `sanitized` must be a sub-multiset of `original`. None means sanitized dropped the whole sample.
    """
    kept = Counter(int(n) for n in ([] if sanitized is None else sanitized))
    flags = []
    for n in original:
        n = int(n)
        if kept[n] > 0:
            kept[n] -= 1
            flags.append(False)
        else:
            flags.append(True)
    if +kept:
        raise ValueError(f"sanitized lengths {dict(+kept)} are not in the original sample")
    return flags


def explode(samples: pd.DataFrame, sanitized_sizes: dict[str, Sequence[int]]) -> pd.DataFrame:
    """One row per completion, with `is_degenerate` taken from the sanitized config."""
    lengths = samples.output_sizes.map(len).to_numpy()
    flags = [
        flag
        for sample_id, sizes in zip(samples.sample_id, samples.output_sizes)
        for flag in degenerate_flags(sizes, sanitized_sizes.get(sample_id))
    ]
    return pd.DataFrame({
        "sample_id": np.repeat(samples.sample_id.to_numpy(), lengths),
        "prompt_id": np.repeat(samples.prompt_id.to_numpy(), lengths),
        "model": pd.Categorical(np.repeat(samples.model.to_numpy(), lengths)),
        "input_size": np.repeat(samples.input_size.to_numpy(), lengths).astype(np.int32),
        "completion_idx": np.concatenate([np.arange(n) for n in lengths]).astype(np.int8),
        "output_tokens": np.concatenate([np.asarray(s) for s in samples.output_sizes]).astype(np.int32),
        "is_degenerate": np.asarray(flags, dtype=bool),
    })


def build_split(split: str, owner: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    original = read_raw("original", split)
    sanitized = read_raw("sanitized", split, columns=("sample_id", "prompt_id", "output_sizes"))
    if not set(sanitized.sample_id) <= set(original.sample_id):
        raise ValueError(f"{split}: sanitized has samples that original does not")
    if not original.sample_id.is_unique:
        raise ValueError(f"{split}: duplicate sample_id")

    keep = original.prompt_id.map(owner).eq(split)
    stats = {
        "prompts_moved_out": int(original.loc[~keep, "prompt_id"].nunique()),
        "samples_moved_out": int((~keep).sum()),
    }
    original = original[keep].reset_index(drop=True)
    sanitized = sanitized[sanitized.prompt_id.map(owner).eq(split)]

    completions = explode(original, dict(zip(sanitized.sample_id, sanitized.output_sizes)))
    expected_clean = int(sanitized.output_sizes.map(len).sum())
    if int((~completions.is_degenerate).sum()) != expected_clean:
        raise ValueError(f"{split}: non-degenerate completions do not match the sanitized config")

    prompts = original[["prompt_id", "prompt_text", "dataset", "category"]].drop_duplicates()
    if not prompts.prompt_id.is_unique:
        raise ValueError(f"{split}: a prompt_id has more than one text, dataset or category")
    prompts = prompts.assign(split=split)

    stats |= {
        "prompts": len(prompts),
        "samples": len(original),
        "samples_fully_degenerate": len(original) - len(sanitized),
        "completions_original": len(completions),
        "completions_sanitized": expected_clean,
        "completions_degenerate": int(completions.is_degenerate.sum()),
    }
    return completions, prompts, stats


def main() -> None:
    prompts_by_split = {s: set(read_raw("original", s, columns=("prompt_id",)).prompt_id) for s in SPLITS}
    owner = assign_splits(prompts_by_split)
    shared = {p for p in owner if sum(p in ids for ids in prompts_by_split.values()) > 1}
    print(f"prompts in more than one official split: {len(shared)} (each kept in its earliest split)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_prompts, manifest = [], {"revision": REVISION, "shared_prompts_moved": len(shared), "splits": {}}
    for split in SPLITS:
        completions, prompts, stats = build_split(split, owner)
        completions.to_parquet(OUT_DIR / f"completions_{split}.parquet", index=False)
        all_prompts.append(prompts)
        manifest["splits"][split] = stats

    prompts = pd.concat(all_prompts, ignore_index=True)
    if not prompts.prompt_id.is_unique:
        raise AssertionError("a prompt is still in more than one split")
    prompts.to_parquet(OUT_DIR / "prompts.parquet", index=False)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    table = pd.DataFrame(manifest["splits"]).T
    print(table.to_string())
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/")


def load_completions(split: str, config: str = "sanitized") -> pd.DataFrame:
    """Completions for one split; `config` is "sanitized" (degenerate dropped) or "original"."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    df = pd.read_parquet(OUT_DIR / f"completions_{split}.parquet")
    if config == "sanitized":
        df = df[~df.is_degenerate].reset_index(drop=True)
    return df


def load_prompts() -> pd.DataFrame:
    return pd.read_parquet(OUT_DIR / "prompts.parquet")


if __name__ == "__main__":
    main()
