import json

import pandas as pd
import pytest

from llm_reserve import data
from llm_reserve.data import LEAKY_COLUMNS, OUT_DIR, SPLITS, assign_splits, degenerate_flags, explode


def test_shared_prompt_goes_to_earliest_split():
    owner = assign_splits({
        "train": {"a", "b"},
        "validate": {"b", "c", "d"},
        "test": {"a", "d", "e"},
    })
    assert owner == {"a": "train", "b": "train", "c": "validate", "d": "validate", "e": "test"}


def test_degenerate_flags_handles_repeated_lengths():
    # Two completions of length 50; sanitized kept one of them and dropped the 15,000-token run.
    assert degenerate_flags([50, 15000, 50, 30], [30, 50, 50]) == [False, True, False, False]
    assert degenerate_flags([50, 15000, 50, 30], [50, 30]) == [False, True, True, False]


def test_degenerate_flags_whole_sample_dropped():
    assert degenerate_flags([15000, 15000], None) == [True, True]


def test_degenerate_flags_rejects_lengths_not_in_original():
    with pytest.raises(ValueError):
        degenerate_flags([10, 20], [10, 99])


def test_explode_one_row_per_completion():
    samples = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "prompt_id": ["p1", "p2"],
        "model": ["m1", "m2"],
        "input_size": [7, 9],
        "output_sizes": [[5, 15000, 6], [8]],
    })
    out = explode(samples, {"s1": [5, 6]})
    assert out.sample_id.tolist() == ["s1", "s1", "s1", "s2"]
    assert out.completion_idx.tolist() == [0, 1, 2, 0]
    assert out.output_tokens.tolist() == [5, 15000, 6, 8]
    assert out.is_degenerate.tolist() == [False, True, False, True]  # s2 is missing from sanitized


def test_read_raw_refuses_leaky_columns():
    with pytest.raises(ValueError):
        data.read_raw("sanitized", "train", columns=("prompt_id", "output_mean"))


# The tests below check the real output of `make data`. CI skips them: it does not download the dataset.
processed = pytest.mark.skipif(
    not (OUT_DIR / "manifest.json").exists(), reason="data/processed not built; run `make data`",
)


@processed
def test_no_prompt_in_two_splits():
    prompt_sets = {s: set(data.load_completions(s, "original").prompt_id) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not prompt_sets[a] & prompt_sets[b], f"{a} and {b} share prompts"
    prompts = data.load_prompts()
    assert prompts.prompt_id.is_unique
    for split, ids in prompt_sets.items():
        assert ids == set(prompts.loc[prompts.split == split, "prompt_id"])


@processed
def test_counts_match_manifest():
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    for split in SPLITS:
        stats = manifest["splits"][split]
        assert len(data.load_completions(split, "original")) == stats["completions_original"]
        assert len(data.load_completions(split, "sanitized")) == stats["completions_sanitized"]


@processed
def test_processed_files_have_no_leaky_columns():
    for path in OUT_DIR.glob("*.parquet"):
        columns = set(pd.read_parquet(path).columns)
        assert not columns & set(LEAKY_COLUMNS), path.name


@processed
def test_sanitized_view_has_no_capped_runs():
    # CASTILLO's generation cap is 15,000 tokens; any completion that hit it is degenerate.
    for split in SPLITS:
        assert data.load_completions(split, "sanitized").output_tokens.max() < 15000
