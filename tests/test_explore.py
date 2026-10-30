import numpy as np
import pandas as pd
import pytest

from llm_reserve.explore import degenerate_share, log_variance_explained, within_prompt_spread


def test_within_prompt_spread_hand_computed():
    df = pd.DataFrame({
        "sample_id": ["a", "a", "a", "a", "b"],
        "output_tokens": [100, 100, 100, 300, 50],
    })
    spread = within_prompt_spread(df)
    # "b" has one completion, so no spread.
    assert list(spread.index) == ["a"]
    row = spread.loc["a"]
    assert row["mean"] == 150
    assert row["std"] == pytest.approx(100.0)  # sample std, ddof=1
    assert row["cv"] == pytest.approx(100 / 150)
    assert row["max_over_median"] == 3.0


def test_degenerate_share_per_model():
    df = pd.DataFrame({
        "model": pd.Categorical(["m1"] * 4 + ["m2"] * 2),
        "sample_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "is_degenerate": [True, False, False, False, True, True],
    })
    share = degenerate_share(df)
    assert share.loc["m1", "completions"] == 0.25
    assert share.loc["m1", "samples_any"] == 0.5
    assert share.loc["m1", "samples_all"] == 0.0
    assert share.loc["m2", "samples_all"] == 1.0
    assert list(share.index) == ["m1", "m2"]  # sorted by samples_any


def test_log_variance_explained_bounds():
    df = pd.DataFrame({
        "g": ["x", "x", "y", "y"],
        "noise": ["p", "q", "p", "q"],
        "output_tokens": np.expm1([1.0, 1.0, 3.0, 3.0]),
    })
    assert log_variance_explained(df, ["g"]) == pytest.approx(1.0)
    assert log_variance_explained(df, ["noise"]) == pytest.approx(0.0)
