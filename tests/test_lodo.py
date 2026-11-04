import numpy as np
import pandas as pd
import pytest

from llm_reserve import lodo
from llm_reserve.features import FEATURES_DIR, load_features


def part(datasets: list[str], control: list[bool]):
    n = len(datasets)
    X = pd.DataFrame({"f": np.arange(n, dtype=np.float32)})
    meta = pd.DataFrame({"prompt_id": [f"p{i}" for i in range(n)]})
    return X, np.arange(n), meta, np.asarray(datasets, dtype=object), np.asarray(control)


def test_holding_out_a_dataset_drops_all_of_it_including_its_control_prompts():
    p = part(["A", "A", "B", "B", "C"], [False, True, False, True, False])
    X, y, meta = lodo.without(p, "A")
    assert y.tolist() == [2, 3, 4]  # A's control prompt (row 1) leaves with the rest of A
    assert X.index.tolist() == [0, 1, 2] and meta.prompt_id.tolist() == ["p2", "p3", "p4"]


def test_the_control_drops_only_its_random_prompts():
    p = part(["A", "A", "B", "B", "C"], [False, True, False, True, False])
    assert lodo.without(p, lodo.CONTROL)[1].tolist() == [0, 2, 4]


def test_control_prompts_are_an_average_dataset_of_train_prompts():
    prompts = pd.DataFrame({
        "prompt_id": [f"p{i}" for i in range(40)],
        "dataset": ["A", "B", "C", "A"] * 10,
        "split": ["train"] * 30 + ["validate"] * 10,
    })
    ids = lodo.control_prompts(prompts)
    assert len(ids) == 10  # 30 train prompts over 3 datasets
    assert ids <= set(prompts.prompt_id[prompts.split == "train"])
    assert ids == lodo.control_prompts(prompts)  # seeded


@pytest.mark.skipif(not (lodo.LODO_DIR / f"mlp_{lodo.CONTROL}.npz").exists()
                    or not (FEATURES_DIR / "features_test.parquet").exists(), reason="run `make eval`")
def test_lodo_predictions_cover_every_row_for_every_held_out_set():
    rows = {s: len(load_features(s, "original")[1]) for s in ("validate", "test")}
    for held_out in [*lodo.datasets(), lodo.CONTROL]:
        for family in lodo.FAMILIES:
            run = lodo.load(family, held_out)
            assert {s: len(run[s]) for s in rows} == rows
            assert np.isfinite(run["test"]).all() and int(run["train_rows"]) > 0
