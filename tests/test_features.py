import json

import numpy as np
import pandas as pd
import pytest

from llm_reserve.data import LEAKY_COLUMNS, SPLITS
from llm_reserve.features import (
    EMBED_DIM, EMBED_LENGTHS, FEATURES_DIR, META_COLUMNS, NEVER_FEATURES, PROMPT_FEATURES, build_matrix,
    check_no_leaks, feature_columns, fit_ridge, length_hints, load_embeddings, load_features, prompt_features,
    sample_level, text_features,
)


def hint(text: str, name: str) -> float:
    return round(float(np.expm1(length_hints(text)[name])))


def test_length_hints_read_digits_and_number_words():
    assert hint("Summarize this in 100 words.", "hint_words") == 100
    assert hint("Answer in one sentence.", "hint_sentences") == 1
    assert hint("Explain it in a single sentence", "hint_sentences") == 1
    assert hint("Write three paragraphs about rain", "hint_paragraphs") == 3
    assert hint("List 10 tips for sleeping", "hint_items") == 10
    assert hint("Give me five ideas and 2 examples", "hint_items") == 5  # the largest count wins
    assert hint("list the top 7 rivers", "hint_items") == 7


def test_length_hints_are_zero_without_a_count():
    hints = length_hints("Write a poem about the sea, words fail me.")
    assert hints == {"hint_words": 0.0, "hint_sentences": 0.0, "hint_paragraphs": 0.0, "hint_items": 0.0}


def test_text_features_shape_and_cues():
    f = text_features("Briefly explain this code:\n```python\nx = [1, 2]\n```\nWhat does it do?")
    assert list(f) == list(PROMPT_FEATURES)
    assert f["code_fences"] == 1
    assert f["log_lines"] == pytest.approx(np.log1p(5))
    assert f["question_marks"] == 1
    assert f["cue_explain"] == f["cue_code"] == f["cue_brief"] == 1
    assert f["cue_detail"] == f["cue_creative"] == f["starts_wh"] == 0
    assert text_features("How do tides work?")["starts_wh"] == 1
    assert text_features("")["log_chars"] == 0  # empty prompts don't divide by zero


def test_cue_words_respect_word_boundaries():
    f = text_features("A shortlist of codecs")
    assert f["cue_brief"] == 0 and f["cue_code"] == 0 and f["cue_list"] == 0


def toy(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    prompts = pd.DataFrame({"prompt_id": ["p0", "p1", "p2"],
                            "prompt_text": ["List 5 fruits", "Explain gravity in detail", "def f(x): return x"]})
    by_prompt = prompt_features(prompts.prompt_text).assign(prompt_row=np.arange(3, dtype=np.int32))
    by_prompt.index = prompts.prompt_id
    rows = [(p, f"{p}_{m}", m, i, 10 + i, int(rng.integers(1, 2000)), bool(rng.random() < 0.1))
            for p in prompts.prompt_id for m in ("m1", "m2") for i in range(4)]
    completions = pd.DataFrame(rows, columns=["prompt_id", "sample_id", "model", "completion_idx", "input_size",
                                              "output_tokens", "is_degenerate"])
    return completions, by_prompt, ["m1", "m2"]


def test_build_matrix_joins_prompt_features_and_one_hot():
    completions, by_prompt, models = toy()
    m = build_matrix(completions, by_prompt, models)
    assert list(m.columns) == [*META_COLUMNS, *feature_columns(models)]
    assert len(m) == len(completions)
    assert (m.prompt_row.to_numpy() == completions.prompt_id.map({"p0": 0, "p1": 1, "p2": 2}).to_numpy()).all()
    assert (m["model=m1"] + m["model=m2"] == 1).all()
    assert m.loc[m.prompt_id == "p1", "cue_detail"].eq(1).all()
    assert m.log_input_size.to_numpy() == pytest.approx(np.log1p(completions.input_size))


def test_build_matrix_rejects_models_outside_the_train_vocabulary():
    completions, by_prompt, _ = toy()
    with pytest.raises(ValueError):
        build_matrix(completions, by_prompt, ["m1"])


def test_features_do_not_change_when_the_label_or_dataset_does():
    """Leakage, behaviourally: shuffle the label, flag and add the never-columns; the features stay put."""
    completions, by_prompt, models = toy()
    cols = feature_columns(models)
    before = build_matrix(completions, by_prompt, models)[cols]
    tampered = completions.assign(
        output_tokens=completions.output_tokens.sample(frac=1, random_state=1).to_numpy(),
        is_degenerate=~completions.is_degenerate,
        dataset="Apps", category="code", **{c: 1.0 for c in LEAKY_COLUMNS},
    )
    after = build_matrix(tampered, by_prompt, models)
    pd.testing.assert_frame_equal(before, after[cols])
    assert not set(after.columns) & (set(LEAKY_COLUMNS) | {"dataset", "category"})


def test_check_no_leaks_catches_every_never_column():
    for bad in NEVER_FEATURES:
        with pytest.raises(ValueError):
            check_no_leaks(["log_chars", bad])
    with pytest.raises(ValueError):
        check_no_leaks(["dataset=Apps"])  # a one-hot of the dataset is the dataset
    check_no_leaks(feature_columns(["m1"]))
    required = {"longest_response", "shortest_response", "output_mean", "output_std", "output_percentiles",
                "gen_time", "dataset", "category", "output_tokens", "sample_id", "temp", "top_p", "top_k"}
    assert required <= set(NEVER_FEATURES)


def test_sample_level_and_ridge_recover_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 3
    ids = np.repeat(np.arange(100), 2)
    Xs, means, counts, codes = sample_level(X[::2].repeat(2, axis=0), y[::2].repeat(2), ids)
    assert counts.tolist() == [2] * 100 and (codes == ids).all()
    assert means == pytest.approx(y[::2])
    predict = fit_ridge(Xs, means, counts.astype(float), lam=1e-6)
    assert predict(Xs) == pytest.approx(means, abs=1e-4)


built = pytest.mark.skipif(not (FEATURES_DIR / "manifest.json").exists(), reason="run `make features`")


@built
def test_built_matrix_has_no_never_columns_and_no_completion_level_signal():
    """The Day-4 gate on the real matrix: no never-column, and features are constant within each sample,
    so nothing that varies by completion (which is what the label does) can reach a model."""
    manifest = json.loads((FEATURES_DIR / "manifest.json").read_text())
    check_no_leaks(manifest["feature_columns"])
    for split in SPLITS:
        X, y, meta = load_features(split, "original")
        assert list(X.columns) == manifest["feature_columns"]
        assert len(X) == len(y) == manifest["rows"][split]
        assert not X.isna().any().any()
        assert X.groupby(meta.sample_id.to_numpy()).nunique().max().max() == 1


@built
def test_embeddings_cover_every_prompt():
    index = pd.read_parquet(FEATURES_DIR / "prompt_index.parquet")
    for n in EMBED_LENGTHS:
        e = load_embeddings(n)
        assert e.shape == (len(index), EMBED_DIM) and e.dtype == np.float32
        assert np.linalg.norm(e, axis=1) == pytest.approx(1, abs=1e-4)
    # 128 and 256 tokens agree on short prompts and differ on long ones, so they are different builds.
    assert not np.allclose(load_embeddings(128), load_embeddings(256))
