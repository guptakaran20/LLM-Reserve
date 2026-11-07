"""Day-15 demo: the reservation rule, the budget fill, the self-contained bundle and the Streamlit page.

The logic tests need the `app/` folder (`make demo-assets`); the page tests also need streamlit, which is not in the
pinned training requirements (app/requirements.txt has it). Both skip when their inputs are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_reserve import demo
from llm_reserve.demo import fill_budget, reserve_tokens

APP = Path(__file__).resolve().parents[1] / "app"
needs_app = pytest.mark.skipif(not (APP / "bundle" / "serving.json").exists(), reason="run `make demo-assets`")


def test_reserve_is_capped_at_margin_times_bound_and_at_max_tokens():
    assert reserve_tokens(200) == 300
    assert reserve_tokens(201) == 302  # ceil(301.5)
    assert reserve_tokens(3000) == 4096  # never above the client's max_tokens
    assert reserve_tokens(3000, max_tokens=8000) == 4500
    assert reserve_tokens(1, margin=1.0) == 1


def test_fill_budget_admits_in_arrival_order_and_skips_what_does_not_fit():
    f = fill_budget([400, 700, 300, 200], 1000)
    assert f["admitted"] == [True, False, True, True]  # 700 does not fit after 400, later small ones still do
    assert f["reserved"] == 900 and f["remaining"] == 100 and f["count"] == 3
    assert fill_budget([], 10)["count"] == 0
    assert fill_budget([11], 10)["admitted"] == [False]
    assert fill_budget([10], 10)["admitted"] == [True]  # exactly the budget fits, as /reserve admits


def test_examples_are_distinct_and_short_name_drops_the_vendor():
    assert len(set(demo.EXAMPLES.values())) == len(demo.EXAMPLES) >= 10
    assert demo.short_name("meta-llama--Llama-3.1-8B-Instruct") == "Llama-3.1-8B-Instruct"
    assert demo.short_name("plain") == "plain"
    assert demo.estimate_input_tokens("") == 1 and demo.estimate_input_tokens("x" * 40) == 10


@pytest.fixture(scope="module")
def loaded():
    if not (APP / "bundle" / "serving.json").exists():
        pytest.skip("run `make demo-assets`")
    return demo.Demo.load(APP, threads=1)


@needs_app
def test_the_shipped_bundle_is_int8_only_and_verified(loaded):
    spec = json.loads((APP / "bundle" / "serving.json").read_text())
    assert set(spec["sha256"]) == {p.name for p in (APP / "bundle").iterdir()} - {"serving.json"}
    assert "minilm128.onnx" not in spec["sha256"]  # the fp32 encoder is not shipped
    assert loaded.predictor.encoder_precision == "int8"
    assert len(loaded.models) == 13 and set(loaded.bar) == set(loaded.models)


@needs_app
def test_a_bundle_whose_file_does_not_match_its_hash_is_refused(tmp_path):
    from llm_reserve.serving import Predictor

    for f in (APP / "bundle").iterdir():  # symlinks: nothing is copied, nothing real is edited
        (tmp_path / f.name).symlink_to(f)
    spec = json.loads((APP / "bundle" / "serving.json").read_text())
    spec["sha256"]["mlp.onnx"] = "0" * 64
    (tmp_path / "serving.json").unlink()
    (tmp_path / "serving.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="does not match"):
        Predictor(tmp_path, threads=1, encoder="int8")


@needs_app
def test_compare_gives_one_row_per_model_with_sane_reservations(loaded):
    t = loaded.compare(demo.EXAMPLES["Two sentences"])
    assert len(t) == 13 and t.model.is_unique
    assert (t.bound >= 1).all()
    assert (t.reserve_learned <= t.reserve_max).all() and (t.reserve_bar <= t.reserve_max).all()
    assert (t.reserve_learned == np.minimum(np.ceil(1.5 * t.bound), 4096)).all()
    assert t.loc[t.key == "meta-llama--Llama-3.1-8B-Instruct", "bar"].item() == loaded.bar["meta-llama--Llama-3.1-8B-Instruct"]
    # a prompt that asks for a short answer reserves far less than one that asks for a long one
    short = loaded.compare("Answer in one word: what colour is the sky on a clear day?", models=[loaded.models[5]])
    long = loaded.compare(demo.EXAMPLES["Essay"], models=[loaded.models[5]])
    assert short.bound.item() < long.bound.item()


@needs_app
def test_a_lower_max_tokens_caps_every_reservation(loaded):
    t = loaded.compare(demo.EXAMPLES["Essay"], max_tokens=200)
    assert (t.reserve_learned <= 200).all() and (t.reserve_max == 200).all()


@needs_app
def test_trimmed_bundle_predicts_exactly_what_the_full_int8_bundle_does(loaded):
    full = Path(__file__).resolve().parents[1] / "data" / "models" / "onnx"
    if not (full / "serving.json").exists():
        pytest.skip("needs the full bundle from `make onnx`")
    from llm_reserve.serving import Predictor

    ref = Predictor(full, threads=1, encoder="int8")
    text, models = demo.EXAMPLES["Summarize"], ref.models
    a = ref.predict_rows(text, models, [20] * len(models))["bound"]
    b = loaded.predictor.predict_rows(text, models, [20] * len(models))["bound"]
    assert (a == b).all()


@needs_app
def test_simulator_grid_reproduces_the_headline_and_its_ordering(loaded):
    assert loaded.headline_ratio() == pytest.approx(1.476, abs=0.001)  # findings 33 and the README
    for name, sc in loaded.scenarios().items():
        s = loaded.sweep(**sc)
        assert sorted(s.rate.unique()) == loaded.grid["grid"]["rates"], name
        wc = s[s.policy == "winner_capped"].set_index("rate")
        assert (wc.overshoot_max_share == 0).all()  # the cap's guarantee, in every scenario
        mt = s[s.policy == "max_tokens"].set_index("rate")
        assert (wc.admitted >= mt.admitted - 1e-9).all() or name.endswith("100 tokens/s")
    assert len(loaded.scenarios()) == 5
    h = loaded.headline
    assert h["vs_bar_matched"] == pytest.approx(-0.2043, abs=1e-4) and h["gap_closed"] == pytest.approx(0.503, abs=1e-3)


@needs_app
def test_queue_and_summary_show_the_learned_bound_fitting_more_requests(loaded):
    q = loaded.queue(list(demo.EXAMPLES.values()), "meta-llama--Llama-3.1-8B-Instruct")
    s = demo.reserved_per_policy(q, 20_000).set_index("policy")
    by = {p: s.loc[demo.POLICY_LABELS[p], "admitted"] for p in demo.POLICIES}
    assert by["learned"] >= by["bar"] >= by["max_tokens"]
    assert by["learned"] > by["max_tokens"]
    assert (s.reserved <= 20_000).all()


# ---- the page itself, headless -------------------------------------------------------------------------------
@needs_app
def test_page_renders_and_responds_to_its_controls():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4 and len(at.metric) >= 8
    labels = {m.label for m in at.metric}
    assert any("Fewer tokens reserved" in x for x in labels) and any("Learned reserve" in x for x in labels)

    # pick another example: the prompt box follows and the bound changes
    before = [m.value for m in at.metric if m.label == "Predicted 95% bound"][0]
    at.selectbox(key="pick").select("Essay").run()
    assert not at.exception, at.exception
    after = [m.value for m in at.metric if m.label == "Predicted 95% bound"][0]
    assert before != after

    # an empty prompt asks for text instead of failing
    at.selectbox(key="pick").select("(my own)").run()
    assert not at.exception and any("Type a prompt" in i.value for i in at.info)

    # the budget demo reacts to a smaller window
    at.slider[0].set_value(5_000).run()
    assert not at.exception, at.exception
    at.toggle[0].set_value(False).run()  # simulator tab: switch the cap off
    assert not at.exception, at.exception
