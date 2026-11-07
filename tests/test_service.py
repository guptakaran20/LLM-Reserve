import asyncio
import json

import fakeredis
import httpx
import numpy as np
import pytest

from llm_reserve import mispredict, service, serving
from llm_reserve.serving import Prediction

S = service.Settings(budget=10_000, window_seconds=60, prefix="t")
MODELS = ["meta-llama--Llama-3.1-8B-Instruct", "google--gemma-3-4b-it"]
HAS_BUNDLE = (serving.BUNDLE_DIR / serving.SPEC).exists()


# ---- the Lua scripts, on their own ----

@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def reserve(r, window, rid, cost, budget=S.budget):
    script = r.register_script(service.RESERVE_LUA)
    return script(keys=[S.window_key(window), S.reservation_key(window, rid)], args=[cost, budget, 120, 3600])


def settle(r, window, rid, actual):
    script = r.register_script(service.SETTLE_LUA)
    return script(keys=[S.window_key(window), S.reservation_key(window, rid)], args=[actual])


def test_reserve_admits_until_the_budget_and_an_exact_fill_fits(r):
    assert reserve(r, 7, "a", 6_000) == [1, 6_000]
    assert reserve(r, 7, "b", 4_001) == [0, 6_000]  # one token over: refused, nothing booked
    assert reserve(r, 7, "c", 4_000) == [1, 10_000]  # exactly the budget fits
    assert reserve(r, 7, "d", 1) == [0, 10_000]
    assert int(r.get(S.window_key(7))) == 10_000
    assert not r.exists(S.reservation_key(7, "b")) and not r.exists(S.reservation_key(7, "d"))
    assert int(r.get(S.reservation_key(7, "c"))) == 4_000


def test_reserve_sets_ttls_and_refuses_a_reused_id(r):
    reserve(r, 1, "a", 100)
    assert 0 < r.ttl(S.window_key(1)) <= 120 and 0 < r.ttl(S.reservation_key(1, "a")) <= 3600
    assert reserve(r, 1, "a", 100) == [-1, 100]  # never booked twice
    assert int(r.get(S.window_key(1))) == 100


def test_a_scripts_keys_share_a_cluster_slot():
    assert S.window_key(42) == "t:{42}:window" and S.reservation_key(42, "x") == "t:{42}:resv:x"


def test_windows_are_independent(r):
    reserve(r, 1, "a", 10_000)
    assert reserve(r, 1, "b", 1)[0] == 0
    assert reserve(r, 2, "c", 10_000)[0] == 1


def test_settle_charges_the_real_cost_once(r):
    reserve(r, 3, "a", 5_000)
    reserve(r, 3, "b", 2_000)
    assert settle(r, 3, "a", 1_200) == [1, 5_000, 3_200]  # INCRBY (actual - reserved) frees 3,800
    assert settle(r, 3, "a", 1_200) == [0, 0, 0]  # already settled: refused, not applied twice
    assert not r.exists(S.reservation_key(3, "a"))
    assert settle(r, 3, "b", 2_500) == [1, 2_000, 3_700]  # an under-reservation adds the difference
    assert settle(r, 3, "never", 10) == [0, 0, 0]


def test_settle_after_the_window_expired_adjusts_nothing(r):
    reserve(r, 4, "a", 500)
    r.delete(S.window_key(4))  # what the TTL does two windows later
    assert settle(r, 4, "a", 100) == [1, 500, -1]
    assert not r.exists(S.window_key(4))  # not recreated by a late INCRBY


def test_concurrent_reserves_never_pass_the_budget():
    # The race the Lua script exists for. Day 13 repeats this with 1,000 requests against real Redis.
    async def run():
        r = fakeredis.FakeAsyncRedis()
        script = r.register_script(service.RESERVE_LUA)
        costs = np.random.default_rng(0).integers(50, 400, size=300)

        async def one(i, cost):
            return await script(keys=[S.window_key(0), S.reservation_key(0, str(i))],
                                args=[int(cost), S.budget, 120, 3600])

        out = await asyncio.gather(*(one(i, c) for i, c in enumerate(costs)))
        return costs, out, int(await r.get(S.window_key(0)))

    costs, out, spent = asyncio.run(run())
    admitted = np.array([o[0] == 1 for o in out])
    assert spent <= S.budget
    assert spent == costs[admitted].sum()
    assert admitted.any() and not admitted.all()


# ---- the strategy-A cap ----

def test_capped_output_is_mispredicts_rule():
    bounds = np.array([1, 2, 3, 100, 333, 2730, 2731, 4000])
    expected = mispredict.capped(bounds, 1.5, 4096).astype(int)
    assert [service.capped_output(int(b), 4096, 1.5) for b in bounds] == expected.tolist()
    assert service.capped_output(None, 512) == 512  # no bound: the request's own max_tokens
    assert service.capped_output(100, 120) == 120


def test_default_margin_is_the_day10_decision():
    path = mispredict.FIG_DIR / "mispredict.json"
    if not path.exists():
        pytest.skip("docs/figures/mispredict.json not built")
    assert service.CAP_MARGIN == json.loads(path.read_text())["chosen_margin"]["mlp_xgb_avg"]


def test_canonical_model_accepts_hugging_face_ids():
    assert service.canonical_model("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama--Llama-3.1-8B-Instruct"
    assert service.canonical_model("google--gemma-3-4b-it") == "google--gemma-3-4b-it"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("BUDGET_TOKENS", "123")
    monkeypatch.setenv("CAP_MARGIN", "2")
    s = service.Settings.from_env()
    assert (s.budget, s.cap_margin, s.window_seconds) == (123, 2.0, 60)


# ---- the API, with a stub predictor ----

class StubPredictor:
    spec = {"winner": "stub", "members": {"a": {}, "b": {}}}
    models = MODELS

    def __init__(self):
        self.calls = []

    def predict(self, text, model, input_size):
        self.calls.append((text, model, input_size))
        return Prediction(bound=len(text) * 10, log_pred=1.0, members={"a": 0.9, "b": 1.1})


class Clock:
    def __init__(self, t=600.0):
        self.t = t

    def __call__(self):
        return self.t


async def call(app, steps):
    """Run (method, path, json) steps through the app with its lifespan; returns the responses."""
    out = []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            for method, path, body in steps:
                out.append(await c.request(method, path, json=body))
    return out


def make_app(clock=None, predictor=None):
    return service.create_app(S, predictor=predictor or StubPredictor(), redis=fakeredis.FakeAsyncRedis(),
                              clock=clock or Clock())


def test_predict_reserve_settle_flow():
    app = make_app()
    prompt = "Explain recursion."  # stub bound = 18 chars x 10 = 180
    pred, = asyncio.run(call(app, [("POST", "/predict", {"prompt": prompt, "model": MODELS[0],
                                                         "input_tokens": 40, "max_tokens": 1000})]))
    assert pred.status_code == 200, pred.text
    p = pred.json()
    assert (p["bound"], p["max_tokens"], p["reserve_tokens"]) == (180, 270, 310)
    assert p["members"] == {"a": 0.9, "b": 1.1}

    res, budget, settled, again = _flow(app, p)
    r = res.json()
    assert res.status_code == 200 and r["reserved"] == 310 and r["max_tokens"] == 270
    assert r["window"] == 10 and r["spent"] == 310 and r["remaining"] == S.budget - 310
    assert budget.json()["spent"] == 310
    s = settled.json()
    assert settled.status_code == 200
    assert (s["reserved"], s["actual"], s["delta"], s["spent"], s["overshoot"]) == (310, 240, -70, 240, 0)
    assert again.status_code == 404


def _flow(app, p):
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                res = await c.post("/reserve", json={"input_tokens": 40, "max_tokens": 1000, "bound": p["bound"]})
                budget = await c.get("/budget")
                body = {"reservation_id": res.json()["reservation_id"], "input_tokens": 40, "output_tokens": 200}
                settled = await c.post("/settle", json=body)
                again = await c.post("/settle", json=body)
                return res, budget, settled, again
    return asyncio.run(run())


def test_predict_accepts_hf_ids_and_defaults_max_tokens():
    stub = StubPredictor()
    resp, = asyncio.run(call(make_app(predictor=stub), [
        ("POST", "/predict", {"prompt": "x" * 1000, "model": "google/gemma-3-4b-it", "input_tokens": 5})]))
    assert resp.status_code == 200
    assert resp.json()["max_tokens"] == service.MAX_TOKENS  # 15,000 capped at the default 4,096
    assert stub.calls == [("x" * 1000, "google--gemma-3-4b-it", 5)]


def test_predict_rejects_unknown_models_and_bad_input():
    bad_model, negative = asyncio.run(call(make_app(), [
        ("POST", "/predict", {"prompt": "hi", "model": "openai/gpt-9", "input_tokens": 5}),
        ("POST", "/predict", {"prompt": "hi", "model": MODELS[0], "input_tokens": -1}),
    ]))
    assert bad_model.status_code == 422 and "unknown model" in bad_model.text
    assert negative.status_code == 422


def test_reserve_returns_429_with_retry_after_when_full():
    clock = Clock(615.0)  # 15 s into window 10
    full, refused, unbounded = asyncio.run(call(make_app(clock), [
        ("POST", "/reserve", {"input_tokens": 9_000, "max_tokens": 1000}),  # no bound: holds all of max_tokens
        ("POST", "/reserve", {"input_tokens": 1, "max_tokens": 100}),
        ("POST", "/reserve", {"input_tokens": 0, "max_tokens": 100, "bound": 1}),  # needs 2 tokens: still over
    ]))
    assert full.status_code == 200 and full.json()["reserved"] == 10_000
    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "45" and refused.json()["retry_after"] == 45
    assert refused.json()["requested"] == 101 and refused.json()["spent"] == 10_000
    assert unbounded.status_code == 429


def test_a_request_is_billed_to_the_window_that_admitted_it():
    clock = Clock(659.0)  # last second of window 10
    app = make_app(clock)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                res = (await c.post("/reserve", json={"input_tokens": 0, "max_tokens": 5_000})).json()
                clock.t = 661.0  # settles in window 11
                s = (await c.post("/settle", json={"reservation_id": res["reservation_id"],
                                                     "input_tokens": 0, "output_tokens": 100})).json()
                now = (await c.get("/budget")).json()
                return res, s, now

    res, s, now = asyncio.run(run())
    assert res["window"] == 10 and s["spent"] == 100  # window 10 was charged the real cost
    assert now["window"] == 11 and now["spent"] == 0  # window 11 untouched


def test_settle_rejects_malformed_and_unknown_ids():
    bad, unknown = asyncio.run(call(make_app(), [
        ("POST", "/settle", {"reservation_id": "../../etc", "input_tokens": 1, "output_tokens": 1}),
        ("POST", "/settle", {"reservation_id": f"10-{'0' * 32}", "input_tokens": 1, "output_tokens": 1}),
    ]))
    assert bad.status_code == 422 and unknown.status_code == 404


def test_healthz():
    resp, = asyncio.run(call(make_app(), [("GET", "/healthz", None)]))
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "winner": "stub", "models": 2, "budget": S.budget,
                           "window_seconds": 60, "cap_margin": 1.5}


@pytest.mark.skipif(not HAS_BUNDLE, reason="ONNX bundle not built (make onnx)")
def test_predict_serves_the_onnx_winner():
    predictor = serving.Predictor()
    prompt = "Write a haiku about the sea."
    model = predictor.models[0]
    resp, = asyncio.run(call(make_app(predictor=predictor), [
        ("POST", "/predict", {"prompt": prompt, "model": model, "input_tokens": 20})]))
    assert resp.status_code == 200, resp.text
    expected = predictor.predict(prompt, model, 20)
    assert resp.json()["bound"] == expected.bound
    assert resp.json()["max_tokens"] == service.capped_output(expected.bound, service.MAX_TOKENS)
