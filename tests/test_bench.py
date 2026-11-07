import asyncio
import json

import fakeredis
import httpx
import numpy as np
import pandas as pd
import pytest

from llm_reserve import bench, quantize, service, serving
from llm_reserve.serving import Prediction

S = service.Settings(budget=10_000, window_seconds=60, prefix="b")
MODEL = "meta-llama--Llama-3.1-8B-Instruct"
HAS_BUNDLE = (serving.BUNDLE_DIR / serving.SPEC).exists()
HAS_INT8 = HAS_BUNDLE and "int8" in json.loads((serving.BUNDLE_DIR / serving.SPEC).read_text())["encoder"]


class StubPredictor:
    spec = {"winner": "stub", "members": {"a": {}}}
    models = [MODEL]
    encoder_precision = "stub"

    def predict(self, text, model, input_size):
        return Prediction(bound=200, log_pred=5.3, members={"a": 5.3})


def app():
    return service.create_app(S, predictor=StubPredictor(), redis=fakeredis.FakeAsyncRedis(), clock=lambda: 600.0)


def reqs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"prompt": [f"prompt {i}" for i in range(n)], "model": MODEL,
                         "input_tokens": rng.integers(20, 400, n), "output_tokens": rng.integers(1, 900, n)})


def through_app(fn):
    """Run fn(client) against the app (stub predictor, fakeredis) over httpx's ASGI transport."""
    a = app()

    async def run():
        async with a.router.lifespan_context(a):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=a), base_url="http://t") as c:
                return await fn(c)
    return asyncio.run(run())


# ---- the concurrency check, in-process ----

def test_burst_of_1000_reserves_and_a_settle_race_never_pass_the_cap():
    r = reqs()
    out = through_app(lambda c: bench.burst(c, r, np.full(len(r), 200)))
    first, race = out["reserve_burst"], out["settle_race"]
    assert out["passed"]
    assert first["requests"] == 1000 and first["refused_429"] > 0 and first["admitted"] > 0
    assert first["spent_after"] == first["admitted_tokens"] <= S.budget
    assert first["max_reported_spent"] <= S.budget
    assert race["spent_after"] == race["settled_real_tokens"] + race["admitted_tokens"] <= S.budget
    assert race["settled_real_tokens"] <= first["admitted_tokens"]  # strategy A: a settle never costs more
    assert out["spent_at_end"] == race["settled_real_tokens"]  # the second wave was settled to zero
    for part in (first, race):
        assert part["responses_over_budget"] == part["wrongful_429"] == part["server_errors"] == 0


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self._body = status_code, body

    def json(self):
        return self._body


def test_check_responses_flags_each_kind_of_failure():
    ok = bench.check_responses([("reserve", 100, FakeResponse(200, {"spent": 1000})),
                                ("reserve", 500, FakeResponse(429, {"spent": 9_600})),
                                ("settle", 80, FakeResponse(200, {"spent": 900, "overshoot": 0}))], 10_000)
    assert ok == {"responses_over_budget": 0, "wrongful_429": 0, "settles_over_budget": 0, "server_errors": 0}
    bad = bench.check_responses([("reserve", 100, FakeResponse(200, {"spent": 10_001})),
                                 ("reserve", 400, FakeResponse(429, {"spent": 9_600})),  # 400 fit: wrongful
                                 ("settle", 80, FakeResponse(200, {"spent": 10_050, "overshoot": 50})),
                                 ("error", 0, FakeResponse(500, None))], 10_000)
    assert bad == {"responses_over_budget": 1, "wrongful_429": 1, "settles_over_budget": 1, "server_errors": 1}


# ---- the latency pass ----

def test_latency_pass_times_every_call_and_leaves_the_window_empty(monkeypatch):
    monkeypatch.setattr(bench, "WARMUP", 3)
    monkeypatch.setattr(bench, "PREDICT_LEVELS", {1: 25, 4: 20})  # the first level covers every request
    monkeypatch.setattr(bench, "RESERVE_LEVELS", {1: 30, 8: 40})
    r = reqs(25)

    async def run(c):
        lat, bounds = await bench.latency(c, r)
        return lat, bounds, (await c.get("/budget")).json()

    lat, bounds, budget = through_app(run)
    assert (bounds == 200).all()
    assert set(lat["predict"]) == {"1", "4"} and set(lat["reserve"]) == {"1", "8"}
    assert lat["predict"]["4"]["client_ms"]["n"] == 20 and lat["reserve"]["8"]["client_ms"]["n"] == 40
    for call in lat.values():
        for s in call.values():
            assert s["statuses"] == {"200": s["client_ms"]["n"]}
            assert 0 < s["server_ms"]["p50"] <= s["client_ms"]["p99"]  # Server-Timing came back on every call
    assert budget["spent"] == 0


def test_server_timing_header_is_on_every_response():
    async def run(c):
        return [await c.get("/healthz"), await c.post("/reserve", json={"bound": 1, "max_tokens": 5,
                                                                          "input_tokens": 1}), await c.get("/nope")]
    for r in through_app(run):
        assert r.headers["server-timing"].startswith("app;dur=") and bench.server_ms(r) >= 0


def test_percentiles_and_config_name():
    p = bench.percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["max"] == 100 and p["n"] == 100
    assert bench.config_name({"encoder": "int8", "predict_threads": 2, "workers": 1}) == "int8_t2"
    assert bench.config_name({"encoder": "int8", "predict_threads": 2, "workers": 2}) == "int8_t2_w2"
    assert bench.config_name({"encoder": "fp32", "predict_threads": 1}) == "fp32_t1"


# ---- the benchmark's own HTTP client ----

def test_client_reuses_a_keep_alive_connection_and_parses_responses():
    async def run():
        connections, seen = [], []

        async def handle(reader, writer):
            connections.append(writer)
            while line := await reader.readline():
                method, path, _ = line.decode().split()
                headers = {}
                while (h := await reader.readline()) != b"\r\n":
                    k, _, v = h.decode().partition(":")
                    headers[k.strip().lower()] = v.strip()
                body = await reader.readexactly(int(headers.get("content-length", 0)))
                seen.append((method, path, body))
                out = json.dumps({"path": path, "echo": json.loads(body) if body else None}).encode()
                status = b"429 Too Many Requests" if path == "/full" else b"200 OK"
                writer.write(b"HTTP/1.1 " + status + b"\r\ncontent-type: application/json\r\nserver-timing: app;dur=0.5"
                             b"\r\ncontent-length: " + str(len(out)).encode() + b"\r\n\r\n" + out)
                await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        c = bench.Client(f"http://127.0.0.1:{port}")
        a = await c.get("/budget")
        b = await c.post("/reserve", json={"x": 1})
        full = await c.post("/full", json={})
        await c.aclose()
        server.close()
        await server.wait_closed()
        return a, b, full, connections, seen

    a, b, full, connections, seen = asyncio.run(run())
    assert a.status_code == 200 and a.json() == {"path": "/budget", "echo": None}
    assert b.json()["echo"] == {"x": 1} and bench.server_ms(b) == 0.5
    assert full.status_code == 429
    with pytest.raises(RuntimeError, match="429"):
        full.raise_for_status()
    assert len(connections) == 1  # three requests, one connection
    assert [m for m, _, _ in seen] == ["GET", "POST", "POST"]


def test_raise_open_files_is_safe_to_call():
    bench.raise_open_files(256)


# ---- the service changes ----

def test_redis_pool_waits_instead_of_failing():
    from redis.asyncio import BlockingConnectionPool

    s = service.Settings(redis_max_connections=7, redis_pool_timeout=1.5)
    pool = service.redis_client(s).connection_pool
    assert isinstance(pool, BlockingConnectionPool)
    assert pool.max_connections == 7 and pool.timeout == 1.5


def test_settings_read_the_day13_env(monkeypatch):
    monkeypatch.setenv("ENCODER", "fp32")
    monkeypatch.setenv("PREDICT_THREADS", "3")
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "9")
    s = service.Settings.from_env()
    assert (s.encoder, s.predict_threads, s.redis_max_connections) == ("fp32", 3, 9)
    d = service.Settings()
    assert (d.encoder, d.predict_threads) == ("int8", 2)  # the Day-13 defaults (findings 45)


def test_healthz_reports_the_serving_configuration(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    h = through_app(lambda c: c.get("/healthz")).json()
    assert (h["encoder"], h["predict_threads"], h["workers"]) == ("stub", 2, 2)


# ---- the int8 encoder ----

def test_int8_acceptance_rule():
    val = {"fp32": {"coverage": 0.9500, "ratio": 1.756}}
    assert quantize.accepted({**val, "int8": {"coverage": 0.9491, "ratio": 1.764}})
    assert not quantize.accepted({**val, "int8": {"coverage": 0.9489, "ratio": 1.756}})  # 0.11 point lower
    assert not quantize.accepted({**val, "int8": {"coverage": 0.9500, "ratio": 1.766}})  # 0.57% more reserved
    assert quantize.accepted({**val, "other": {"coverage": 0.951, "ratio": 1.70}}, "other")


@pytest.mark.skipif(not HAS_BUNDLE, reason="needs the ONNX bundle (make onnx)")
def test_unknown_encoder_is_refused():
    with pytest.raises(ValueError, match="encoder must be one of"):
        serving.Predictor(encoder="int4", verify=False)


@pytest.mark.skipif(not HAS_INT8, reason="needs the int8 encoder (make onnx)")
def test_int8_encoder_is_hashed_and_close_to_fp32():
    spec = json.loads((serving.BUNDLE_DIR / serving.SPEC).read_text())
    assert spec["encoder"]["int8"]["file"] == quantize.INT8_FILE
    assert serving.sha256(serving.BUNDLE_DIR / quantize.INT8_FILE) == spec["sha256"][quantize.INT8_FILE]
    fp32, int8 = serving.Predictor(encoder="fp32"), serving.Predictor(encoder="int8")
    assert int8.encoder_precision == "int8"
    text = "Write a Python function that merges two sorted lists, and explain its time complexity."
    a, b = fp32.embed(text), int8.embed(text)
    assert float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) > 0.99
    pa, pb = fp32.predict(text, MODEL, 30), int8.predict(text, MODEL, 30)
    assert abs(pa.log_pred - pb.log_pred) < 0.1


@pytest.mark.skipif(not (bench.FIG_DIR / "int8_check.json").exists(), reason="needs make onnx")
def test_int8_check_result_is_the_recorded_decision():
    r = json.loads((bench.FIG_DIR / "int8_check.json").read_text())
    assert r["accepted"] and not r["per_tensor_accepted"] and not r["rule"]["shift_retuned"]
    assert r["accepted"] == quantize.accepted(r["validate"])
    if r["served_accepted"] is not None:  # `make int8-served`: the decision on the service image's own embeddings
        assert r["served_accepted"] == quantize.accepted(r["validate"], "int8_served")
        assert r["served_embeddings"]["system"] == "Linux" and r["served_embeddings"]["encoder"] == "int8"
    assert abs(r["validate"]["fp32"]["coverage"] - 0.95) < 1e-4  # the served fp32 path still sits on its tuned 95%


@pytest.mark.skipif(not HAS_BUNDLE, reason="needs the ONNX bundle (make onnx)")
def test_predict_rows_with_a_given_embedding_matches_the_encoders():
    p = serving.Predictor()
    text = "List three uses of a hash map."
    own = p.predict_rows(text, [MODEL], [12])
    given = p.predict_rows(text, [MODEL], [12], embedding=p.embed(text))
    assert own["winner"][0] == given["winner"][0] and own["bound"][0] == given["bound"][0]
    other = p.predict_rows(text, [MODEL], [12], embedding=p.embed("Write a long essay about the history of Rome."))
    assert other["winner"][0] != own["winner"][0]
