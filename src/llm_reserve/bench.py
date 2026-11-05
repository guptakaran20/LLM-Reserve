"""Day-13 load test and latency benchmark against the running service (`make serve`, then `make bench`).

Two parts, both over HTTP with a small asyncio client (`Client`, below), against the API and Redis in docker
compose:

1. **Latency.** Closed-loop clients send test-split requests (1,000 random prompts, one model each, seed 0)
   after a warm-up: `/predict` at 1, 4 and 8 concurrent clients, `/reserve` at 1, 16 and 64. Each
   reservation is settled at once with zero tokens, so the benchmark leaves the window empty. The gate is
   the PRD's, at one client: `/predict` p99 <= 15 ms (embedding included), `/reserve` p99 <= 5 ms. Every
   response carries `Server-Timing: app;dur=...` (service.ServerTiming), so the client latency splits into
   time inside the app and time in HTTP, Docker's port forwarding and the client itself.

2. **Concurrency.** 1,000 `/reserve` calls fired at once, each holding what the gateway would reserve for a
   test request (input + strategy-A cap, from the served bound). The budget binds (they ask for several
   windows' worth), and the checks are: no response ever reports a window over budget; every 429 asked for
   more than was left at that moment (no wrongful refusal by the script); the window's counter equals the
   sum of the admitted reservations exactly. Then 1,000 more reserves race 1,000 settles of the admitted ones
   (each charged its real answer length, cut at the cap as a provider would) and the same checks hold, with
   the counter equal to the real cost settled plus the new reservations. Everything is settled at the end.

The client is a keep-alive HTTP/1.1 connection per worker on asyncio streams, not httpx: httpx's connection
pool capped the first runs at ~400 requests/s with 100-370 ms p99 at 16-64 clients while the app's own p99
stayed under 4 ms (findings 46). This client adds ~0.1 ms. The functions below take any client with httpx's
`get`/`post` interface, so the tests drive them through httpx's ASGI transport.

Writes docs/figures/bench_runs/<encoder>_t<threads>.json for the configuration it finds at /healthz,
bench.json for the default one (int8, 2 threads), and bench.png with every saved configuration. The burst
waits for a window with at least MIN_WINDOW_LEFT seconds to go, so it lands in one window.
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from llm_reserve.data import ROOT, load_completions, load_prompts

URL = os.environ.get("LLM_RESERVE_URL", "http://localhost:8000")
FIG_DIR = ROOT / "docs" / "figures"
RUNS_DIR = FIG_DIR / "bench_runs"
DEFAULT_CONFIG = "int8_t2"  # the compose defaults (service.Settings)
SPLIT = "test"
N_PROMPTS = 1000
SEED = 0
WARMUP = 50
PREDICT_LEVELS = {1: 1000, 4: 600, 8: 800}  # concurrent clients: requests (kept small: one laptop, nice)
RESERVE_LEVELS = {1: 2000, 16: 4000, 64: 6000}
BURST = 1000
MIN_WINDOW_LEFT = 20  # seconds; the burst and its checks take about two
TARGET_MS = {"predict": 15.0, "reserve": 5.0}  # PRD: p99 on CPU
MAX_TOKENS = 4096
YTICKS = {"predict": [5, 10, 15, 20, 30, 40, 60], "reserve": [0.2, 0.5, 1, 2, 5, 10, 20, 50]}


class Response:
    """The part of Response the benchmark uses."""

    def __init__(self, status_code: int, headers: dict[str, str], content: bytes):
        self.status_code, self.headers, self.content = status_code, headers, content

    @property
    def text(self) -> str:
        return self.content.decode()

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self) -> Response:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")
        return self


class Client:
    """Minimal keep-alive HTTP/1.1 client on asyncio streams: an idle connection is reused, else one is opened.

    Enough for this service (JSON bodies with Content-Length, no chunking, no TLS); about 0.1 ms per request.
    """

    def __init__(self, url: str):
        u = urlsplit(url)
        self.host, self.port = u.hostname, u.port or 80
        self.idle: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def open(self, n: int) -> None:
        """Pre-open n connections, so no measured request pays for a TCP handshake."""
        self.idle += await asyncio.gather(*(asyncio.open_connection(self.host, self.port)
                                            for _ in range(n - len(self.idle))))

    async def request(self, method: str, path: str, body: dict | None = None) -> Response:
        reader, writer = self.idle.pop() if self.idle else await asyncio.open_connection(self.host, self.port)
        data = b"" if body is None else json.dumps(body).encode()
        writer.write(f"{method} {path} HTTP/1.1\r\nHost: {self.host}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        status = int((await reader.readline()).split()[1])
        headers = {}
        while (line := await reader.readline()) not in (b"\r\n", b""):
            k, _, v = line.decode().partition(":")
            headers[k.strip().lower()] = v.strip()
        if "chunked" in headers.get("transfer-encoding", ""):
            raise NotImplementedError("chunked responses")
        content = await reader.readexactly(int(headers.get("content-length", 0)))
        if headers.get("connection", "").lower() == "close":
            writer.close()
        else:
            self.idle.append((reader, writer))
        return Response(status, headers, content)

    async def get(self, path: str) -> Response:
        return await self.request("GET", path)

    async def post(self, path: str, json: dict) -> Response:  # noqa: A002 - httpx's keyword
        return await self.request("POST", path, json)

    async def aclose(self) -> None:
        for _, writer in self.idle:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for _, writer in self.idle), return_exceptions=True)
        self.idle.clear()


def requests(n_prompts: int = N_PROMPTS, seed: int = SEED) -> pd.DataFrame:
    """One (prompt, model) request per sampled test prompt, with a real answer length to settle with.

    `output_tokens` is one of the 10 recorded completions, picked at random: what the call would have cost.
    """
    c = load_completions(SPLIT, "original")
    rng = np.random.default_rng(seed)
    prompts = rng.choice(np.sort(c.prompt_id.unique()), size=n_prompts, replace=False)
    c = c[c.prompt_id.isin(prompts)]
    picked = c.groupby("prompt_id").sample(1, random_state=seed).sample(frac=1, random_state=seed)
    text = load_prompts().set_index("prompt_id").prompt_text
    return pd.DataFrame({"prompt": text.loc[picked.prompt_id].to_numpy(), "model": picked.model.astype(str).to_numpy(),
                         "input_tokens": picked.input_size.astype(int).to_numpy(),
                         "output_tokens": picked.output_tokens.astype(int).to_numpy()})


def percentiles(ms: list[float]) -> dict:
    p = np.percentile(np.asarray(ms), [50, 90, 95, 99, 99.9])
    return {"p50": float(p[0]), "p90": float(p[1]), "p95": float(p[2]), "p99": float(p[3]), "p999": float(p[4]),
            "max": float(np.max(ms)), "n": len(ms)}


def server_ms(r: Response) -> float:
    return float(r.headers["server-timing"].split("dur=")[1])


@dataclass
class Timed:
    client_ms: list[float]
    server_ms: list[float]
    wall_s: float
    statuses: dict[int, int]

    def summary(self) -> dict:
        overhead = np.asarray(self.client_ms) - np.asarray(self.server_ms)
        return {"client_ms": percentiles(self.client_ms), "server_ms": percentiles(self.server_ms),
                "overhead_ms": percentiles(list(overhead)), "throughput_rps": len(self.client_ms) / self.wall_s,
                "statuses": {str(k): v for k, v in sorted(self.statuses.items())}}


async def closed_loop(client: Client, call, n: int, concurrency: int) -> Timed:
    """`concurrency` workers each send their next request as soon as the last one returns; n requests in all.

    `call(i)` makes request i and returns the response to time; anything it does afterwards is untimed.
    """
    client_ms, srv_ms, statuses = [], [], {}
    counter = iter(range(n))
    if hasattr(client, "open"):
        await client.open(concurrency)

    async def worker():
        for i in counter:
            t0 = time.perf_counter()
            r, after = await call(i)
            client_ms.append((time.perf_counter() - t0) * 1e3)
            srv_ms.append(server_ms(r))
            statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
            if after is not None:
                await after
    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return Timed(client_ms, srv_ms, time.perf_counter() - t0, statuses)


def predict_body(row) -> dict:
    return {"prompt": row.prompt, "model": row.model, "input_tokens": int(row.input_tokens), "max_tokens": MAX_TOKENS}


async def post(client: Client, path: str, body: dict) -> Response:
    return await client.post(path, json=body)


async def latency(client: Client, reqs: pd.DataFrame) -> tuple[dict, np.ndarray]:
    rows = list(reqs.itertuples())
    bounds = np.zeros(len(rows), dtype=int)

    async def predict(i):
        row = rows[i % len(rows)]
        r = await post(client, "/predict", predict_body(row))
        r.raise_for_status()
        bounds[i % len(rows)] = r.json()["bound"]
        return r, None

    async def settle_zero(r):
        s = await post(client, "/settle", {"reservation_id": r.json()["reservation_id"], "input_tokens": 0,
                                           "output_tokens": 0})
        s.raise_for_status()

    async def reserve(i):
        # One token, settled to zero at once: the latency pass never competes for the window's budget (64
        # clients of real requests would hold ~64k tokens in flight). The cost does not change the script's work.
        r = await post(client, "/reserve", {"input_tokens": 0, "max_tokens": 1})
        r.raise_for_status()
        return r, settle_zero(r)

    out = {"predict": {}, "reserve": {}}
    await closed_loop(client, predict, WARMUP, 1)
    for c, n in PREDICT_LEVELS.items():
        out["predict"][str(c)] = (await closed_loop(client, predict, n, c)).summary()
        print(f"/predict  x{c:<3} {fmt(out['predict'][str(c)])}", flush=True)
    assert (bounds > 0).all(), "the latency pass predicts every request before /reserve uses its bound"
    await closed_loop(client, reserve, WARMUP, 1)
    for c, n in RESERVE_LEVELS.items():
        out["reserve"][str(c)] = (await closed_loop(client, reserve, n, c)).summary()
        print(f"/reserve  x{c:<3} {fmt(out['reserve'][str(c)])}", flush=True)
    return out, bounds


def fmt(s: dict) -> str:
    c, v = s["client_ms"], s["server_ms"]
    return (f"client p50 {c['p50']:.2f} p99 {c['p99']:.2f} ms | app p50 {v['p50']:.2f} p99 {v['p99']:.2f} ms | "
            f"{s['throughput_rps']:.0f} req/s")


async def fresh_window(client: Client) -> dict:
    b = (await client.get("/budget")).json()
    if b["resets_in"] < MIN_WINDOW_LEFT:
        await asyncio.sleep(b["resets_in"] + 0.5)
        b = (await client.get("/budget")).json()
    return b


def check_responses(results: list[tuple[str, int, Response]], budget: int) -> dict:
    """Every 200 leaves the window at or under budget; every 429 asked for more than was left then."""
    over, wrongful, overshoot, errors = 0, 0, 0, 0
    for kind, cost, r in results:
        if kind == "error" or r.status_code not in (200, 429):
            errors += 1
            continue
        j = r.json()
        if kind == "reserve" and r.status_code == 200:
            over += j["spent"] > budget
        elif kind == "reserve" and r.status_code == 429:
            wrongful += j["spent"] + cost <= budget
        elif kind == "settle":
            overshoot += j["overshoot"] > 0 or (j["spent"] is not None and j["spent"] > budget)
    return {"responses_over_budget": int(over), "wrongful_429": int(wrongful), "settles_over_budget": int(overshoot),
            "server_errors": int(errors)}


async def burst(client: Client, reqs: pd.DataFrame, bounds: np.ndarray) -> dict:
    """1,000 parallel reserves, then 1,000 more racing 1,000 settles; see the module docstring."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(reqs), size=BURST, replace=BURST > len(reqs))
    rows, b = reqs.iloc[idx], bounds[idx]
    start = await fresh_window(client)
    budget, base = start["budget"], start["spent"]

    async def reserve(row, bound):
        body = {"input_tokens": int(row.input_tokens), "max_tokens": MAX_TOKENS, "bound": int(bound)}
        r = await post(client, "/reserve", body)
        if r.status_code not in (200, 429):
            return "error", 0, r
        cost = r.json()["reserved"] if r.status_code == 200 else r.json()["requested"]
        return "reserve", int(cost), r

    async def settle(res: Response, row):
        j = res.json()
        out = min(int(row.output_tokens), j["max_tokens"])  # the provider stops at the cap it was sent
        r = await post(client, "/settle", {"reservation_id": j["reservation_id"], "input_tokens": int(row.input_tokens),
                                           "output_tokens": out})
        return "settle", int(row.input_tokens) + out, r

    t0 = time.perf_counter()
    first = await asyncio.gather(*(reserve(row, bd) for row, bd in zip(rows.itertuples(), b)))
    t_first = time.perf_counter() - t0
    after_first = (await client.get("/budget")).json()
    admitted = [(cost, r, row) for (_, cost, r), row in zip(first, rows.itertuples()) if r.status_code == 200]
    asked = sum(c for _, c, _ in first)
    windows = {r.json()["window"] for _, r, _ in admitted}
    part1 = {
        "requests": BURST, "seconds": t_first, "asked_tokens": int(asked), "budget": budget, "spent_before": base,
        "admitted": len(admitted), "refused_429": sum(r.status_code == 429 for _, _, r in first),
        "admitted_tokens": int(sum(c for c, _, _ in admitted)), "spent_after": after_first["spent"],
        "max_reported_spent": max((int(r.json()["spent"]) for _, _, r in first if r.status_code in (200, 429)),
                                  default=None),
        "windows": sorted(windows), **check_responses(first, budget),
    }
    part1["counter_matches"] = (len(windows) == 1 and after_first["window"] in windows
                                and part1["spent_after"] == base + part1["admitted_tokens"])

    # 1,000 new reserves race the settles of everything admitted above, in random order.
    idx2 = rng.choice(len(reqs), size=BURST, replace=BURST > len(reqs))
    rows2, b2 = reqs.iloc[idx2], bounds[idx2]
    jobs = [settle(r, row) for _, r, row in admitted] + [reserve(row, bd) for row, bd in zip(rows2.itertuples(), b2)]
    order = rng.permutation(len(jobs))
    t0 = time.perf_counter()
    second = await asyncio.gather(*(jobs[k] for k in order))
    t_second = time.perf_counter() - t0
    after = (await client.get("/budget")).json()
    settled_cost = sum(c for kind, c, _ in second if kind == "settle")
    new = [(c, r) for kind, c, r in second if kind == "reserve" and r.status_code == 200]
    windows2 = {r.json()["window"] for _, r in new} | windows
    part2 = {
        "settles": len(admitted), "reserves": BURST, "seconds": t_second, "settled_real_tokens": int(settled_cost),
        "freed_tokens": int(part1["admitted_tokens"] - settled_cost), "admitted": len(new),
        "refused_429": sum(kind == "reserve" and r.status_code == 429 for kind, _, r in second),
        "admitted_tokens": int(sum(c for c, _ in new)), "spent_after": after["spent"],
        "max_reported_spent": max((int(r.json()["spent"]) for _, _, r in second
                                   if r.status_code in (200, 429) and r.json().get("spent") is not None), default=None),
        "windows": sorted(windows2), **check_responses(second, budget),
    }
    part2["counter_matches"] = (len(windows2) == 1 and after["window"] in windows2
                                and part2["spent_after"] == base + settled_cost + part2["admitted_tokens"])

    # Clean up: settle the second wave's reservations to zero.
    await asyncio.gather(*(post(client, "/settle", {"reservation_id": r.json()["reservation_id"], "input_tokens": 0,
                                                    "output_tokens": 0}) for _, r in new))
    end = (await client.get("/budget")).json()
    passed = all(p[k] == 0 for p in (part1, part2) for k in ("responses_over_budget", "wrongful_429",
                                                             "settles_over_budget", "server_errors")) \
        and part1["counter_matches"] and part2["counter_matches"] and part1["refused_429"] > 0 \
        and part2["refused_429"] > 0
    return {"reserve_burst": part1, "settle_race": part2, "spent_at_end": end["spent"], "passed": bool(passed)}


def raise_open_files(n: int = 4096) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = n if hard == resource.RLIM_INFINITY else min(n, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


def config_name(service: dict) -> str:
    """e.g. int8_t2 (encoder, ONNX threads), with _w2 appended when uvicorn runs more than one worker."""
    workers = int(service.get("workers") or 1)
    return f"{service.get('encoder')}_t{service.get('predict_threads')}" + (f"_w{workers}" if workers > 1 else "")


def plot(runs: dict[str, dict]) -> None:
    """p50 and p99 vs concurrent clients for each service configuration benchmarked, both calls."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = dict(zip(sorted(runs), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, call in zip(axes, ("predict", "reserve")):
        for name, result in sorted(runs.items()):
            levels = result["latency"][call]
            xs = [int(c) for c in levels]
            ax.plot(xs, [levels[str(c)]["client_ms"]["p99"] for c in xs], "-o", color=colors[name],
                    label=f"{name} p99")
            ax.plot(xs, [levels[str(c)]["client_ms"]["p50"] for c in xs], "--", color=colors[name], alpha=0.6,
                    label=f"{name} p50")
        ax.axhline(TARGET_MS[call], color="crimson", lw=1, ls=":", label=f"target p99 {TARGET_MS[call]:g} ms")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_yticks(YTICKS[call], [f"{t:g}" for t in YTICKS[call]])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("concurrent clients (closed loop)")
        ax.set_ylabel("client latency (ms)")
        ax.set_title(f"POST /{call}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Service latency in docker compose on the laptop: encoder_t<ONNX threads>[_w<uvicorn workers>]",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bench.png", dpi=150)
    plt.close(fig)


async def run(url: str) -> dict:
    reqs = requests()
    raise_open_files()
    client = Client(url)
    try:
        health = (await client.get("/healthz")).raise_for_status().json()
        lat, bounds = await latency(client, reqs)
        conc = await burst(client, reqs, bounds)
    finally:
        await client.aclose()
    gate = {call: {"p99_ms": lat[call]["1"]["client_ms"]["p99"], "target_ms": TARGET_MS[call],
                   "met": lat[call]["1"]["client_ms"]["p99"] <= TARGET_MS[call]} for call in TARGET_MS}
    return {"url": url, "service": health, "split": SPLIT, "prompts": len(reqs), "seed": SEED, "warmup": WARMUP,
            "max_tokens": MAX_TOKENS, "gate_one_client": gate, "latency": lat, "concurrency": conc,
            "client": {"python": sys.version.split()[0], "kind": "asyncio streams, one keep-alive connection per worker"}}


def main(argv: list[str]) -> None:
    """Benchmarks the running service; saves the run under its configuration (encoder, threads) in
    docs/figures/bench_runs/, and as bench.json when it is the default configuration (int8, 2 threads)."""
    url = argv[0] if argv else URL
    result = asyncio.run(run(url))
    name = config_name(result["service"])
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if name == DEFAULT_CONFIG:
        (FIG_DIR / "bench.json").write_text(json.dumps(result, indent=2) + "\n")
    plot({p.stem: json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))})
    print(json.dumps({"config": name, "gate_one_client": result["gate_one_client"],
                      "concurrency_passed": result["concurrency"]["passed"]}, indent=2))
    print(f"wrote {(RUNS_DIR / f'{name}.json').relative_to(ROOT)} and bench.png")
    if not result["concurrency"]["passed"]:
        raise SystemExit("concurrency check failed: a window went over budget, a 429 was wrongful, "
                         "or the counter does not match the admitted reservations")


if __name__ == "__main__":
    main(sys.argv[1:])
