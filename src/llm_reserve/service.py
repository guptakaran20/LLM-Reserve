"""Day-12 service: the spend gateway's three calls, over the ONNX predictor and atomic Redis windows.

    POST /predict  prompt, model, input_tokens[, max_tokens] -> 95% output bound and the cap to send
    POST /reserve  input_tokens, max_tokens[, bound]        -> reservation id, or 429 if the window is full
    POST /settle   reservation_id, input_tokens, output_tokens -> the window is charged the real cost

The gateway calls /predict, then /reserve with the bound, forwards the LLM call with the `max_tokens`
that /reserve returns, and settles with the provider's reported usage. /reserve does no inference, so
it stays inside its 5 ms target; /predict carries the 15 ms one.

Budget model (the same as the Day-8 simulator):

- Fixed windows of WINDOW_SECONDS (default 60 s) with a budget of BUDGET_TOKENS (default 200,000) each.
  A window's counter holds the reservations in flight plus the real cost of the settled ones.
- A request is billed to the window that admitted it, even if it settles in the next one: the window
  index is part of the reservation id, so /settle touches the same counter without a lookup.
- Strategy A (Day 10, the default): a request with a bound reserves and is sent with
  max_tokens = min(ceil(bound × CAP_MARGIN), max_tokens), so it can never cost more than it reserved
  and a window never overshoots. Without a bound it reserves its full max_tokens (the baseline policy).

Both Redis steps are Lua scripts, so each is one atomic step under any number of concurrent callers:
reserve checks and increments in one go (no check-then-act race), and settle applies
INCRBY (actual − reserved) and deletes the reservation together, so a repeated settle is refused
instead of applied twice. Keys: `{prefix}:{index}:window` and `{prefix}:{index}:resv:{id}`, the window
index a hash tag so both keys of a script share a Redis Cluster slot.

Run it with `docker compose up` (API and Redis), or locally with `python -m llm_reserve.service`.
"""

from __future__ import annotations

import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_reserve.serving import BUNDLE_DIR

CAP_MARGIN = 1.5  # Day-10 decision: strategy A's margin, chosen on validation (docs/figures/mispredict.json)
MAX_TOKENS = 4096  # a request's max_tokens when it gives none (the simulator's default)

# KEYS[1] window counter, KEYS[2] reservation. ARGV: cost, budget, window ttl (s), reservation ttl (s).
# Returns {1, spent} when admitted, {0, spent} when the cost does not fit, {-1, spent} for a reused id.
RESERVE_LUA = """
local cost = tonumber(ARGV[1])
local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
if spent + cost > tonumber(ARGV[2]) then
  return {0, spent}
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return {-1, spent}
end
spent = redis.call('INCRBY', KEYS[1], cost)
if redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
end
redis.call('SET', KEYS[2], cost, 'EX', tonumber(ARGV[4]))
return {1, spent}
"""

# KEYS[1] window counter, KEYS[2] reservation. ARGV: actual cost.
# Returns {0, 0, 0} for an unknown (never made, expired or already settled) reservation, else
# {1, reserved, spent}, with spent = -1 when the window has expired and there is nothing left to adjust.
SETTLE_LUA = """
local reserved = redis.call('GET', KEYS[2])
if not reserved then
  return {0, 0, 0}
end
reserved = tonumber(reserved)
redis.call('DEL', KEYS[2])
local spent = -1
if redis.call('EXISTS', KEYS[1]) == 1 then
  spent = redis.call('INCRBY', KEYS[1], tonumber(ARGV[1]) - reserved)
end
return {1, reserved, spent}
"""

RESERVATION_ID = re.compile(r"^(\d+)-[0-9a-f]{32}$")


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    budget: int = 200_000  # tokens per window
    window_seconds: int = 60
    cap_margin: float = CAP_MARGIN
    default_max_tokens: int = MAX_TOKENS
    reservation_ttl: int = 3600  # an unsettled reservation is forgotten after this (it stays charged)
    # Day 13 (findings 47): a bounded pool that makes a request wait for a free connection. redis-py's default
    # pool raised "Too many connections" (a 500) for half of 1,000 simultaneous reserves.
    redis_max_connections: int = 64
    redis_pool_timeout: float = 5.0  # seconds a request waits for a connection before failing
    prefix: str = "llm-reserve"
    bundle: Path = field(default=BUNDLE_DIR)
    encoder: str = "int8"  # Day 13: the per-channel int8 MiniLM (findings 42-44); "fp32" is the Day-11 export
    predict_threads: int = 2  # ONNX Runtime intra-op threads per session (findings 45)

    @classmethod
    def from_env(cls) -> Settings:
        env, d = os.environ, cls()
        return cls(
            redis_url=env.get("REDIS_URL", d.redis_url),
            budget=int(env.get("BUDGET_TOKENS", d.budget)),
            window_seconds=int(env.get("WINDOW_SECONDS", d.window_seconds)),
            cap_margin=float(env.get("CAP_MARGIN", d.cap_margin)),
            default_max_tokens=int(env.get("DEFAULT_MAX_TOKENS", d.default_max_tokens)),
            reservation_ttl=int(env.get("RESERVATION_TTL", d.reservation_ttl)),
            redis_max_connections=int(env.get("REDIS_MAX_CONNECTIONS", d.redis_max_connections)),
            redis_pool_timeout=float(env.get("REDIS_POOL_TIMEOUT", d.redis_pool_timeout)),
            prefix=env.get("KEY_PREFIX", d.prefix),
            bundle=Path(env.get("BUNDLE_DIR", d.bundle)),
            encoder=env.get("ENCODER", d.encoder),
            predict_threads=int(env.get("PREDICT_THREADS", d.predict_threads)),
        )

    # The window index is a hash tag ({...}), so a script's two keys share one Redis Cluster slot.
    def window_key(self, window: int) -> str:
        return f"{self.prefix}:{{{window}}}:window"

    def reservation_key(self, window: int, reservation_id: str) -> str:
        return f"{self.prefix}:{{{window}}}:resv:{reservation_id}"


def capped_output(bound: int | None, max_tokens: int, margin: float = CAP_MARGIN) -> int:
    """Output tokens to reserve and send as max_tokens: strategy A's ceil(bound × margin), at most max_tokens.

    The same rule as mispredict.capped; without a bound, the request's own max_tokens.
    """
    if bound is None:
        return max_tokens
    return max(1, min(math.ceil(bound * margin), max_tokens))


def canonical_model(name: str) -> str:
    """Accept Hugging Face ids ("meta-llama/Llama-3.1-8B-Instruct") as well as CASTILLO's "org--name"."""
    return name.replace("/", "--")


class PredictIn(BaseModel):
    prompt: str
    model: str
    input_tokens: int = Field(ge=0, description="prompt length in the target model's tokens")
    max_tokens: int | None = Field(None, ge=1)


class PredictOut(BaseModel):
    model: str
    bound: int  # 95% output-token bound
    max_tokens: int  # send this to the provider: the strategy-A cap
    reserve_tokens: int  # what /reserve will hold: input_tokens + max_tokens
    log_pred: float
    members: dict[str, float]


class ReserveIn(BaseModel):
    input_tokens: int = Field(ge=0)
    max_tokens: int | None = Field(None, ge=1)
    bound: int | None = Field(None, ge=1, description="from /predict; omit to reserve the full max_tokens")


class ReserveOut(BaseModel):
    reservation_id: str
    reserved: int  # tokens held in the window: input + output
    max_tokens: int  # send this to the provider
    window: int
    spent: int  # the window's counter after this reservation
    remaining: int


class SettleIn(BaseModel):
    reservation_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SettleOut(BaseModel):
    reservation_id: str
    reserved: int
    actual: int
    delta: int  # actual − reserved, applied to the window
    spent: int | None  # the window's counter after settling; None if the window has expired
    overshoot: int  # tokens the window is now over budget (0 under strategy A)


class ServerTiming:
    """Adds `Server-Timing: app;dur=<ms>` to every response: the time from the request reaching the app to
    its response headers, so the Day-13 benchmark can split model and Redis time from HTTP and Docker.

    A plain ASGI middleware (not BaseHTTPMiddleware), so it adds a clock read, not a task, per request.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0 = time.perf_counter()

        async def timed_send(message):
            if message["type"] == "http.response.start":
                dur = (time.perf_counter() - t0) * 1e3
                message = {**message, "headers": [*message.get("headers", []),
                                                   (b"server-timing", f"app;dur={dur:.3f}".encode())]}
            await send(message)

        await self.app(scope, receive, timed_send)


def redis_client(settings: Settings):
    """An async Redis client whose pool blocks (up to redis_pool_timeout) instead of failing when every
    one of its redis_max_connections connections is busy."""
    from redis.asyncio import BlockingConnectionPool, Redis

    pool = BlockingConnectionPool.from_url(settings.redis_url, max_connections=settings.redis_max_connections,
                                           timeout=settings.redis_pool_timeout)
    return Redis(connection_pool=pool)


def create_app(settings: Settings | None = None, predictor=None, redis=None, clock=time.time) -> FastAPI:
    """The FastAPI app. Tests pass a stub predictor, fakeredis and a clock; otherwise all come from the env."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_redis = redis is None
        if predictor is None:
            from llm_reserve.serving import Predictor

            app.state.predictor = Predictor(settings.bundle, threads=settings.predict_threads, encoder=settings.encoder)
        else:
            app.state.predictor = predictor
        if owns_redis:
            app.state.redis = redis_client(settings)
        else:
            app.state.redis = redis
        app.state.reserve = app.state.redis.register_script(RESERVE_LUA)
        app.state.settle = app.state.redis.register_script(SETTLE_LUA)
        yield
        if owns_redis:
            await app.state.redis.aclose()

    app = FastAPI(title="llm-reserve", version="0.1.0", lifespan=lifespan)
    app.add_middleware(ServerTiming)

    def window_now() -> int:
        return int(clock() // settings.window_seconds)

    def retry_after(window: int) -> int:
        return max(1, math.ceil((window + 1) * settings.window_seconds - clock()))

    @app.get("/healthz")
    async def healthz(request: Request) -> dict:
        await request.app.state.redis.ping()
        p = request.app.state.predictor
        return {"status": "ok", "winner": p.spec["winner"], "models": len(p.models), "budget": settings.budget,
                "window_seconds": settings.window_seconds, "cap_margin": settings.cap_margin,
                "encoder": getattr(p, "encoder_precision", None), "predict_threads": settings.predict_threads,
                "workers": int(os.environ.get("WEB_CONCURRENCY", 1))}

    @app.get("/budget")
    async def budget(request: Request) -> dict:
        window = window_now()
        spent = int(await request.app.state.redis.get(settings.window_key(window)) or 0)
        return {"window": window, "budget": settings.budget, "spent": spent,
                "remaining": settings.budget - spent, "resets_in": retry_after(window)}

    @app.post("/predict", response_model=PredictOut)
    def predict(body: PredictIn, request: Request) -> PredictOut:
        # A plain def: FastAPI runs it in its thread pool, so ~7 ms of inference never blocks the event loop.
        p = request.app.state.predictor
        model = canonical_model(body.model)
        if model not in p.models:
            raise HTTPException(422, f"unknown model {body.model!r}; the predictor knows {p.models}")
        pred = p.predict(body.prompt, model, body.input_tokens)
        cap = capped_output(pred.bound, body.max_tokens or settings.default_max_tokens, settings.cap_margin)
        return PredictOut(model=model, bound=pred.bound, max_tokens=cap, reserve_tokens=body.input_tokens + cap,
                          log_pred=pred.log_pred, members=pred.members)

    @app.post("/reserve", response_model=ReserveOut, responses={429: {"description": "window budget is full"}})
    async def reserve(body: ReserveIn, request: Request):
        output = capped_output(body.bound, body.max_tokens or settings.default_max_tokens, settings.cap_margin)
        cost = body.input_tokens + output
        window = window_now()
        rid = f"{window}-{uuid.uuid4().hex}"
        status, spent = await request.app.state.reserve(
            keys=[settings.window_key(window), settings.reservation_key(window, rid)],
            args=[cost, settings.budget, 2 * settings.window_seconds, settings.reservation_ttl])
        if status == -1:  # a 122-bit random id collided: vanishingly rare, but never double-book
            raise HTTPException(503, "reservation id collision, retry")
        if status == 0:
            wait = retry_after(window)
            return JSONResponse(status_code=429, headers={"Retry-After": str(wait)}, content={
                "detail": "token budget for this window is full", "requested": cost, "spent": int(spent),
                "budget": settings.budget, "retry_after": wait})
        return ReserveOut(reservation_id=rid, reserved=cost, max_tokens=output, window=window, spent=int(spent),
                          remaining=settings.budget - int(spent))

    @app.post("/settle", response_model=SettleOut)
    async def settle(body: SettleIn, request: Request) -> SettleOut:
        m = RESERVATION_ID.match(body.reservation_id)
        if not m:
            raise HTTPException(422, "malformed reservation_id")
        actual = body.input_tokens + body.output_tokens
        window = int(m.group(1))
        found, reserved, spent = await request.app.state.settle(
            keys=[settings.window_key(window), settings.reservation_key(window, body.reservation_id)],
            args=[actual])
        if not found:
            raise HTTPException(404, "unknown reservation: never made, expired or already settled")
        spent = None if spent == -1 else int(spent)
        return SettleOut(reservation_id=body.reservation_id, reserved=int(reserved), actual=actual,
                         delta=actual - int(reserved), spent=spent,
                         overshoot=0 if spent is None else max(0, spent - settings.budget))

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
