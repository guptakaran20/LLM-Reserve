# llm-reserve

Predict a tight 95th-percentile bound on an LLM request's output tokens, so a spend gateway can reserve what the request will likely cost instead of its full `max_tokens`.

Work in progress. See [docs/PRD.md](docs/PRD.md) for the plan and the data check, and [docs/findings.md](docs/findings.md) for results so far.

## Run

Requires Python 3.12.

```bash
make setup     # .venv from the pinned requirements.txt
make data      # download CASTILLO (~600 MB, cached in data/raw) and build data/processed
make explore   # Day-2 plots and stats from the train split -> docs/figures/
make baselines # Day-3 baseline table on validation at 95% coverage -> docs/figures/
make features  # Day-4 feature matrix + MiniLM embeddings (128 and 256 tokens) -> data/features/
make models    # Day-5 linear + XGBoost quantile models at 95% validation coverage -> docs/figures/, data/models/
make mlp       # Day-6 PyTorch MLP (pinball loss) and the winning model, on validation -> docs/figures/, data/models/
make eval      # Day-7 test split, scored once (frozen models), error analysis, leave-one-dataset-out -> docs/figures/
make sim       # Day-8 budget-gateway simulator, max_tokens policy end to end -> docs/figures/
make sweep     # Day-9 every policy over arrival rate, budget and decode speed, 3 seeds -> docs/figures/
make mispredict # Day-10 misprediction strategies (cap at bound x margin, overdraft buffer), knobs from validation -> docs/figures/
make onnx      # Day-11 ONNX bundle of the winner + Day-13 int8 encoder and its coverage check -> data/models/onnx/, docs/figures/
make serve     # Day-12 API + Redis in docker compose on http://localhost:8000 (needs `make onnx`)
make demo      # predict -> reserve -> settle with curl against the running service
make int8-served # Day-13 coverage check on the encoder embeddings the service image computes -> docs/figures/int8_check.json
make bench     # Day-13 latency benchmark and 1,000-request concurrency check against the running service -> docs/figures/
make down
make test
```

The service has three calls. A gateway asks `/predict` for the bound, reserves it, sends the LLM call with the `max_tokens` that `/reserve` returns, and settles with the provider's reported usage:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"prompt": "Explain in two sentences why the sky is blue.", "model": "meta-llama/Llama-3.1-8B-Instruct", "input_tokens": 20}'
curl -s localhost:8000/reserve -H 'content-type: application/json' \
  -d '{"input_tokens": 20, "max_tokens": 4096, "bound": 211}'   # 429 with Retry-After when the window is full
curl -s localhost:8000/settle -H 'content-type: application/json' \
  -d '{"reservation_id": "<from /reserve>", "input_tokens": 20, "output_tokens": 126}'
```

Settings come from the environment (`BUDGET_TOKENS`, default 200,000 per `WINDOW_SECONDS` = 60; `CAP_MARGIN`, default 1.5; `ENCODER`, `int8` or `fp32`; `PREDICT_THREADS`, default 2; `WEB_CONCURRENCY`, uvicorn workers, default 1; `REDIS_MAX_CONNECTIONS`, default 64).

Measured in docker compose on an M5 laptop (`make bench`, [findings](docs/findings.md#day-13-tests-and-benchmarks-5-nov-2026)), one client at a time: `/predict` p50 7.0 ms, p99 12.8 ms; `/reserve` p50 0.32 ms, p99 0.45 ms. Under concurrent load both queue on this laptop (`/predict` p99 21 ms at 4 clients; `/reserve` p99 3–8 ms at 16, 22 ms at 64). 1,000 parallel reserves never push a window past its budget.

Data: [CASTILLO](https://huggingface.co/datasets/danfperam/castillo) (CC BY 4.0), Perez-Ramirez, Kostic and Boman, [arXiv 2505.16881](https://arxiv.org/abs/2505.16881).
