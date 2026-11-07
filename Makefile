PYTHON ?= python3.12
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup lock data explore baselines features models mlp train eval sim sweep mispredict onnx int8-served serve demo bench down test clean-data demo-assets app app-test

# Create the virtualenv from the pinned requirements.
setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
	touch $@

# Re-pin requirements.txt after editing requirements.in.
lock: setup
	$(PY) -m pip install -r requirements.in
	$(PY) -m pip freeze --exclude-editable > requirements.txt
	@# Linux xgboost wheels pull in NCCL (CUDA); CI and Docker get the CPU-only build of the same version.
	awk '/^xgboost==/ { v = substr($$0, 10); print "xgboost==" v "; sys_platform == \"darwin\""; \
		print "xgboost-cpu==" v "; sys_platform == \"linux\""; next } 1' requirements.txt > requirements.tmp
	mv requirements.tmp requirements.txt

# Download CASTILLO (cached in data/raw) and rebuild data/processed.
data: setup
	rm -rf data/processed
	$(PY) -m llm_reserve.data

# Day-2 plots and stats from the train split -> docs/figures/.
explore: setup
	$(PY) -m llm_reserve.explore

# Day-3 baseline table on validation at matched 95% coverage, plus the 80-99% curve -> docs/figures/.
baselines: setup
	$(PY) -m llm_reserve.baselines

# Day-4 feature matrix for all splits, plus cached MiniLM embeddings (128 and 256 tokens) -> data/features/.
features: setup
	$(PY) -m llm_reserve.features

# Day-5 linear and XGBoost quantile models, tuned to 95% coverage on validation -> docs/figures/ and data/models/.
models: setup
	$(PY) -m llm_reserve.models

# Day-6 PyTorch MLP (pinball loss), its configuration sweep and the winning model, on validation -> docs/figures/, data/models/.
mlp: setup
	$(PY) -m llm_reserve.mlp

# Everything the frozen winner needs, in order: features, the Day-5 models (XGBoost uses every core: OMP_NUM_THREADS=4
# caps it) and the Day-6 MLP. About 11 minutes on an M5 laptop at 4 threads.
train: features models mlp

# Day-7 test split, scored once with every choice frozen: results, breakdowns, error analysis and
# leave-one-dataset-out -> docs/figures/. XGBoost and torch run in separate processes (findings 20).
eval: setup
	$(PY) -m llm_reserve.predict xgb
	$(PY) -m llm_reserve.predict mlp
	$(PY) -m llm_reserve.lodo xgb
	$(PY) -m llm_reserve.lodo mlp
	$(PY) -m llm_reserve.holdout

# Day-8 budget-gateway simulator on the test split (original config): the max_tokens policy end to end,
# 3 seeds -> docs/figures/sim_max_tokens.json and sim_windows.png.
sim: setup
	$(PY) -m llm_reserve.simulate

# Day-9 policies (max_tokens, fixed percentiles, the frozen winner, the oracle) swept over arrival rate,
# budget and decode speed, 3 seeds each -> docs/figures/sim_policies.json and sim_policies.png.
sweep: setup
	nice $(PY) -m llm_reserve.sweep

# Day-10 misprediction strategies on the bar and the winner: A caps max_tokens at bound x margin, B holds back
# a share of each window, sized on validation -> docs/figures/mispredict.json and mispredict.png.
mispredict: setup
	nice $(PY) -m llm_reserve.mispredict

# Day-11 ONNX bundle of the frozen winner (MiniLM-128 encoder, MLP, XGBoost) -> data/models/onnx/, and the
# parity check on 1,000 test prompts with ONNX Runtime alone -> docs/figures/onnx_parity.json. Needs `make eval`.
# Then the Day-13 int8 encoder and its coverage check on validation and test -> docs/figures/int8_check.json.
onnx: setup
	nice $(PY) -m llm_reserve.export torch
	nice $(PY) -m llm_reserve.export xgb
	nice $(PY) -m llm_reserve.export check
	nice $(PY) -m llm_reserve.quantize

# Day-13 int8 coverage check on the embeddings the service image itself computes: ONNX Runtime's int8 kernels
# on Linux arm64 differ from macOS's (findings 44). Needs `make onnx`; adds int8_served to int8_check.json.
int8-served: setup
	$(PY) -m llm_reserve.quantize texts
	docker compose run --build --rm --no-deps -v "$(CURDIR)/data/onnx_parity:/app/check" api python -m llm_reserve.quantize embed /app/check
	nice $(PY) -m llm_reserve.quantize served

# Day-12 service: the API (FastAPI + ONNX Runtime) and Redis in docker compose on http://localhost:8000.
# Needs `make onnx` (the image copies data/models/onnx/). `make demo` runs predict -> reserve -> settle with curl.
serve:
	docker compose up --build --detach --wait

demo:
	./scripts/demo.sh

# Day-13 load test and latency benchmark against the running service (`make serve` first) -> docs/figures/bench*.
# Saves the run under the service's configuration; ENCODER, PREDICT_THREADS and WEB_CONCURRENCY pick it.
bench: setup
	nice $(PY) -m llm_reserve.bench

down:
	docker compose down

# Day-15 Streamlit demo. `demo-assets` packs the int8 ONNX bundle, the per-model bar and the simulator results into app/
# (needs `make data onnx sweep mispredict eval`); `app` runs the page from its own small venv, the same dependencies a
# free host installs (app/requirements.txt: no torch or xgboost); `app-test` runs the page headless in that venv.
demo-assets: setup
	$(PY) -m llm_reserve.demo_assets

.venv-app/.installed: app/requirements.txt
	$(PYTHON) -m venv .venv-app
	.venv-app/bin/python -m pip install -r app/requirements.txt pytest
	touch $@

app: .venv-app/.installed
	OMP_NUM_THREADS=2 .venv-app/bin/streamlit run app/streamlit_app.py

app-test: .venv-app/.installed
	OMP_NUM_THREADS=2 PYTHONPATH=src .venv-app/bin/python -m pytest tests/test_demo.py --noconftest -q

test: setup
	$(PY) -m pytest -q

clean-data:
	rm -rf data/raw data/processed data/features data/predictions data/lodo data/onnx_parity
