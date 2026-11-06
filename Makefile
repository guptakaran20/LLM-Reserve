PYTHON ?= python3.12
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup lock data explore baselines features models mlp eval sim sweep mispredict test clean-data

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

test: setup
	$(PY) -m pytest -q

clean-data:
	rm -rf data/raw data/processed data/features data/predictions data/lodo
