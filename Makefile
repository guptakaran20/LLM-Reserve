PYTHON ?= python3.12
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup lock data test clean-data

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

# Download CASTILLO (cached in data/raw) and rebuild data/processed.
data: setup
	rm -rf data/processed
	$(PY) -m llm_reserve.data

test: setup
	$(PY) -m pytest -q

clean-data:
	rm -rf data/raw data/processed
