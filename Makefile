.PHONY: setup test ingest run sweep collect

PY := .venv/bin/python
CONFIG ?= configs/experiments/baseline.yaml
SWEEP ?= configs/sweeps/models.yaml

setup:        ## Install the pinned versions (requirements.txt), then the package itself
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

test:         ## Run the test suite (leakage tests first)
	$(PY) -m pytest

ingest:       ## Build data/sales_shifted.parquet from the raw CSV
	$(PY) -m churn ingest

run:          ## Run one experiment: make run CONFIG=configs/experiments/hgb.yaml
	$(PY) -m churn run $(CONFIG)

sweep:        ## Run a series of experiments: make sweep SWEEP=configs/sweeps/models.yaml
	$(PY) -m churn sweep $(SWEEP)

collect:      ## Merge all runs/ into results/*.parquet
	$(PY) -m churn collect
