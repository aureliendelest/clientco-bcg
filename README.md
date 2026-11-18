# clientco-bcg
B2B retail case study (CentraleSupélec × BCG): customer churn prevention for ClientCo, an international construction materials distributor. Using two years of transaction data, the project models attrition risk, flags at-risk clients, and outlines an activation roadmap to protect revenue and prove the value of advanced analytics.


## Structure

```
configs/         experiment definitions (YAML): base.yaml, experiments/, sweeps/
src/churn/       the `churn` package (skeleton: bricks are stubs to be filled in)
  cli.py         commands: ingest | run | sweep | collect | score
  config.py      typed configuration, YAML loading, run id, project paths
  data.py        ingestion (CSV -> shifted Parquet) and DuckDB access
  snapshots.py   snapshot tables: features, labels, segments, temporal splits
  models.py      registry (baseline, logreg, HistGradientBoosting, XGBoost), calibration, tuning
  scoring.py     value at risk V, haircut rho, expected value, capacity tiers, reasons, batch list
  evaluation.py  metrics (from saved predictions) and impact simulation
  pipeline.py    run, sweep, collect
tests/           anti-leakage and unit tests
notebooks/       exploration notebooks and report.ipynb (reads results/ only)
docs/            documentation (French); the case brief itself lives in docs/subject/ (not versioned)
data/            raw and derived data (not versioned)
runs/            one folder per run (not versioned)
results/         tables merged by `churn collect`, read by the report notebook
```

Notebooks:

- `notebooks/churn_exploration_viz.ipynb`: visual exploration of the dataset for the churn-prevention use case (data quality, client behaviour, churn definition, value at stake, early-warning signals, baseline score). Run top to bottom; outputs are embedded.
- `notebooks/exploratory_and_data_update.ipynb`: initial exploratory analysis and data update of the sales dataset.
- `notebooks/report.ipynb`: figures and tables of the experiments; reads `results/` only.

The raw transaction data (`dataset_sales_2026.csv`, ~4 GB) is not versioned; place it in `data/` or at the repo root.

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # exact, tested versions
pip install -e .                    # the `churn` package (or: make setup)
python -m ipykernel install --user --name clientco-bcg
```

Then open a notebook and select the `.venv` / `clientco-bcg` kernel.

`notebooks/churn_exploration_viz.ipynb` converts the 4 GB CSV into `data/sales.parquet` on its first run (~1 min, not versioned); later runs reuse it. All heavy aggregation is done in DuckDB, so ~8 GB of RAM is enough.

## Usage

Every experiment is one YAML file (see `configs/base.yaml` for all options). Each run writes `runs/<run_id>/`; `collect` merges all runs into `results/`.

```bash
python -m churn ingest                                   # data/sales_shifted.parquet (+2 years, once)
python -m churn run configs/experiments/hgb.yaml         # one experiment
python -m churn sweep configs/sweeps/models.yaml         # a series of experiments
python -m churn collect                                  # runs/ -> results/*.parquet
python -m churn score <run_id> --date 2026-03-22         # weekly action list (CSV)
make test                                                # anti-leakage and unit tests
```

The commands are in place but most bricks are still stubs: they exit with `not implemented yet` until filled in.

## Documentation

See [docs/churn_enjeux_decisions_vigilances.md](docs/churn_enjeux_decisions_vigilances.md) (French): issues, decisions, vigilances and open questions.
