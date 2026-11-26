# clientco-bcg
B2B retail case study (CentraleSupélec × BCG): customer churn prevention for ClientCo, an international construction materials distributor. Using two years of transaction data, the project models attrition risk, flags at-risk clients, and outlines an activation roadmap to protect revenue and prove the value of advanced analytics.


## Structure

```
configs/         experiment definitions (YAML): base.yaml, experiments/, sweeps/
src/churn/       the `churn` package
  cli.py         commands: ingest | run | sweep | collect | score
  config.py      typed configuration, YAML loading, run id, project paths
  data.py        ingestion (CSV -> shifted Parquet) and DuckDB access
  snapshots.py   snapshot tables: features, labels, segments, temporal splits
  models.py      registry (baseline, BG/NBD, logreg, HistGradientBoosting, XGBoost), calibration, tuning
  scoring.py     value at risk V, expected value (P x V), list size K, reasons, batch list
  evaluation.py  metrics, computed from saved predictions
  impact.py      impact in M EUR and tornado table
  comparison.py  paired comparison of runs by date, acceptance rule
  simulation.py  value definitions and monthly lists with a rest period
  pipeline.py    run, sweep, collect
tests/           anti-leakage and unit tests
notebooks/       numbered analysis notebooks 01-09 and experiment_report.ipynb (skeleton, not filled in)
front/           local replay dashboard of the call lists (static HTML)
docs/            documentation (French); the case brief itself lives in docs/subject/ (not versioned)
data/            raw and derived data (not versioned)
runs/            one folder per run (not versioned)
results/         tables merged by `churn collect`, read by the report notebook
```

Notebooks (French, run top to bottom, outputs are embedded):

| Notebook | Content |
|---|---|
| `01_data_quality_and_activity_overview` | Data quality, activity overview (channels, branches, products) |
| `02_clients_rhythm_and_churn_definition` | Who the clients are, purchase rhythm, what a lost client is, revenue at stake |
| `03_churn_drivers_early_signals_baseline` | Cohorts, where churn concentrates, early-warning signals, baseline score |
| `04_churn_scope_total_vs_partial` | Model scope: total vs partial churn, segments, regular clients on 180 days |
| `05_label_test_and_list_allocation` | Total vs partial label test, net margin, split of the list K |
| `06_label_horizon_calibration` | Training-label horizon: 41 rules, label A vs relative label B |
| `07_realized_revenue_loss` | Revenue actually lost by churned clients (haircut of the total-churn revenue) |
| `08_label_rolling_backtest` | Labels A and B in a rolling backtest (monthly retraining) |
| `09_returns_matching_by_product` | Matching returns to purchases by product, effect on purchase days |
| `experiment_report` | Skeleton only (TODO cells, never executed) |

The raw transaction data (`dataset_sales_2026.csv`, ~4 GB) is not versioned; place it in `data/` (`churn ingest` only reads it there).

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # exact, tested versions
pip install -e . --no-deps          # the `churn` package (both steps: make setup)
python -m ipykernel install --user --name clientco-bcg
```

Then open a notebook and select the `.venv` / `clientco-bcg` kernel.

`notebooks/01_data_quality_and_activity_overview.ipynb` converts the 4 GB CSV into `data/sales.parquet` on its first run (~1 min, not versioned); later runs reuse it. All heavy aggregation is done in DuckDB, so ~8 GB of RAM is enough.

## Usage

Every experiment is one YAML file (see `configs/base.yaml` for all options). Each run writes `runs/<run_id>/`; `collect` merges all runs into `results/`.

```bash
python -m churn ingest                                      # data/sales_shifted.parquet (+2 years, once)
python -m churn sweep configs/sweeps/final_backtest.yaml    # rolling backtest of the final configuration
python -m churn collect                                     # runs/ -> results/*.parquet
python -m churn collect --universe final                    # same, plus total-churn metrics for runs named final*
python -m churn score <run_id> --date 2026-02-22            # action list of 700 clients (CSV)
.venv/bin/python front/build.py                             # local dashboard: data/front/index.html (needs the final_backtest runs)
make test                                                   # anti-leakage and unit tests
```

**Final test guard.** Any run or list dated 2026-03-22 on the real data is refused unless `CHURN_FINAL_TEST=1`. `configs/base.yaml` points to that date, so `run configs/experiments/*.yaml`, `sweep configs/sweeps/models.yaml`, `make run` and `make sweep` (default arguments) are refused without it. The final configuration is `configs/experiments/final.yaml`.

The dashboard output holds client-level data: keep it local. Known gaps at the freeze: [docs/07_risks_and_open_questions.md](docs/07_risks_and_open_questions.md), section 7.

## Documentation

All documents are in French and live in `docs/`, numbered in reading order. Each fact has one home; the others link to it.

| Doc | Content |
|---|---|
| [01_start_here.md](docs/01_start_here.md) | The case, the mission, deadlines, where we are, team rules, reading order |
| [02_modeling_decisions.md](docs/02_modeling_decisions.md) | **Modelling decisions, current state (single source of truth)** |
| [03_evidence.md](docs/03_evidence.md) | The numbers behind each decision (lemmas 1 to 7, each tied to a notebook) |
| [04_data_findings.md](docs/04_data_findings.md) | What the data says: concentration, rhythm, signals, seasonality, anomalies |
| [05_impact_and_pilot.md](docs/05_impact_and_pilot.md) | Impact in M EUR, retention scenarios, plans A and B, pilot |
| [06_engineering.md](docs/06_engineering.md) | Code architecture, commands, interface contracts, implementation order |
| [07_risks_and_open_questions.md](docs/07_risks_and_open_questions.md) | Vigilances, open questions, next steps |
| [08_decision_log.md](docs/08_decision_log.md) | Dated history of decisions and corrections |
| [09_presentation_notes.md](docs/09_presentation_notes.md) | Deck storyline, justification speech, likely questions |
| [glossary.md](docs/glossary.md) | Project vocabulary |

New to the project? Start with [docs/01_start_here.md](docs/01_start_here.md), then [docs/02_modeling_decisions.md](docs/02_modeling_decisions.md).
