# clientco-bcg
B2B retail case study (CentraleSupélec × BCG): customer churn prevention for ClientCo, an international construction materials distributor. Using two years of transaction data, the project models attrition risk, flags at-risk clients, and outlines an activation roadmap to protect revenue and prove the value of advanced analytics.


## Structure

- `churn_exploration_viz.ipynb`: visual exploration of the dataset for the churn-prevention use case (data quality, client behaviour, churn definition, value at stake, early-warning signals, baseline score). Run top to bottom; outputs are embedded.
- `exploratory_and_data_update.ipynb`: initial exploratory analysis and data update of the sales dataset.
- `docs/subject/`: case brief, interviews and financials (not versioned, kept locally).
- `dataset_sales_2026.csv`: raw transaction data (~4 GB, not versioned; place it in `data/` or at the repo root).

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name clientco-bcg
```

Then open the notebook and select the `.venv` / `clientco-bcg` kernel.

The first run of `churn_exploration_viz.ipynb` converts the 4 GB CSV into `data/sales.parquet` (~1 min, not versioned); later runs reuse it. All heavy aggregation is done in DuckDB, so ~8 GB of RAM is enough.

## Requirements

`pandas`, `matplotlib`, `duckdb` (notebook dependencies) and `ipykernel` (to run the notebook). See [requirements.txt](requirements.txt).
