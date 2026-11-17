# clientco-bcg
B2B retail case study (CentraleSupélec × BCG): customer churn prevention for ClientCo, an international construction materials distributor. Using two years of transaction data, the project models attrition risk, flags at-risk clients, and outlines an activation roadmap to protect revenue and prove the value of advanced analytics.


## Structure

- `exploratory_and_data_update.ipynb`: exploratory analysis and data update of the sales dataset.
- `docs/`: case brief, interviews and financials.
- `dataset_sales_2026.csv`: raw transaction data (~4 GB, not versioned; place it at the repo root).

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name clientco-bcg
```

Then open the notebook and select the `.venv` / `clientco-bcg` kernel.

## Requirements

`pandas`, `matplotlib` (notebook dependencies) and `ipykernel` (to run the notebook). See [requirements.txt](requirements.txt).
