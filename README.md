# Decode & Detect — Bath Bytes

Suspicion scoring for the UK CBT *Decode & Detect* XRPL hackathon (24 October 2026).
Score = 100 × Average Precision on 40,101 held-out test transactions.

This repository currently holds **Person 2's deliverable: wallet-graph features**.

## Quick start

```bash
pip install -r requirements.txt

# 1. Build the features (pass train AND test transfers so the graph is complete)
python src/graph_features.py \
    --transfers data/raw/train/transfers.parquet data/raw/test/transfers.parquet \
    --transactions data/raw/train/transactions.parquet data/raw/test/transactions.parquet \
    --out outputs/graph_features.csv

# 2. Validate the contract before handing the file to the ML engineer
python src/check_graph_features.py \
    --features outputs/graph_features.csv \
    --transactions data/raw/train/transactions.parquet data/raw/test/transactions.parquet
```

No organiser data is committed — see `.gitignore`. To develop without it:

```bash
python src/make_sample_data.py --out-dir data/sample   # synthetic stand-in
python -m pytest tests -q
```

## Layout

| Path | What it is |
| --- | --- |
| `src/graph_features.py` | Builds the directed wallet graph and writes `graph_features.csv` |
| `src/check_graph_features.py` | Contract check: one row per hash, no NaNs, no label leakage |
| `src/make_sample_data.py` | Synthetic transfers with implanted fan-in/out, scatter-gather, cycles |
| `src/sample_signal_demo.py` | Proves the features separate the implanted patterns |
| `tests/` | Pattern-definition tests and the end-to-end contract test |
| `docs/person2_graph_features.md` | Feature dictionary, handover notes, team workflow |
| `outputs/` | Generated feature tables (gitignored; commit the `.csv.gz` instead) |

## Output contract

`outputs/graph_features.csv` has exactly one row per `TRANSACTION_HASH`, all numeric
columns, no missing values, and no labels. Person 3 joins it on `TRANSACTION_HASH`.
