# Decode & Detect — Bath Bytes

One command builds the clean tables, the graph features, the behaviour features,
and the leaderboard file.

```bat
pip install -r requirements.txt
python src/run_all.py
```

Put the organiser files here first:

```
data/raw/train/transactions.parquet
data/raw/train/transfers.parquet
data/raw/train/blocks.parquet
data/raw/train/market_rates.csv
data/raw/train/labels.csv
data/raw/test/transactions.parquet
data/raw/test/transfers.parquet
data/raw/test/blocks.parquet
data/raw/test/market_rates.csv
```

Upload `submission/predictions.csv` at hackathon.ukcbt.org. It is two columns,
`TRANSACTION_HASH` and `score`, one row for every test transaction.

## What the command writes

Score = 100 × Average Precision on the 40,101 test transactions.

## Graph-only quick start

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
