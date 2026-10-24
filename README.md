# Decode & Detect — UK CBT

A transaction-level detector for the Decode & Detect synthetic ledger. It gives
every payment a suspicion score in `[0, 1]` so the laundering motifs planted in
the data rise to the top of the ranking.

## The exact command

Install the dependencies, then run the pipeline against the training split and
the split you want to score:

```sh
pip install -r requirements.txt
PYTHONPATH=src python -m dd.pipeline \
    --train data_raw/train \
    --test  data_raw/test \
    --out   submission/predictions.csv \
    --cache cache \
    --report artifacts/metrics.json
```

`--train` must contain `transactions.parquet`, `transfers.parquet`,
`blocks.parquet`, `market_rates.csv` and `labels.csv`. `--test` needs the same
files except `labels.csv`. The command writes `submission/predictions.csv`
with one `TRANSACTION_HASH,score` row per test transaction, and a JSON metrics
report alongside it.

**Hardware and time.** It runs on a 4-core machine with 8 GB of RAM in a few
minutes end to end: roughly 40 seconds to build the features for each split and
the rest in cross-validated model fitting. No GPU is used. Every random seed is
fixed (`dd.model.SEED = 20261024`), so a rerun reproduces the same file.

## How it works

The ledger is projected into one effective payment per transaction — sender,
destination, and the delivered amount valued in reference USD at the hourly
rate for the asset actually delivered. Everything else is built on top of that
projection, then a gradient-boosted tree ensemble learns the ranking.

### Feature groups

| Group | What it captures |
|---|---|
| `amount` | Size, roundness, trailing zeros, distance below a round threshold, time of day |
| `conversion` | Direct vs converted vs XRP-bridged, leg count, assets touched, the `SendMax` spread |
| `account` | Whole-window profile of both parties: volume, degree, flow-through ratio, burstiness |
| `pair` | What this specific sender→receiver relationship normally looks like, and whether it is new |
| `velocity` | Payment and counterparty counts in ±10 m, ±1 h, ±6 h and ±24 h windows around each payment |
| `flow` | Layering evidence — value arriving just before, a near-identical amount leaving just after |
| `structure` | Repeated near-identical amounts from one account, the smurfing and stacking signature |
| `graph` | PageRank, HITS, betweenness, k-core, communities, short-cycle membership, shared neighbours |

The `flow` group is the one that does most of the work. Every multi-hop motif
in this dataset — chain, stack, multi-stage, relay — leaves the same trace: an
amount lands on an account and a near-identical amount leaves it a short time
later. Measuring that match in log space makes it symmetric and scale-free, so
a £200 hop and a £200,000 hop score the same way.

Each motif family in the training labels contains both suspicious transactions
and legitimate look-alikes, so shape alone cannot separate them. The features
are built to describe *context* — how unusual this payment is for these two
accounts, at this moment, in this part of the graph — rather than to match a
fixed pattern.

### Model

LightGBM binary classifier, repeated stratified 5-fold cross-validation with 3
repeats, early stopping on average precision. Test scores are the **mean
percentile rank** across all fold models rather than the mean probability: the
fold models stop at different iterations and are not calibrated against each
other, and average precision depends only on the ordering.

### Leakage safeguards

- Labels are read in exactly one place, `dd.loading.load_labels`, and never
  reach feature construction.
- Features for a split are computed from that split alone. Wallets and hashes
  are disjoint between splits, so train and test are profiled identically and
  independently; nothing about which split a row came from is a feature.
- `scenario_id`, `cohort_id`, `family` and `difficulty` are annotation metadata
  that do not exist in the test data, so they are never used for training.
- The feature table is asserted to have exactly one row per `TRANSACTION_HASH`,
  and the submission is asserted to be unique, complete and within `[0, 1]`.

## Results

See `artifacts/metrics.json` for the out-of-fold average precision, the
equivalent leaderboard score, and the top-1% precision and recall computed with
the organisers' hash-ascending tie-break. `docs/methodology.md` records the
method in full.

## Layout

```
src/dd/loading.py    split loading, JSON amount decoding, USD valuation
src/dd/features.py   all eight feature groups
src/dd/model.py      cross-validation, blending, top-k metrics
src/dd/pipeline.py   command-line entry point
tests/               correctness tests for valuation, features and submission shape
docs/methodology.md  method writeup
docs/ai_usage.md     what is not our own work
```

## Not our own work

See `docs/ai_usage.md`.
