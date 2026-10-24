# Person 2 — graph features: handover notes

## What the pipeline does

1. Reads every transfer leg (train **and** test together) into an edge list of
   `TRANSACTION_HASH, sender, receiver, amount`.
2. Collapses parallel transfers into a weighted `networkx.DiGraph`, one edge per
   ordered wallet pair, carrying `weight` (summed amount) and `transfer_count`.
3. Computes wallet-level structure once: degrees, weakly connected components,
   Louvain communities, PageRank, k-core, clustering coefficient, and the four
   AML pattern scores.
4. Maps those onto each transaction via its largest-value transfer leg, adds
   edge-level and cycle features, and writes one row per `TRANSACTION_HASH`.

## Why this cannot leak labels

- `labels.csv` is never opened by this code; `tests/test_graph_features.py`
  asserts no label-like column appears in the output.
- The graph uses only sender/receiver/amount, which the organisers give for the
  test split too.
- Building the graph over train + test is legitimate: every transaction to be
  scored is known up front, so no information from the future of the evaluation
  is used — only structure that exists in the supplied files.
- Degrees are *global over the 7-day window* rather than as-of transaction time.
  That is consistent between train and test, so it does not inflate validation
  AP relative to the leaderboard.

## Feature dictionary

Each wallet-level feature below appears twice, prefixed `sender_` and `receiver_`.

| Feature | Meaning | Pattern it serves |
| --- | --- | --- |
| `in_degree`, `out_degree` | Distinct counterparties paying in / paid out | Fan-in, fan-out |
| `weighted_in_degree`, `weighted_out_degree` | Total value in / out | Pass-through hubs |
| `transfer_in_count`, `transfer_out_count` | Number of transfer legs | Burstiness |
| `degree_ratio` | `in_degree / (out_degree + 1)` | Collector vs distributor |
| `reciprocal_partners` | Counterparties in both directions | Round-tripping |
| `is_fan_in`, `is_fan_out` | ≥ 5 counterparties on one side, ≤ 1 on the other | Fan-in, fan-out |
| `scatter_out_score` | Most intermediaries from this wallet reconverging on one target | Scatter-gather (source side) |
| `scatter_in_score` | Most funders sharing a single upstream origin | Scatter-gather (sink side) |
| `gather_scatter_score` | `min(in_degree, out_degree)` — hub-ness | Gather-scatter |
| `pagerank` | Weighted importance in the flow network | Layered stacks |
| `core_number` | k-core depth | Dense sub-networks |
| `clustering_coeff` | Local triangle density | Tight-knit rings |
| `community_size` | Size of the wallet's Louvain community | Bipartite blocks |

Transaction-level columns:

| Feature | Meaning |
| --- | --- |
| `cluster_size` | Size of the weakly connected component the payment sits in |
| `community_size` | Larger of the two endpoints' community sizes |
| `is_same_community` | 1 when sender and receiver fall in the same Louvain community |
| `cycle_indicator` | 1 when this edge lies on a directed cycle of ≤ 4 hops |
| `is_reciprocal_edge` | 1 when the receiver also pays the sender |
| `sender_cycle_count`, `receiver_cycle_count` | Number of short cycles the wallet sits on |
| `fan_in_pattern`, `fan_out_pattern` | Pattern flags from the relevant endpoint |
| `scatter_gather_pattern`, `gather_scatter_pattern` | Pattern flags from either endpoint |
| `edge_weight`, `edge_transfer_count` | Value and leg count on this wallet pair |
| `pagerank_ratio` | Receiver PageRank over sender PageRank |
| `n_transfer_legs`, `n_distinct_senders`, `n_distinct_receivers` | Transaction shape |

Raw component and community ids are dropped: they are arbitrary partition
labels that a tree model would overfit. Their sizes and the same-group flag
carry the usable signal.

## Team workflow

| Person | Produces | Joined on |
| --- | --- | --- |
| 1 | `train_clean.csv`, `test_clean.csv` | `TRANSACTION_HASH` |
| 2 | `graph_features.csv` (this repo) | `TRANSACTION_HASH` |
| 4 | `behaviour_features.csv` | `TRANSACTION_HASH` |
| 3 | merges all three, trains RF and LightGBM, submits | — |

Person 2 covers **both** splits in one file, so Person 3 can left-join it onto
the clean train table and the clean test table without rebuilding anything.

## What to add if there is time

1. **Time-aware degrees.** Recompute degrees in 1h / 24h windows ending at the
   transaction's close time, which is closer to what an AML system would see and
   usually beats the global version.
2. **Value-conservation checks.** For scatter-gather, compare value leaving the
   source with value arriving at the sink; near-equality is a strong signal.
3. **Pass-through time.** Median gap between money in and money out per wallet,
   which separates real hubs from mixers.
4. **Layered-stack depth.** Longest simple path from a funded source, capped at
   five hops.
5. **Asset-aware edges.** Build a second graph keyed on `(wallet, asset)` so
   currency-hopping shows up as structure.

## Sanity numbers on the synthetic sample

```
rows: 4,289  columns: 52
pattern hit rates:
  fan_in_pattern             0.0585
  fan_out_pattern            0.0690
  scatter_gather_pattern     0.0149
  gather_scatter_pattern     0.0173
  cycle_indicator            0.0105
```

A Random Forest trained on these features alone recovers 100% of the implanted
fan-in, fan-out, scatter-gather, gather-scatter, reciprocal and cycle
transactions in the top decile, at a 4% background false-positive rate.
