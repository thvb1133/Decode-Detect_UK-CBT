# Methodology

## 1. Data and preprocessing

Each split is loaded and processed in complete isolation. Wallets and
transaction hashes are disjoint between splits, so a profile built across
splits would be meaningless, and which split a row came from is explicitly not
a feature.

`src/dd/loading.py` projects the ledger into **one effective payment per
transaction**:

- `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT` are JSON text, so both are
  parsed per row. The destination comes from the former; the delivered asset
  and amount from the latter.
- XRP amounts arrive as a string of integer drops. Issued currencies arrive as
  an object and are scaled by 1e6, which keeps them exact at the six decimal
  places the dataset uses.
- An asset is identified by **both issuer and currency** (`issuer:CURRENCY`),
  never by the currency code alone. USD from one fictional issuer is a
  different asset from USD from the other, and conflating them would compare
  unlike money.
- Every amount is valued in reference USD using `market_rates.csv` at the hour
  containing the ledger timestamp. This gives a single comparable scale across
  XRP and both issuers' tokens.
- `SendMax` is valued the same way but kept in a **separate** column. It is the
  source spending ceiling, not the actual debit, so it is never substituted for
  the delivered amount. Its gap above the delivery is retained as a conversion
  cost signal.
- Fees are valued in USD at the contemporaneous XRP price.

Transfers are retained at leg level and used only for conversion structure:
leg count, distinct assets and tokens touched, intermediary hops, and whether
the payment was direct, a direct conversion or a three-leg XRP bridge.

## 2. Feature groups

252 features in eight groups, all label-free. The full definitions are in
`src/dd/features.py`.

**amount** — value in USD and in native units, fee and fee share, trailing
decimal zeros, roundness at several scales, fractional part, and the distance
below the next round threshold. Hour, minute, second, day of week, night and
weekend flags, and offset into the window.

**conversion** — leg count, distinct symbols and token addresses, hop count,
direct vs path payment, bridge flag, presence of a `Paths` field, the `SendMax`
spread, and whether the source asset differs from the delivered asset.

**account** — whole-window profile of sender and receiver: inbound and outbound
counts, sums, means, medians, standard deviations and maxima; in- and
out-degree; lifespan; rate per hour; coefficient of variation. Two derived
ratios matter most: `flow_through` (the smaller of inbound and outbound value
over the larger, which approaches 1 for a pure conduit) and `flow_imbalance`.
Each payment also carries its size relative to the parties' own norms — ratios
to mean, median and maximum, and z-scores.

**pair** — statistics of this specific sender→receiver edge: count, sum, mean,
spread, span, sequence position, whether it is the first payment on the edge,
time since the previous one, size relative to the previous one, whether the
reverse edge also exists, and the edge's share of each party's total volume.

**velocity** — counts, summed value and distinct counterparties for the sender's
outgoing and the receiver's incoming payments in ±10 minutes, ±1 hour, ±6 hours
and ±24 hours, plus inter-arrival gaps either side. Queried by binary search
over per-account time-sorted arrays, so cost is linear in the events touched.

**flow** — the layering group, and the most important one. For each payment of
amount `A` at time `t` from `s` to `d`, over 1-hour, 6-hour and 24-hour windows:

- *forward*: everything `d` sends after `t` — count, total, ratio to `A`, fan-out,
  latency to the first onward payment, how many land within 1% and 5% of `A`,
  and the closest match measured as `|log1p(B) - log1p(A)|`
- *backward*: everything `s` received before `t`, symmetrically

Matching in log space makes the measure symmetric in the direction of the
discrepancy and scale-free, so a small hop and a large hop are judged the same
way. A combined `chain_score` multiplies the backward and forward match
qualities: it peaks exactly when a payment is simultaneously the tail of one
hop and the head of the next, which is what chain, stack, multi-stage and relay
motifs all look like from the inside.

**structure** — within ±24 hours, how many of the account's other payments sit
within 5% of this amount in log space, the spread of those amounts, and this
payment's rank among them. Near-identical repeated amounts are the smurfing and
stacking signature.

**graph** — the directed account graph for the window, aggregated over edges.
It is small (hundreds of nodes), so exact computation is affordable: PageRank
in both directions, HITS hub and authority scores, betweenness, undirected
clustering, k-core number, strongly-connected-component membership and size,
and greedy-modularity communities. Edge-level: whether the reverse edge exists,
shared-neighbour count and Jaccard overlap (the bipartite motif), two-hop path
count, and membership of a short directed cycle. Cycle enumeration is bounded
to length 4 and capped, because a hub-and-spoke graph has combinatorially many
short cycles and the counts saturate long before enumeration would finish.

## 3. Why context rather than pattern matching

Cross-tabulating the training labels by family shows that **every motif family
contains both suspicious transactions and legitimate look-alikes** — for
example `fan_out_in` is 152 legitimate against 178 suspicious, and `cycle` is
113 against 78. The two large relay families are overwhelmingly legitimate
(around 10,000 rows each) with roughly 100 hard suspicious transactions hidden
inside.

So a detector that recognises motif shapes cannot work: the shapes are present
on both sides. The features are therefore built to describe *context* — how
unusual this payment is for these two accounts, at this moment, in this part of
the graph — rather than to match fixed patterns. This is also why the pair and
velocity groups rank so highly in the fitted importances.

## 4. Train/validation strategy and leakage safeguards

Repeated stratified k-fold cross-validation, 5 folds × 3 repeats, stratified on
the 3.1% positive rate, with early stopping on average precision against the
held-out fold. Out-of-fold predictions are averaged across repeats before
scoring, which stabilises the estimate at this positive rate.

Safeguards:

- Labels are read in exactly one function, `dd.loading.load_labels`, which is
  called only by the pipeline. Feature code has no access to them. A test
  (`test_features_never_see_labels`) writes a `labels.csv` into a split
  directory and asserts the feature table is byte-identical.
- `scenario_id`, `cohort_id`, `family` and `difficulty` are annotation metadata
  absent from the test data and are never used.
- Features are a function of one split only, so train and test are profiled by
  identical, independent computations.
- The feature table is asserted to have exactly one row per `TRANSACTION_HASH`.

Account- and graph-level aggregates do use the whole supplied window. The rules
permit this explicitly — it is an offline investigation, not real-time
detection — and because the aggregates are computed without labels, they carry
no label information between folds.

## 5. Models and results

LightGBM binary classifier. `scripts/blend_runs.py` additionally fits three
differently-shaped configurations (deep/permissive, shallow/heavily-subsampled,
balanced) and grid-searches non-negative blend weights **on out-of-fold
predictions only**, falling back to the best single configuration when the
blend does not beat it.

Test scores are the **mean percentile rank** across fold models, not the mean
probability. Fold models stop at different iterations and are not calibrated
against one another, so averaging raw probabilities lets whichever model
happens to be most confident dominate. Average precision depends only on the
ordering, so rank averaging is the blend that matches the metric.

Final scores are themselves converted to percentile ranks before writing. This
spreads them across `[0, 1]` and avoids the tie-grouping penalty in the
organisers' AP implementation.

Measured results are recorded in `artifacts/metrics.json`. The headline figures
from the submitted run are an out-of-fold average precision of **0.8526**
(leaderboard-equivalent score **85.26**) with **98.5% precision** and 31.6%
recall in the top 1% of the ranking, evaluated with the organisers'
hash-ascending tie-break.

### What limits the score

225 of the 1,250 positives are hard, and 25 of those are described by the
organisers as deliberately ambiguous ordinary payments whose labels are
independent of any observable feature. Those 25 are unreachable by any model;
they bound what is attainable. The remaining headroom is in the roughly 200
hard cases with planted historical patterns, buried in the two large relay
families, which need longer-range aggregate reasoning than the ±24-hour
windows used here.

## 6. Reproducibility

- Python 3.12.3, dependencies pinned in `requirements.txt`.
- Every seed derives from `dd.model.SEED = 20261024`, including per-fold
  bagging and feature-sampling seeds.
- One command reproduces the submitted CSV; see `README.md`.
- Runs on 4 cores and 8 GB of RAM in a few minutes. No GPU, no network access
  at run time, no external data.
- `scripts/validate_submission.py` checks the output against every stated
  submission rule before upload.
