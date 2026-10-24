# What we did not write ourselves

## Libraries

All open source, installed from PyPI at the versions pinned in
`requirements.txt`:

| Library | Licence | Used for |
|---|---|---|
| pandas | BSD 3-Clause | Table manipulation and grouped aggregation |
| numpy | BSD 3-Clause | Numeric arrays and window arithmetic |
| pyarrow | Apache 2.0 | Reading and writing Parquet |
| scikit-learn | BSD 3-Clause | Stratified cross-validation splits, average precision, precision/recall |
| LightGBM | MIT | Gradient-boosted tree classifier |
| networkx | BSD 3-Clause | Account-graph centralities, communities and cycle enumeration |
| scipy | BSD 3-Clause | Transitive dependency of scikit-learn |

## Organiser-provided material

- The dataset itself, and the field documentation in `data/raw/README.md`.
- `payments.py` from the starter pack. Our `src/dd/loading.py` is an
  independent reimplementation of the same idea — decode the JSON amount,
  value it at the hourly reference price — written as vectorised pandas and
  extended with `SendMax` valuation, fee valuation and conversion metadata.
  The decoding convention and the hourly-bucket rule come from the starter
  code and the field documentation.

## Pre-trained models and outside services

None. No pre-trained model, no external data source, no network call at run
time, and no external labels. The detector is fitted from scratch on the
supplied training split every time the pipeline runs.

## AI assistance

An AI coding assistant (Cursor, Claude) was used to help write and review the
pipeline source in this repository: the feature engineering in
`src/dd/features.py`, the model and evaluation code in `src/dd/model.py`, the
loader in `src/dd/loading.py`, the tests, and this documentation. Every
feature definition, the validation design and the leakage safeguards were
reviewed and run by the team against the training labels before submission.
No AI-generated content was used as data, as labels, or as a source of
information about the dataset beyond the organiser documentation.
