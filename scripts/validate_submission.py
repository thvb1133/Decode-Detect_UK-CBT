"""Check a predictions CSV against the organisers' stated submission rules.

Run before every upload: the site rejects a file with a missing, extra or
duplicated hash, and a rejected upload is cheaper to avoid than to debug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MAX_BYTES = 12 * 1024 * 1024
MAX_ROWS = 100_000


def validate(csv_path, test_dir):
    problems = []
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)

    if list(frame.columns[:2]) != ["TRANSACTION_HASH", "score"]:
        problems.append(f"columns must start TRANSACTION_HASH,score; found {list(frame.columns)}")
    if frame["TRANSACTION_HASH"].duplicated().any():
        problems.append(f"{int(frame['TRANSACTION_HASH'].duplicated().sum())} duplicate hashes")
    if not frame["score"].between(0, 1).all():
        problems.append("scores outside [0, 1]")
    if frame["score"].isna().any():
        problems.append("missing scores")
    if len(frame) > MAX_ROWS:
        problems.append(f"{len(frame)} rows exceeds the {MAX_ROWS} limit")

    size = csv_path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{size} bytes exceeds the {MAX_BYTES} byte limit")

    expected = set(pd.read_parquet(Path(test_dir) / "transactions.parquet")["TRANSACTION_HASH"])
    got = set(frame["TRANSACTION_HASH"])
    if got - expected:
        problems.append(f"{len(got - expected)} hashes not present in the test data")
    if expected - got:
        problems.append(f"{len(expected - got)} test transactions have no score")

    spread = frame["score"].nunique()
    print(f"rows                {len(frame)}")
    print(f"unique hashes       {frame['TRANSACTION_HASH'].nunique()}")
    print(f"distinct scores     {spread}")
    print(f"score range         [{frame['score'].min():.6f}, {frame['score'].max():.6f}]")
    print(f"file size           {size / 1024 / 1024:.2f} MiB (limit 12 MiB)")
    if spread < len(frame) / 100:
        problems.append("scores are heavily tied; AP groups ties at one threshold")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="submission/predictions.csv")
    parser.add_argument("--test", required=True)
    args = parser.parse_args(argv)
    problems = validate(args.csv, args.test)
    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASSED: submission satisfies every stated rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
