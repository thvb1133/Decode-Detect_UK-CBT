"""Generate synthetic XRPL-shaped transfer data with implanted AML patterns.

The real competition files are not committed to this repository, so this script
produces stand-in ``transactions.parquet`` / ``transfers.parquet`` files with the
same shape. It lets the graph pipeline be developed and tested before the real
data lands, and the implanted patterns give the tests a known ground truth.

    python src/make_sample_data.py --out-dir data/sample --n-background 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 7


def _hash(idx: int) -> str:
    return f"TX{idx:08d}"


def generate(n_background: int = 4000, n_wallets: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    wallets = [f"rWallet{i:05d}" for i in range(n_wallets)]
    rows: list[dict] = []
    counter = 0

    def add(sender: str, receiver: str, amount: float, pattern: str) -> None:
        nonlocal counter
        rows.append(
            {
                "TRANSACTION_HASH": _hash(counter),
                "SENDER": sender,
                "RECEIVER": receiver,
                "AMOUNT_USD": round(float(amount), 2),
                "pattern": pattern,
            }
        )
        counter += 1

    # Every amount below is drawn from the same distribution, so the implanted
    # patterns are distinguishable only by graph structure.
    for _ in range(n_background):
        sender, receiver = rng.choice(wallets, size=2, replace=False)
        add(sender, receiver, rng.lognormal(5, 1), "background")

    # Fan-in: 12 mules each pay a single collector wallet.
    for group in range(5):
        collector = f"rFanInHub{group}"
        for mule in range(12):
            add(f"rFanInMule{group}_{mule}", collector, rng.lognormal(5, 1), "fan_in")

    # Fan-out: one source sprays 12 destinations.
    for group in range(5):
        source = f"rFanOutHub{group}"
        for dest in range(12):
            add(source, f"rFanOutLeaf{group}_{dest}", rng.lognormal(5, 1), "fan_out")

    # Scatter-gather: source -> 8 intermediaries -> single sink.
    for group in range(4):
        source = f"rScatterSrc{group}"
        sink = f"rScatterSink{group}"
        for hop in range(8):
            mid = f"rScatterMid{group}_{hop}"
            add(source, mid, rng.lognormal(5, 1), "scatter_gather")
            add(mid, sink, rng.lognormal(5, 1), "scatter_gather")

    # Gather-scatter: 8 sources -> single hub -> 8 destinations.
    for group in range(4):
        hub = f"rGatherHub{group}"
        for hop in range(8):
            add(f"rGatherSrc{group}_{hop}", hub, rng.lognormal(5, 1), "gather_scatter")
            add(hub, f"rGatherDst{group}_{hop}", rng.lognormal(5, 1), "gather_scatter")

    # Reciprocal round-trips between wallet pairs.
    for group in range(10):
        a, b = f"rPing{group}", f"rPong{group}"
        add(a, b, rng.lognormal(5, 1), "reciprocal")
        add(b, a, rng.lognormal(5, 1), "reciprocal")

    # Circular flows of length 3 and 4.
    for group in range(6):
        ring = [f"rCycle{group}_{i}" for i in range(3 + group % 2)]
        for i, node in enumerate(ring):
            add(node, ring[(i + 1) % len(ring)], rng.lognormal(5, 1), "cycle")

    transfers = pd.DataFrame(rows)

    # Roughly 10% of transactions carry a second transfer leg, mirroring
    # multi-path payments in the real data.
    multi = transfers.sample(frac=0.1, random_state=RNG_SEED).copy()
    multi["RECEIVER"] = rng.choice(wallets, size=len(multi))
    multi["AMOUNT_USD"] = (multi["AMOUNT_USD"] * 0.1).round(2)
    transfers = pd.concat([transfers, multi], ignore_index=True)

    transactions = (
        transfers.groupby("TRANSACTION_HASH", as_index=False)
        .agg(AMOUNT_USD=("AMOUNT_USD", "sum"), pattern=("pattern", "first"))
        .assign(
            TRANSACTION_TYPE="Payment",
            FEE_DROPS=12,
            LEDGER_INDEX=lambda d: np.arange(len(d)) + 80_000_000,
        )
    )
    return transactions, transfers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--n-background", type=int, default=4000)
    args = parser.parse_args()

    transactions, transfers = generate(args.n_background)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(args.out_dir / "transactions.parquet", index=False)
    transfers.to_parquet(args.out_dir / "transfers.parquet", index=False)
    print(f"transactions: {len(transactions):,} rows -> {args.out_dir/'transactions.parquet'}")
    print(f"transfers:    {len(transfers):,} rows -> {args.out_dir/'transfers.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
