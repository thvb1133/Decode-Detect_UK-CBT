"""Person 2 (Graph Analysis Engineer) - wallet graph features for AML detection.

Builds a directed wallet graph from transfer-level data and emits one row of
graph-derived features per TRANSACTION_HASH.

Only structural information is used: no labels, and labels.csv is never read by
this module. The graph is built once over train + test transfers, which is the
information available at prediction time (every transaction to be scored is
known up front), so no validation label ever leaks into a feature.

Usage
-----
    python src/graph_features.py \
        --transfers data/raw/transfers.parquet \
        --transactions data/raw/transactions.parquet \
        --out outputs/graph_features.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

# The organisers' column naming is not fixed across files, so each logical
# field is resolved from a list of candidates (first match wins, case-insensitive).
HASH_COLUMNS = ["TRANSACTION_HASH", "TX_HASH", "HASH", "TRANSACTIONHASH"]
SENDER_COLUMNS = [
    "SENDER",
    "FROM_ADDRESS",
    "FROM_WALLET",
    "FROM",
    "ACCOUNT",
    "SOURCE",
    "SOURCE_ADDRESS",
    "SENDER_ADDRESS",
]
RECEIVER_COLUMNS = [
    "RECEIVER",
    "TO_ADDRESS",
    "TO_WALLET",
    "TO",
    "DESTINATION",
    "DESTINATION_ADDRESS",
    "RECEIVER_ADDRESS",
]
AMOUNT_COLUMNS = [
    "AMOUNT_USD",
    "VALUE_USD",
    "USD_AMOUNT",
    "AMOUNT_XRP",
    "AMOUNT",
    "SOURCE_VALUE",
    "DESTINATION_VALUE",
    "VALUE",
    "DELIVERED_AMOUNT",
]

FAN_THRESHOLD = 5  # distinct counterparties before a wallet counts as fan-in/out


def resolve_column(df: pd.DataFrame, candidates: list[str], *, required: bool = True,
                   label: str = "") -> str | None:
    lookup = {c.upper(): c for c in df.columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    if required:
        raise KeyError(
            f"Could not find a {label or candidates[0]} column. "
            f"Tried {candidates}; available columns are {list(df.columns)}."
        )
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def read_tables(paths: list[Path]) -> pd.DataFrame:
    """Concatenate several files of the same shape (e.g. train + test transfers)."""
    frames = [read_table(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def load_edges(
    transfers_paths: list[Path], transactions_paths: list[Path] | None
) -> pd.DataFrame:
    """Return a tidy edge list: TRANSACTION_HASH, sender, receiver, amount."""
    transfers = read_tables(transfers_paths)
    hash_col = resolve_column(transfers, HASH_COLUMNS, label="transaction hash")
    sender_col = resolve_column(transfers, SENDER_COLUMNS, label="sender wallet")
    receiver_col = resolve_column(transfers, RECEIVER_COLUMNS, label="receiver wallet")
    amount_col = resolve_column(transfers, AMOUNT_COLUMNS, required=False)

    edges = pd.DataFrame(
        {
            "TRANSACTION_HASH": transfers[hash_col].astype(str),
            "sender": transfers[sender_col].astype(str),
            "receiver": transfers[receiver_col].astype(str),
        }
    )
    edges["amount"] = (
        pd.to_numeric(transfers[amount_col], errors="coerce").fillna(0.0)
        if amount_col
        else 1.0
    )

    if transactions_paths:
        transactions = read_tables(transactions_paths)
        tx_hash_col = resolve_column(transactions, HASH_COLUMNS, label="transaction hash")
        known = set(transactions[tx_hash_col].astype(str))
        missing = known - set(edges["TRANSACTION_HASH"])
        if missing:
            # Transactions with no transfer legs still need a feature row, so
            # attach them to a placeholder self-edge-free entry handled later.
            edges = pd.concat(
                [
                    edges,
                    pd.DataFrame(
                        {
                            "TRANSACTION_HASH": sorted(missing),
                            "sender": pd.NA,
                            "receiver": pd.NA,
                            "amount": 0.0,
                        }
                    ),
                ],
                ignore_index=True,
            )

    return edges


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Collapse parallel transfers into a weighted directed graph."""
    valid = edges.dropna(subset=["sender", "receiver"])
    valid = valid[valid["sender"] != valid["receiver"]]
    grouped = (
        valid.groupby(["sender", "receiver"], as_index=False)
        .agg(weight=("amount", "sum"), transfer_count=("amount", "size"))
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(pd.unique(valid[["sender", "receiver"]].values.ravel()))
    for sender, receiver, weight, count in grouped.itertuples(index=False):
        graph.add_edge(sender, receiver, weight=float(weight), transfer_count=int(count))
    return graph


def wallet_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Per-wallet structural features: degrees, component, community, patterns."""
    nodes = list(graph.nodes())
    undirected = graph.to_undirected(as_view=False)

    components = {}
    component_size = {}
    for idx, component in enumerate(nx.weakly_connected_components(graph)):
        for node in component:
            components[node] = idx
            component_size[node] = len(component)

    communities = nx.community.louvain_communities(undirected, seed=42, weight="weight")
    community_id = {}
    community_size = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_id[node] = idx
            community_size[node] = len(community)

    pagerank = nx.pagerank(graph, weight="weight")
    core_number = nx.core_number(nx.DiGraph(graph))
    clustering = nx.clustering(undirected)

    successors = {n: set(graph.successors(n)) for n in nodes}
    predecessors = {n: set(graph.predecessors(n)) for n in nodes}

    rows = []
    for node in nodes:
        out_nbrs = successors[node]
        in_nbrs = predecessors[node]
        out_degree = len(out_nbrs)
        in_degree = len(in_nbrs)

        # Scatter-gather seen from the source: funds leave this wallet through
        # many intermediaries that reconverge on one downstream wallet.
        downstream = defaultdict(int)
        for intermediary in out_nbrs:
            for target in successors[intermediary]:
                if target != node:
                    downstream[target] += 1
        scatter_out = max(downstream.values()) if downstream else 0

        # The same shape seen from the sink: many of this wallet's funders share
        # a single upstream origin.
        upstream = defaultdict(int)
        for intermediary in in_nbrs:
            for source in predecessors[intermediary]:
                if source != node:
                    upstream[source] += 1
        scatter_in = max(upstream.values()) if upstream else 0

        # Gather-scatter: a hub that collects from many wallets and immediately
        # redistributes to many others.
        gather_scatter = min(in_degree, out_degree)

        rows.append(
            {
                "wallet": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "weighted_in_degree": float(graph.in_degree(node, weight="weight")),
                "weighted_out_degree": float(graph.out_degree(node, weight="weight")),
                "transfer_in_count": int(graph.in_degree(node, weight="transfer_count")),
                "transfer_out_count": int(graph.out_degree(node, weight="transfer_count")),
                "degree_ratio": in_degree / (out_degree + 1.0),
                "reciprocal_partners": len(in_nbrs & out_nbrs),
                "is_fan_in": int(in_degree >= FAN_THRESHOLD and out_degree <= 1),
                "is_fan_out": int(out_degree >= FAN_THRESHOLD and in_degree <= 1),
                "scatter_out_score": int(scatter_out),
                "scatter_in_score": int(scatter_in),
                "gather_scatter_score": int(gather_scatter),
                "component_id": components[node],
                "cluster_size": component_size[node],
                "community_id": community_id[node],
                "community_size": community_size[node],
                "pagerank": pagerank[node],
                "core_number": core_number[node],
                "clustering_coeff": clustering[node],
            }
        )
    return pd.DataFrame(rows).set_index("wallet")


def cycle_edges(graph: nx.DiGraph, max_length: int = 4) -> tuple[set, dict]:
    """Edges that take part in a directed cycle of at most ``max_length`` hops."""
    on_cycle = set()
    node_cycle_count = defaultdict(int)
    for cycle in nx.simple_cycles(graph, length_bound=max_length):
        for idx, node in enumerate(cycle):
            nxt = cycle[(idx + 1) % len(cycle)]
            on_cycle.add((node, nxt))
            node_cycle_count[node] += 1
    return on_cycle, node_cycle_count


def transaction_features(
    edges: pd.DataFrame, graph: nx.DiGraph, max_cycle_length: int = 4
) -> pd.DataFrame:
    """One row per TRANSACTION_HASH built from the wallet-level features."""
    wallets = wallet_features(graph)
    on_cycle, node_cycle_count = cycle_edges(graph, max_cycle_length)

    legs = edges.dropna(subset=["sender", "receiver"]).copy()
    legs = legs[legs["sender"] != legs["receiver"]]

    # A transaction can carry several transfer legs; the largest-value leg is
    # treated as the primary edge and the rest are summarised as aggregates.
    legs = legs.sort_values("amount", ascending=False)
    primary = legs.drop_duplicates(subset="TRANSACTION_HASH", keep="first")

    sender_feats = wallets.reindex(primary["sender"]).reset_index(drop=True)
    receiver_feats = wallets.reindex(primary["receiver"]).reset_index(drop=True)

    out = pd.DataFrame({"TRANSACTION_HASH": primary["TRANSACTION_HASH"].to_numpy()})
    for column in wallets.columns:
        out[f"sender_{column}"] = sender_feats[column].to_numpy()
        out[f"receiver_{column}"] = receiver_feats[column].to_numpy()

    out["is_same_community"] = (
        out["sender_community_id"] == out["receiver_community_id"]
    ).astype(int)
    # Sender and receiver of an observed edge are always in the same weakly
    # connected component, so one cluster_size column says everything.
    out["cluster_size"] = out[["sender_cluster_size", "receiver_cluster_size"]].max(axis=1)
    out["community_size"] = out[
        ["sender_community_size", "receiver_community_size"]
    ].max(axis=1)

    pairs = list(zip(primary["sender"], primary["receiver"]))
    out["cycle_indicator"] = [int(pair in on_cycle) for pair in pairs]
    out["is_reciprocal_edge"] = [
        int(graph.has_edge(receiver, sender)) for sender, receiver in pairs
    ]
    out["sender_cycle_count"] = [node_cycle_count.get(s, 0) for s, _ in pairs]
    out["receiver_cycle_count"] = [node_cycle_count.get(r, 0) for _, r in pairs]

    out["fan_in_pattern"] = out["receiver_is_fan_in"]
    out["fan_out_pattern"] = out["sender_is_fan_out"]
    out["scatter_gather_pattern"] = (
        (out["sender_scatter_out_score"] >= FAN_THRESHOLD)
        | (out["receiver_scatter_in_score"] >= FAN_THRESHOLD)
    ).astype(int)
    out["gather_scatter_pattern"] = (
        (out["sender_gather_scatter_score"] >= FAN_THRESHOLD)
        | (out["receiver_gather_scatter_score"] >= FAN_THRESHOLD)
    ).astype(int)

    out["edge_weight"] = [
        float(graph[s][r]["weight"]) if graph.has_edge(s, r) else 0.0 for s, r in pairs
    ]
    out["edge_transfer_count"] = [
        int(graph[s][r]["transfer_count"]) if graph.has_edge(s, r) else 0 for s, r in pairs
    ]
    out["pagerank_ratio"] = out["receiver_pagerank"] / (out["sender_pagerank"] + 1e-12)

    leg_stats = (
        legs.groupby("TRANSACTION_HASH")
        .agg(
            n_transfer_legs=("amount", "size"),
            n_distinct_senders=("sender", "nunique"),
            n_distinct_receivers=("receiver", "nunique"),
        )
        .reset_index()
    )
    out = out.merge(leg_stats, on="TRANSACTION_HASH", how="left")

    # Transactions with no usable transfer legs still need a row so downstream
    # joins stay one-to-one.
    all_hashes = pd.DataFrame({"TRANSACTION_HASH": edges["TRANSACTION_HASH"].unique()})
    out = all_hashes.merge(out, on="TRANSACTION_HASH", how="left")

    numeric = out.columns.drop("TRANSACTION_HASH")
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Raw ids identify arbitrary partitions and would be meaningless to a tree
    # model, but their sizes and the same-group flags are informative.
    out = out.drop(
        columns=[
            "sender_component_id",
            "receiver_component_id",
            "sender_community_id",
            "receiver_community_id",
            "sender_cluster_size",
            "receiver_cluster_size",
        ]
    )
    return out.sort_values("TRANSACTION_HASH").reset_index(drop=True)


def build_graph_features(
    transfers_paths: Path | list[Path],
    transactions_paths: Path | list[Path] | None = None,
    max_cycle_length: int = 4,
) -> pd.DataFrame:
    if isinstance(transfers_paths, (str, Path)):
        transfers_paths = [Path(transfers_paths)]
    if isinstance(transactions_paths, (str, Path)):
        transactions_paths = [Path(transactions_paths)]
    edges = load_edges(transfers_paths, transactions_paths)
    graph = build_graph(edges)
    return transaction_features(edges, graph, max_cycle_length)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfers", type=Path, required=True, nargs="+",
                        help="transfers.parquet files; pass train and test together")
    parser.add_argument("--transactions", type=Path, default=None, nargs="+",
                        help="transactions.parquet files, to guarantee full hash coverage")
    parser.add_argument("--out", type=Path, default=Path("outputs/graph_features.csv"))
    parser.add_argument("--max-cycle-length", type=int, default=4)
    args = parser.parse_args(argv)

    features = build_graph_features(args.transfers, args.transactions, args.max_cycle_length)

    duplicated = features["TRANSACTION_HASH"].duplicated().sum()
    if duplicated:
        raise SystemExit(f"{duplicated} duplicated TRANSACTION_HASH rows - refusing to write")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)

    print(f"wrote {args.out}")
    print(f"rows: {len(features):,}  columns: {features.shape[1]}")
    print(f"unique TRANSACTION_HASH: {features['TRANSACTION_HASH'].nunique():,}")
    flags = [c for c in features.columns if c.endswith("_pattern")] + ["cycle_indicator"]
    print("pattern hit rates:")
    for flag in flags:
        print(f"  {flag:<26} {features[flag].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
