import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

SRC = "IPV4_SRC_ADDR"
DST = "IPV4_DST_ADDR"
LABEL_COL = "y_multiclass"


def build_node_index(df: pl.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Deterministic (sorted) IP -> node-index mapping for one flow batch."""
    node_ids = sorted(set(df[SRC].to_list()) | set(df[DST].to_list()))
    return node_ids, {ip: i for i, ip in enumerate(node_ids)}


def aggregate_node_features(
    edge_feats: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Per-node feature vector = mean of incident (in + out) flow features.

    Every flow's feature vector is scattered onto both its source and
    destination node, then averaged by incidence count. This is what lets
    GAT's attention mechanism compare nodes that only ever send, only ever
    receive, or do both, on equal footing.
    """
    n_feats = edge_feats.shape[1]
    sums = np.zeros((n_nodes, n_feats), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)

    np.add.at(sums, src_idx, edge_feats)
    np.add.at(counts, src_idx, 1)
    np.add.at(sums, dst_idx, edge_feats)
    np.add.at(counts, dst_idx, 1)

    counts = np.maximum(counts, 1)[
        :, None
    ]  # every node has >=1 incident flow by construction
    return (sums / counts).astype(np.float32)


def build_snapshot(
    df: pl.DataFrame, feature_cols: list[str], label_col: str = LABEL_COL
) -> Data:
    """Turn one flow batch (a time window for one client) into a PyG ``Data`` graph."""
    node_ids, node_to_idx = build_node_index(df)

    mapped = df.select(
        pl.col(SRC).replace_strict(node_to_idx).alias("__s"),
        pl.col(DST).replace_strict(node_to_idx).alias("__d"),
    )
    src_idx = mapped["__s"].to_numpy()
    dst_idx = mapped["__d"].to_numpy()

    edge_feats = df.select(feature_cols).to_numpy().astype(np.float32)
    node_feats = aggregate_node_features(edge_feats, src_idx, dst_idx, len(node_ids))

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long),
        edge_attr=torch.tensor(edge_feats, dtype=torch.float),
        y=torch.tensor(df[label_col].to_numpy().astype(np.int64), dtype=torch.long),
    )
    data.node_ids = node_ids
    data.feature_names = list(feature_cols)
    return data
