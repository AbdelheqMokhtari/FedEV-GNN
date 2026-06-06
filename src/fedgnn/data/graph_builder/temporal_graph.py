"""Slice client flows into temporal PyG snapshot sequences for T-GAT training."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from scipy.io import savemat
from torch_geometric.data import Data

from fedgnn.data.graph_builder.edge_graph import LABEL_COL, build_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLIENTS_DIR = PROJECT_ROOT / "data" / "clients"

TIME_COL = "FLOW_START_MILLISECONDS"
DEFAULT_WINDOW_MS = 3_600_000  # 1 hour
MIN_EDGES = 1


# Temporal windowing


def iter_windows(
    df: pl.DataFrame, window_ms: int, time_col: str = TIME_COL
) -> Iterator[tuple[int, int, pl.DataFrame]]:
    """Yield ``(start_ms, end_ms, sub_df)`` for each non-empty window, in order.

    Windows are anchored to the batch's earliest timestamp so bucket
    boundaries are stable regardless of how the data was partitioned.
    """
    t0 = df[time_col].min()
    tagged = df.with_columns(((pl.col(time_col) - t0) // window_ms).alias("__win"))
    for (win_id,), sub in tagged.group_by("__win", maintain_order=True):
        start = t0 + int(win_id) * window_ms
        yield start, start + window_ms, sub.drop("__win")


def build_snapshot_sequence(
    df: pl.DataFrame,
    feature_cols: list[str],
    window_ms: int = DEFAULT_WINDOW_MS,
    label_col: str = LABEL_COL,
    min_edges: int = MIN_EDGES,
) -> list:
    """Build one ``Data`` snapshot per time window, skipping near-empty ones."""
    snapshots = []
    for start, end, sub in iter_windows(df, window_ms):
        if sub.height < min_edges:
            continue
        snap = build_snapshot(sub, feature_cols, label_col)
        snap.window = (start, end)
        snapshots.append(snap)
    return snapshots


# Saving


def save_pt(graph_or_snapshots: Data | list, out_path: Path) -> None:
    """Persist a ``Data`` graph (or an ordered ``list[Data]``) for training.

    Thin ``torch.save`` wrapper — used both for the single whole-client
    static graph (``graph.pt``) and the temporal snapshot sequence
    (``graphs.pt``); ``torch.load`` hands back the exact same structure.
    """
    torch.save(graph_or_snapshots, out_path)


def save_mat(graph: Data, out_path: Path) -> None:

    n_nodes = graph.num_nodes
    src_idx, dst_idx = graph.edge_index.numpy()
    weights = np.ones(src_idx.shape[0], dtype=np.float32)
    adjacency = sp.coo_matrix(
        (weights, (src_idx, dst_idx)), shape=(n_nodes, n_nodes)
    ).tocsr()

    savemat(
        out_path,
        {
            "adjacency": adjacency,
            "node_features": graph.x.numpy(),
            "edge_features": graph.edge_attr.numpy(),
            "labels": graph.y.numpy(),
            "node_ids": np.array(graph.node_ids, dtype=object),
            "feature_names": np.array(graph.feature_names, dtype=object),
        },
    )


# Metadata


def _stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "avg": round(sum(values) / len(values), 1),
        "min": min(values),
        "max": max(values),
    }


def _class_distribution(
    labels: torch.Tensor, label_map_inv: dict[int, str]
) -> dict[str, int]:
    if labels.numel() == 0:
        return {}
    uniq, counts = torch.unique(labels, return_counts=True)
    return {
        label_map_inv.get(int(u), str(int(u))): int(c)
        for u, c in zip(uniq.tolist(), counts.tolist())
    }


def static_graph_metadata(graph: Data, label_map_inv: dict[int, str]) -> dict:
    """Summarise the whole-client static graph: size + class distribution."""
    return {
        "nodes": graph.num_nodes,
        "edges": int(graph.edge_index.shape[1]),
        "class_distribution": _class_distribution(graph.y, label_map_inv),
    }


def temporal_metadata(snapshots: list, label_map_inv: dict[int, str]) -> dict:
    """Summarise the temporal snapshot sequence: counts + class distribution."""
    node_counts = [s.num_nodes for s in snapshots]
    edge_counts = [int(s.edge_index.shape[1]) for s in snapshots]
    all_labels = (
        torch.cat([s.y for s in snapshots])
        if snapshots
        else torch.empty(0, dtype=torch.long)
    )

    return {
        "n_snapshots": len(snapshots),
        "nodes": _stats(node_counts),
        "edges": _stats(edge_counts),
        "class_distribution": _class_distribution(all_labels, label_map_inv),
    }


def build_metadata(
    client_id: int,
    hub_ip: str,
    graph: Data,
    snapshots: list,
    label_map_inv: dict[int, str],
) -> dict:
    """Per-client metadata: the static graph first, the temporal sequence second.

    ``graph`` mirrors ``graph.pt``/``graph.mat`` (the single graph most work
    starts from); ``temporal`` mirrors ``graphs.pt`` (the T-GAT sequence).
    """
    return {
        "client_id": client_id,
        "hub_ip": hub_ip,
        "graph": static_graph_metadata(graph, label_map_inv),
        "temporal": temporal_metadata(snapshots, label_map_inv),
    }


# Theoretical T-GAT parameter count


def gat_layer_params(
    in_channels: int, out_channels: int, heads: int, concat: bool = True
) -> int:

    lin = in_channels * heads * out_channels
    attention = 2 * heads * out_channels
    bias = heads * out_channels if concat else out_channels
    return lin + attention + bias


def report_parameter_count(in_channels: int, num_classes: int) -> dict[str, int]:

    l1_per_head, l1_heads = 64, 2
    l2_per_head, l2_heads = 32, 1

    l1_out = l1_per_head * l1_heads
    l2_out = l2_per_head * l2_heads

    gat1 = gat_layer_params(in_channels, l1_per_head, l1_heads)
    gat2 = gat_layer_params(l1_out, l2_per_head, l2_heads)
    classifier = l2_out * num_classes + num_classes  # weight + bias
    total = gat1 + gat2 + classifier

    print("\n[build] theoretical local T-GAT parameter count")
    print(f"  input                                    : {in_channels} features")
    print(
        f"  GAT layer 1  (heads={l1_heads}, {l1_per_head}/head -> {l1_out:>3}) "
        f"   : {gat1:>9,}"
    )
    print(
        f"  GAT layer 2  (heads={l2_heads}, {l2_per_head}/head -> {l2_out:>3}) "
        f"  : {gat2:>9,}"
    )
    print(
        f"  Linear classifier  ({l2_out} -> {num_classes})       "
        f"  : {classifier:>9,}"
    )
    print(f"  {'TOTAL':>54}: {total:>9,}")

    return {
        "input_features": in_channels,
        "num_classes": num_classes,
        "layers": {
            "gat1": {
                "heads": l1_heads,
                "out_per_head": l1_per_head,
                "out_channels": l1_out,
                "params": gat1,
            },
            "gat2": {
                "heads": l2_heads,
                "out_per_head": l2_per_head,
                "out_channels": l2_out,
                "params": gat2,
            },
            "classifier": {
                "in_channels": l2_out,
                "out_channels": num_classes,
                "params": classifier,
            },
        },
        "total_params": total,
    }


#  run


def run(
    window_ms: int = DEFAULT_WINDOW_MS,
    min_edges: int = MIN_EDGES,
    n_clients: int | None = None,
) -> None:
    """Build PyG static + temporal graphs, ``.mat`` exports, and metadata per client."""
    feature_cols = json.loads((PROCESSED_DIR / "selected_features.json").read_text())
    label_map = json.loads((PROCESSED_DIR / "label_map.json").read_text())
    label_map_inv = {v: k for k, v in label_map.items()}

    client_dirs = sorted(CLIENTS_DIR.glob("client_*"))
    if not client_dirs:
        raise FileNotFoundError(
            "No federated clients found — run `fedgnn-data split` first"
        )
    if n_clients is not None:
        client_dirs = client_dirs[:n_clients]

    print(
        f"[build] {len(feature_cols)} edge features | {len(label_map)} classes | "
        f"{window_ms / 60_000:.0f}-minute windows"
    )

    for client_dir in client_dirs:
        cid = int(client_dir.name.split("_")[-1])
        meta_in = json.loads((client_dir / "meta.json").read_text())
        hub_ip = meta_in.get("hub_ip", "?")

        df = pl.read_parquet(client_dir / "flows.parquet")
        snapshots = build_snapshot_sequence(
            df, feature_cols, window_ms, min_edges=min_edges
        )

        # One whole-client static graph — same PyG `Data` structure as each
        # snapshot, just built over every flow. Saved as `.pt` so it can be
        # loaded straight into training (e.g. a non-temporal baseline GAT),
        # and mirrored to `.mat` for inspection from the very same object.
        static_graph = build_snapshot(df, feature_cols, LABEL_COL)
        save_pt(static_graph, client_dir / "graph.pt")
        save_mat(static_graph, client_dir / "graph.mat")

        save_pt(snapshots, client_dir / "graphs.pt")

        meta = build_metadata(cid, hub_ip, static_graph, snapshots, label_map_inv)
        (client_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2))

        g = meta["graph"]
        print(
            f"  client_{cid:02d}  hub={hub_ip:<16} "
            f"graph: {g['nodes']:>4} nodes / {g['edges']:>7,} edges  |  "
            f"temporal: {len(snapshots):>4} snapshots"
        )

    param_report = report_parameter_count(
        in_channels=len(feature_cols), num_classes=len(label_map)
    )
    param_path = CLIENTS_DIR / "tgat_param_count.json"
    param_path.write_text(json.dumps(param_report, indent=2))
    print(f"[build] wrote {param_path.relative_to(PROJECT_ROOT)}")
