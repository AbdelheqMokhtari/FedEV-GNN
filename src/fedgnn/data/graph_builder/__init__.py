"""PyG graph construction per federated client."""

from fedgnn.data.graph_builder.edge_graph import (
    aggregate_node_features,
    build_node_index,
    build_snapshot,
)
from fedgnn.data.graph_builder.temporal_graph import (
    build_metadata,
    build_snapshot_sequence,
    iter_windows,
    report_parameter_count,
    run,
    save_mat,
    save_pt,
    static_graph_metadata,
    temporal_metadata,
)

__all__ = [
    "aggregate_node_features",
    "build_metadata",
    "build_node_index",
    "build_snapshot",
    "build_snapshot_sequence",
    "iter_windows",
    "report_parameter_count",
    "run",
    "save_mat",
    "save_pt",
    "static_graph_metadata",
    "temporal_metadata",
]
