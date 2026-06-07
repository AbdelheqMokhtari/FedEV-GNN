"""Federated training pipeline for the FedEV-GNN intrusion detection system.

The package mirrors the simulated FL topology:

* :mod:`fedgnn.train.client` — the edge device. A bottlenecked GAT that turns a
  client's whole-flow graph (``graph.pt``) into per-edge intrusion predictions,
  trains locally, and exports a payload (weights + hub-node embedding + score).
* :mod:`fedgnn.train.server` — the coordinator. Builds a client-overlay graph
  from the hub embeddings, runs a server-side GNN for global context, and fuses
  the local models with **Adaptive Weighted FedAvg** plus robust checkpointing.
* :mod:`fedgnn.train.cli` — the single-machine orchestrator simulating the rounds.
"""

from fedgnn.train.client import ClientPayload, FederatedClient, LocalGAT
from fedgnn.train.experiment import Experiment
from fedgnn.train.server import (
    AdaptiveFedAvg,
    CheckpointManager,
    FederatedServer,
    OverlayGraph,
    OverlayGraphBuilder,
    ServerGNN,
)

__all__ = [
    "AdaptiveFedAvg",
    "CheckpointManager",
    "ClientPayload",
    "Experiment",
    "FederatedClient",
    "FederatedServer",
    "LocalGAT",
    "OverlayGraph",
    "OverlayGraphBuilder",
    "ServerGNN",
]
