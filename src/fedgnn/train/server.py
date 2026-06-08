"""Server-side overlay graph, server GNN, adaptive aggregation, and checkpointing.

The server never sees raw flows. Each round it receives one
:class:`~fedgnn.train.client.ClientPayload` per client and

1. Stacks **every community embedding from every client** into a *community
   overlay graph* — nodes are communities, edges connect communities whose
   embeddings are cosine-similar above a **dynamic threshold** (lowered while any
   node is isolated). See :class:`OverlayGraphBuilder`.
2. Runs a :class:`ServerGNN` (GraphSAGE) over that overlay to produce a global
   context vector per community (an analysis/telemetry signal — see the note on
   ``ServerGNN`` and ``docs/train-pipeline.md``).
3. Fuses the local weights with **Adaptive Weighted FedAvg** — performance-based
   ``w_k``, *not* sample-size weighting — into the new global model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from fedgnn.train.client import ClientPayload

# Overlay graph


@dataclass
class OverlayGraph:
    """The community-overlay graph plus the diagnostics from building it."""

    data: Data  # PyG graph: x = community embeddings, edge_index = similarity edges
    similarity: torch.Tensor  # [N, N] full cosine-similarity matrix
    threshold: float  # the threshold actually used
    n_isolated: int  # communities still isolated at that threshold
    n_edges: int  # directed edge count (incl. self-loops)


class OverlayGraphBuilder:
    """Build the **community** overlay graph from all clients' community embeddings.

    Nodes = every community embedding from every client. An undirected edge joins
    two communities whose (L2-normalised) embeddings have cosine similarity
    ``>= threshold``. The threshold is **dynamic**: it starts high (sparse, only
    very-similar communities connect) and is lowered step-by-step *only while some
    node is isolated*, down to a floor — so the graph is as sparse as possible
    while keeping (almost) every community connected. Self-loops are added for the
    GraphSAGE pass.

    The cosine similarity is fully vectorised: one ``L2-normalise`` + one matmul
    gives the entire ``[N, N]`` similarity matrix; thresholding is a boolean mask.
    """

    def __init__(
        self,
        start_threshold: float = 0.7,
        min_threshold: float = 0.3,
        step: float = 0.05,
        add_self_loops: bool = True,
    ) -> None:
        self.start_threshold = start_threshold
        self.min_threshold = min_threshold
        self.step = step
        self.add_self_loops = add_self_loops

    def build(self, embeddings: torch.Tensor) -> OverlayGraph:
        """``embeddings``: ``[N_communities, embed_dim]`` -> :class:`OverlayGraph`."""
        n = embeddings.shape[0]
        normed = F.normalize(embeddings, p=2, dim=1)
        sim = normed @ normed.t()  # [N, N] cosine similarity (vectorised)

        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
        offdiag = sim.masked_fill(
            eye, float("-inf")
        )  # ignore self when checking degree

        # Lower the threshold until no node is isolated (or we hit the floor).
        threshold = self.start_threshold
        while True:
            isolated = int(((offdiag >= threshold).sum(dim=1) == 0).sum())
            if isolated == 0 or threshold <= self.min_threshold:
                break
            threshold = max(self.min_threshold, round(threshold - self.step, 6))

        adj = (sim >= threshold) & ~eye
        if self.add_self_loops:
            adj = adj | eye
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()

        data = Data(x=embeddings, edge_index=edge_index, num_nodes=n)
        return OverlayGraph(
            data=data,
            similarity=sim,
            threshold=threshold,
            n_isolated=isolated,
            n_edges=int(edge_index.shape[1]),
        )


# Server GNN


class ServerGNN(nn.Module):
    """A 2-layer GraphSAGE over the community overlay -> per-community context.

    It propagates information between *similar communities across different
    clients*, learning global structural patterns over the federation's community
    space (the FedGATSage idea). The resulting context vectors are logged for
    analysis. The actual weight aggregation is performed by :class:`AdaptiveFedAvg`
    using the ``w_k`` performance formula, so the server GNN does not alter the
    FedAvg math — this keeps the aggregation in strict compliance with standard
    FedAvg requirements. (No global label exists to train it end-to-end here; it
    runs as a fixed structural transform, and can be given a self-supervised
    objective — e.g. overlay link prediction — as a later extension.)
    """

    def __init__(
        self, in_dim: int = 32, hidden_dim: int = 32, out_dim: int = 32
    ) -> None:
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.sage1(x, edge_index))
        return self.sage2(h, edge_index)


# Adaptive Weighted FedAvg


class AdaptiveFedAvg:
    """Performance-based federated averaging (no sample-size weighting).

    For each client ``k`` with validation score ``A_k``::

        w_k = alpha + (1 - alpha) * (A_k - A_min) / (A_max - A_min + 1e-9)

    ``alpha`` is the floor weight every client keeps regardless of performance
    (so weak clients are never zeroed-out, which mitigates catastrophic forgetting
    of their locally-unique attack classes). Raw ``w_k`` are normalised to sum to
    1 before the weighted average of the model tensors.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def compute_weights(self, scores: list[float]) -> tuple[list[float], list[float]]:
        """Return ``(raw_w_k, normalised_w_k)`` for a list of validation scores."""
        a_min, a_max = min(scores), max(scores)
        raw = [
            self.alpha + (1.0 - self.alpha) * ((a - a_min) / (a_max - a_min + 1e-9))
            for a in scores
        ]
        total = sum(raw)
        normalised = [w / total for w in raw]
        return raw, normalised

    @staticmethod
    def aggregate(
        state_dicts: list[dict[str, torch.Tensor]], weights: list[float]
    ) -> dict[str, torch.Tensor]:
        """Weighted average of model state dicts (``weights`` must sum to 1)."""
        if not state_dicts:
            raise ValueError("no client state dicts to aggregate")

        aggregated: dict[str, torch.Tensor] = {}
        for key in state_dicts[0]:
            ref = state_dicts[0][key]
            # Integer buffers (e.g. counters) can't be averaged — copy the first.
            if not torch.is_floating_point(ref):
                aggregated[key] = ref.clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float)
            for w, sd in zip(weights, state_dicts):
                acc += w * sd[key].float()
            aggregated[key] = acc.to(ref.dtype)
        return aggregated


# Checkpointing


class CheckpointManager:
    """Robust per-round checkpointing with best-model tracking and resumption.

    Server-side artifacts are split across the two top-level trees so the server
    mirrors the per-client layout:

    * ``Models/server/latest_global_model.pt`` — every round; carries the round
      index so a resume knows exactly where to continue.
    * ``Models/server/best_global_model.pt``   — only when the average global
      metric improves.
    * ``Results/server/training_state.json``    — round, best metric, and full
      per-round history (client scores, raw + normalised ``w_k``, FLOPs) for
      auditing.
    """

    def __init__(self, models_dir: str | Path, results_dir: str | Path) -> None:
        self.models_dir = Path(models_dir) / "server"
        self.results_dir = Path(results_dir) / "server"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.models_dir / "latest_global_model.pt"
        self.best_path = self.models_dir / "best_global_model.pt"
        self.state_path = self.results_dir / "training_state.json"

    def save_latest(
        self, model_state: dict[str, torch.Tensor], round_idx: int, best_metric: float
    ) -> None:
        torch.save(
            {
                "model_state": model_state,
                "round": round_idx,
                "best_metric": best_metric,
            },
            self.latest_path,
        )

    def save_best(
        self, model_state: dict[str, torch.Tensor], round_idx: int, metric: float
    ) -> None:
        torch.save(
            {"model_state": model_state, "round": round_idx, "metric": metric},
            self.best_path,
        )

    def save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2))

    def load_latest(self) -> dict | None:
        """Return the latest checkpoint dict,
        or ``None`` if there is nothing to resume.
        """
        if not self.latest_path.exists():
            return None
        return torch.load(self.latest_path, weights_only=False)

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text())


# Server orchestration


@dataclass
class AggregationInfo:
    """Auditable record of a single aggregation step."""

    client_ids: list[int]
    scores: list[float]
    raw_weights: list[float]
    weights: list[float]
    communities_per_client: list[int]  # how many communities each client sent
    n_communities: int  # total overlay nodes
    overlay_threshold: float  # dynamic threshold actually used
    overlay_edges: int  # overlay edge count (incl. self-loops)
    overlay_isolated: int  # communities still isolated at that threshold
    context: torch.Tensor  # [n_communities, embed_dim] server-GNN context vectors


class FederatedServer:
    """Coordinator: community overlay construction, server GNN, adaptive aggregation."""

    def __init__(
        self,
        alpha: float = 0.2,
        overlay_start: float = 0.7,
        overlay_min: float = 0.3,
        overlay_step: float = 0.05,
        embed_dim: int = 32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.aggregator = AdaptiveFedAvg(alpha=alpha)
        self.overlay_builder = OverlayGraphBuilder(
            start_threshold=overlay_start,
            min_threshold=overlay_min,
            step=overlay_step,
        )
        self.server_gnn = ServerGNN(in_dim=embed_dim).to(self.device)

    def aggregate(
        self, payloads: list[ClientPayload]
    ) -> tuple[dict[str, torch.Tensor], AggregationInfo]:
        """Fuse client payloads into a new global model + an audit record.

        1. Stack **every** community embedding from **every** client into one node
           set, build the dynamic-threshold cosine-similarity overlay graph.
        2. Run the server GraphSAGE over it (global structural context; logged).
        3. Adaptive Weighted FedAvg of the client state-dicts using ``w_k``.
        """
        # Weight by attack-detection skill (agg_score), not the raw val metric: a
        # benign-only client maxes balanced accuracy but has agg_score 0, so it can
        # no longer dominate FedAvg and drag the global model to "always benign".
        scores = [p.agg_score for p in payloads]
        communities_per_client = [p.community_embeddings.shape[0] for p in payloads]
        embeddings = torch.cat([p.community_embeddings for p in payloads], dim=0).to(
            self.device
        )

        overlay = self.overlay_builder.build(embeddings)
        with torch.no_grad():
            self.server_gnn.eval()
            context = self.server_gnn(overlay.data.x, overlay.data.edge_index).cpu()

        raw_w, norm_w = self.aggregator.compute_weights(scores)
        global_state = self.aggregator.aggregate(
            [p.state_dict for p in payloads], norm_w
        )

        info = AggregationInfo(
            client_ids=[p.client_id for p in payloads],
            scores=scores,
            raw_weights=raw_w,
            weights=norm_w,
            communities_per_client=communities_per_client,
            n_communities=int(embeddings.shape[0]),
            overlay_threshold=overlay.threshold,
            overlay_edges=overlay.n_edges,
            overlay_isolated=overlay.n_isolated,
            context=context,
        )
        return global_state, info
