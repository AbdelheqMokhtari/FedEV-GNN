"""Client-side local model and training loop (the simulated edge device).

Each client owns one ``graph.pt``.

* ``x``          per-node aggregated flow features, shape ``[N_nodes, F]`` where
  ``F`` is the number of selected (engineered) features — see ``selected_features.json``
* ``edge_index`` ``[2, N_edges]`` (one edge per NetFlow record)
* ``y``          **per-edge** class labels, shape ``[N_edges]``  ← labels live on
  the flows, not the nodes, so intrusion detection is *edge classification*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from networkx.algorithms.community import louvain_communities
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from fedgnn.data.graph_builder.edge_graph import aggregate_node_features
from fedgnn.evaluation import (
    count_parameters,
    count_training_flops,
    predict_model,
    report_from_predictions,
    track_resources,
)

# src/fedgnn/train/client.py -> parents[3] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLIENTS_DIR = PROJECT_ROOT / "data" / "clients"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Fallback default only; the real in-channel count is read from the graph at build
# time (``self.graph.x.shape[1]``), so this tracks the selected-feature set size.
IN_CHANNELS = 55
EMBED_DIM = 32
TIME_COL = "FLOW_START_MILLISECONDS"
LABEL_COL = "y_multiclass"


CLASS_WEIGHT_SCHEMES = ("sqrt_inverse", "inverse", "none")


def _compute_class_weights(
    targets: torch.Tensor, num_classes: int, scheme: str = "sqrt_inverse"
) -> torch.Tensor | None:
    """Per-class cross-entropy weights (absent classes -> 0), normalised to mean 1.

    Keeps the loss from collapsing onto the dominant ``benign``/``ddos`` flows
    without over-rewarding the rarest classes. ``scheme``:

    * ``inverse``      — w_c ∝ 1/count_c. Strong, but a class 1000× rarer gets
      1000× the weight, so the model over-predicts rare attacks → many false
      positives.
    * ``sqrt_inverse`` — w_c ∝ 1/sqrt(count_c) (**default, softened**). Same
      1000×-rarer class gets only ~31× weight: still corrects imbalance but far
      less trigger-happy, so benign precision recovers.
    * ``none``         — uniform (no weighting).

    Weights are scaled to average 1 over the present classes so the loss
    magnitude (and thus the effective learning rate) is comparable across schemes.
    """
    if scheme == "none":
        return None
    if scheme not in CLASS_WEIGHT_SCHEMES:
        raise ValueError(
            f"class_weight must be one of {CLASS_WEIGHT_SCHEMES}, got {scheme!r}"
        )

    counts = torch.bincount(targets, minlength=num_classes).float()
    present = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float, device=counts.device)
    if not present.any():
        return weights

    if scheme == "inverse":
        raw = 1.0 / counts[present]
    else:  # sqrt_inverse
        raw = 1.0 / torch.sqrt(counts[present])
    weights[present] = raw / raw.mean()  # normalise to mean 1 over present classes
    return weights


def _fit_node_features(
    edge_attr: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    n_nodes: int,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    node_feats = aggregate_node_features(
        edge_attr[train_mask], src_idx[train_mask], dst_idx[train_mask], n_nodes
    )
    logged = np.log1p(np.clip(node_feats, 0.0, None))
    mean = logged.mean(axis=0, keepdims=True)
    std = logged.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    x_norm = ((logged - mean) / std).astype(np.float32)
    return x_norm, mean, std


def _detect_communities(
    edge_index: torch.Tensor, n_nodes: int, seed: int, resolution: float = 1.0
) -> list[list[int]]:
    """Louvain community detection on the client's graph structure."""
    src, dst = edge_index
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    keep = lo != hi  # drop self-loops for the structural graph
    pairs = torch.stack([lo[keep], hi[keep]])
    uniq, counts = torch.unique(pairs, dim=1, return_counts=True)
    uniq = uniq.cpu().numpy()
    counts = counts.cpu().numpy()

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(
        (int(uniq[0, i]), int(uniq[1, i]), float(counts[i]))
        for i in range(uniq.shape[1])
    )

    communities = louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed
    )
    return [sorted(int(n) for n in c) for c in communities]


# Model


class LocalGAT(nn.Module):
    """Deep multi-head Graph Attention Network for per-flow intrusion detection.

    A scaled-up node encoder — multi-head GAT layers with **ELU + LayerNorm +
    dropout** and **residual connections** between the hidden blocks, so depth can
    be added without the usual GAT instability/over-smoothing::

        GATConv(in -> H, heads)            -> hidden (=H·heads)   ELU,LN,drop
        [ GATConv(hidden -> H, heads) + residual ] × (num_layers-2)
        GATConv(hidden -> embed_dim, heads, concat=False)  -> embed_dim (=z)

    The final ``embed_dim`` node embedding ``z`` is both the input to the edge
    classifier and the hub fingerprint sent to the server overlay (so it stays a
    tunable hyper-parameter shared with :class:`ServerGNN`).

    Edge classifier (deeper MLP) on the concatenated endpoint embeddings::

        [z_src ‖ z_dst] -> 2·embed -> embed -> num_classes   (ReLU, LN, dropout)
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = 10,
        hidden_per_head: int = 64,
        heads: int = 4,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        self.dropout = dropout
        self.embed_dim = embed_dim
        hidden = hidden_per_head * heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer: in_channels -> hidden (heads concatenated).
        self.convs.append(
            GATConv(in_channels, hidden_per_head, heads=heads, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden))

        # Hidden layers: hidden -> hidden, wrapped in a residual connection.
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden, hidden_per_head, heads=heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden))

        # Output layer: hidden -> embed_dim (heads averaged, not concatenated).
        self.convs.append(
            GATConv(hidden, embed_dim, heads=heads, concat=False, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the ``embed_dim`` node embeddings ``z`` (the GAT encoder output)."""
        # Input block.
        h = self.norms[0](F.elu(self.convs[0](x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Residual hidden blocks (dims match: hidden -> hidden).
        for conv, norm in zip(self.convs[1:-1], self.norms[1:-1]):
            res = h
            h = norm(F.elu(conv(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + res

        # Output block.
        return self.norms[-1](self.convs[-1](h, edge_index))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(edge_logits[N_edges, num_classes], node_embeddings[N, embed])``."""
        z = self.encode(x, edge_index)
        src, dst = edge_index
        edge_repr = torch.cat([z[src], z[dst]], dim=1)
        return self.classifier(edge_repr), z


# Payload


@dataclass
class ClientPayload:
    """What a client uploads to the server after a round of local training."""

    client_id: int
    state_dict: dict[str, torch.Tensor]  # local model weights (on CPU)
    community_embeddings: torch.Tensor  # [n_communities, embed_dim] (CPU)
    community_sizes: list[int]  # node count per community (for auditing)
    val_score: float  # local validation score on the --metric (logged)
    agg_score: float  # attack-detection skill -> drives the FedAvg weight
    num_train_edges: int  # for auditing only — NOT used for weighting


# Client


class FederatedClient:
    """One simulated edge device: owns a graph, a local model, and a 3-way split.

    Edges (flows) are split into **train / val / test**. The default split is
    *stratified* (per-class balanced) so every class is present in all three
    splits — the fix for ToN-IoT's time-localised attack campaigns, which made a
    plain ``temporal`` split strand whole attack classes in train-only or
    test-only and pinned ``macro_attack_recall`` at 0 (see ``_make_split``).
    """

    def __init__(
        self,
        client_dir: str | Path,
        num_classes: int,
        device: torch.device | str = "cpu",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split: str = "stratified",
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        dropout: float = 0.5,
        hidden_per_head: int = 64,
        heads: int = 4,
        num_layers: int = 3,
        embed_dim: int = EMBED_DIM,
        louvain_resolution: float = 1.0,
        metric: str = "balanced_accuracy",
        use_class_weights: bool = True,
        class_weight: str = "sqrt_inverse",
        benign_class: int = 0,
        seed: int = 42,
    ) -> None:
        self.client_dir = Path(client_dir)
        self.client_id = int(self.client_dir.name.split("_")[-1])
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.metric = metric
        self.use_class_weights = use_class_weights
        # "none" scheme or use_class_weights=False both mean uniform weighting.
        self.class_weight = class_weight if use_class_weights else "none"
        self.benign_class = benign_class
        self.split_kind = split

        # Load on CPU, build the split + leak-free features in numpy, then move.
        graph: Data = self._load_graph()
        self.hub_ip, self.hub_index = self._resolve_hub(graph)
        train_mask, val_mask, test_mask = self._make_split(
            graph, val_ratio, test_ratio, split, seed
        )
        graph.x = self._rebuild_features(graph, train_mask)
        self.graph = graph.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)

        # Community structure is fixed (graph topology doesn't change across
        # rounds), so detect it once; only the embeddings are re-pooled each round.
        self.communities = _detect_communities(
            graph.edge_index,
            graph.num_nodes,
            seed + self.client_id,
            resolution=louvain_resolution,
        )
        self.community_sizes = [len(c) for c in self.communities]
        self._community_index = [
            torch.tensor(c, dtype=torch.long, device=self.device)
            for c in self.communities
        ]

        self.model = LocalGAT(
            in_channels=self.graph.x.shape[1],
            num_classes=num_classes,
            hidden_per_head=hidden_per_head,
            heads=heads,
            num_layers=num_layers,
            embed_dim=embed_dim,
            dropout=dropout,
        ).to(self.device)
        self.num_params = count_parameters(self.model)

        # Class weights from the local *training* edges only (leak-free).
        self._class_weights = None
        if self.class_weight != "none":
            y_train = self.graph.y[self.train_mask]
            weights = _compute_class_weights(y_train, num_classes, self.class_weight)
            if weights is not None:
                self._class_weights = weights.to(self.device)

        # Computational-cost accounting (filled by train()/evaluate()).
        self.last_flops_per_epoch = 0
        self.last_flops_round = 0
        self.cumulative_flops = 0
        self.last_report: dict | None = None
        self.test_report: dict | None = None
        # Raw (preds, targets) from the most recent eval, kept on CPU so the server
        # can POOL them across clients into one honest multi-class report.
        self.last_preds: torch.Tensor | None = None
        self.last_targets: torch.Tensor | None = None
        self.test_preds: torch.Tensor | None = None
        self.test_targets: torch.Tensor | None = None
        self.last_train_usage = None  # ResourceUsage from the last train()
        self.last_eval_usage = None  # ResourceUsage from the last evaluate()

    # construction helpers

    def _load_graph(self) -> Data:
        graph_path = self.client_dir / "graph.pt"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"[client_{self.client_id:02d}] missing {graph_path} — "
                "run `fedgnn-data build` first"
            )
        # Kept on CPU: we rebuild node features in numpy before moving to device.
        return torch.load(graph_path, weights_only=False)

    def _edge_times(self, graph: Data) -> np.ndarray | None:
        """Per-edge timestamps from ``flows.parquet`` (aligned to edge order)."""
        flows_path = self.client_dir / "flows.parquet"
        if not flows_path.exists():
            return None
        n_edges = int(graph.edge_index.shape[1])
        df = pl.read_parquet(flows_path, columns=[TIME_COL, LABEL_COL])
        if df.height != n_edges:
            return None
        labels = df[LABEL_COL].to_numpy()
        if not np.array_equal(labels, graph.y.cpu().numpy()):
            return None  # row order doesn't match edge order — don't trust times
        return df[TIME_COL].to_numpy()

    def _make_split(
        self, graph: Data, val_ratio: float, test_ratio: float, split: str, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """3-way edge split -> (train_mask, val_mask, test_mask).

        Four strategies (``split``):

        * ``stratified`` (default) — **per-class** random split: within each class
          the edges are shuffled and cut by ratio, so every class with >=3 samples
          lands in train *and* val *and* test. This is the fix for the degenerate
          splits that pinned ``macro_attack_recall`` at 0 — on ToN-IoT the
          time-localised attack campaigns made a plain ``temporal`` split put whole
          attack classes in train-only or test-only (see docs).
        * ``stratified_temporal`` — same per-class balancing, but within each class
          the *earliest* flows go to train and the *latest* to test, keeping a
          past->future flavour without orphaning any class.
        * ``temporal`` — global past->future order (earliest train, latest test).
        * ``random`` — global random order.
        """
        n_edges = int(graph.edge_index.shape[1])
        if split in ("stratified", "stratified_temporal"):
            return self._stratified_split(
                graph, val_ratio, test_ratio, split, seed, n_edges
            )

        n_test = max(1, int(round(n_edges * test_ratio)))
        n_val = max(1, int(round(n_edges * val_ratio)))
        n_train = n_edges - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"[client_{self.client_id:02d}] val+test ratios leave no training "
                f"edges (n_edges={n_edges})"
            )

        order: np.ndarray
        if split == "temporal":
            times = self._edge_times(graph)
            if times is not None:
                order = np.argsort(times, kind="stable")  # earliest -> latest
            else:
                print(
                    f"[client_{self.client_id:02d}] no usable timestamps — "
                    "falling back to a random split"
                )
                order = self._rng_perm(n_edges, seed)
        else:
            order = self._rng_perm(n_edges, seed)

        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]
        return (
            self._to_mask(train_idx, n_edges),
            self._to_mask(val_idx, n_edges),
            self._to_mask(test_idx, n_edges),
        )

    def _stratified_split(
        self,
        graph: Data,
        val_ratio: float,
        test_ratio: float,
        split: str,
        seed: int,
        n_edges: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-class balanced split so every class spans train/val/test.

        Classes too small to span all three splits degrade gracefully: a 2-sample
        class goes train+val, a singleton goes to train only — so a rare attack is
        always *learnable*, never stranded in test alone.
        """
        y = graph.y.cpu().numpy()
        rng = np.random.default_rng(seed + self.client_id)

        times = None
        if split == "stratified_temporal":
            times = self._edge_times(graph)
            if times is None:
                print(
                    f"[client_{self.client_id:02d}] no usable timestamps — "
                    "falling back to a stratified random split"
                )

        train_parts, val_parts, test_parts = [], [], []
        for c in np.unique(y):
            cls_idx = np.where(y == c)[0]
            # order within the class: by time (stratified_temporal) or shuffled
            if times is not None:
                cls_idx = cls_idx[np.argsort(times[cls_idx], kind="stable")]
            else:
                cls_idx = rng.permutation(cls_idx)

            n = len(cls_idx)
            if n == 1:  # singleton: train it (can't also val/test)
                train_parts.append(cls_idx)
                continue
            if n == 2:  # split train/val, skip test
                train_parts.append(cls_idx[:1])
                val_parts.append(cls_idx[1:])
                continue

            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            n_train = n - n_val - n_test
            if n_train < 1:  # tiny class: guarantee >=1 train, share the rest
                n_train = 1
                n_val = (n - 1) // 2
                n_test = n - 1 - n_val
            train_parts.append(cls_idx[:n_train])
            val_parts.append(cls_idx[n_train : n_train + n_val])
            test_parts.append(cls_idx[n_train + n_val :])

        def _cat(parts: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

        return (
            self._to_mask(_cat(train_parts), n_edges),
            self._to_mask(_cat(val_parts), n_edges),
            self._to_mask(_cat(test_parts), n_edges),
        )

    def _to_mask(self, idx: np.ndarray, n_edges: int) -> torch.Tensor:
        m = torch.zeros(n_edges, dtype=torch.bool)
        if len(idx):
            m[torch.from_numpy(np.asarray(idx, dtype=np.int64))] = True
        return m

    def _rng_perm(self, n_edges: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.client_id)
        return rng.permutation(n_edges)

    def _rebuild_features(self, graph: Data, train_mask: torch.Tensor) -> torch.Tensor:
        """Leak-free, normalised node features from train edges only."""
        edge_attr = graph.edge_attr.cpu().numpy()
        src_idx = graph.edge_index[0].cpu().numpy()
        dst_idx = graph.edge_index[1].cpu().numpy()
        x_norm, _, _ = _fit_node_features(
            edge_attr, src_idx, dst_idx, graph.num_nodes, train_mask.cpu().numpy()
        )
        return torch.from_numpy(x_norm)

    def _resolve_hub(self, graph: Data) -> tuple[str, int]:
        """Hub IP -> node index, via graph_meta.json + the graph's node_ids list."""
        meta_path = self.client_dir / "graph_meta.json"
        hub_ip = "?"
        if meta_path.exists():
            hub_ip = json.loads(meta_path.read_text()).get("hub_ip", "?")

        node_ids = list(getattr(graph, "node_ids", []))
        if hub_ip in node_ids:
            return hub_ip, node_ids.index(hub_ip)

        # Fallback: highest-degree node is, by construction, the hub.
        deg = torch.bincount(graph.edge_index.reshape(-1), minlength=graph.num_nodes)
        idx = int(deg.argmax())
        resolved = node_ids[idx] if idx < len(node_ids) else f"node_{idx}"
        return resolved, idx

    # weight exchange (FedAvg)

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Detached CPU copy of the local model weights."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load (a copy of) global weights onto the local model / device."""
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )

    # training / evaluation

    def train(self, epochs: int) -> dict:
        """Run ``epochs`` of full-batch local training.

        Returns a record with the final loss, the computational-cost figures
        (FLOPs per epoch / per round / cumulative), and the **measured** resource
        cost of the local training (wall time, CPU %, process RSS, peak GPU mem) so
        ``Results/`` captures what each round actually cost. A fresh optimiser is
        created each round (standard FedAvg): local optimiser state is not carried
        across rounds because the weights are reset from the global model every round.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=self._class_weights)
        x, edge_index, y = self.graph.x, self.graph.edge_index, self.graph.y

        # Measure the per-step compute once (constant across full-batch epochs).
        self.last_flops_per_epoch = count_training_flops(
            self.model, x, edge_index, y, self.train_mask, criterion
        )

        self.model.train()
        last_loss = 0.0
        # Time/measure only the actual training loop (excludes the FLOPs probe).
        with track_resources(self.device) as res:
            for _ in range(epochs):
                optimizer.zero_grad()
                logits, _ = self.model(x, edge_index)
                loss = criterion(logits[self.train_mask], y[self.train_mask])
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach())

        self.last_flops_round = self.last_flops_per_epoch * epochs
        self.cumulative_flops += self.last_flops_round
        self.last_train_usage = res.usage
        return {
            "final_loss": last_loss,
            "local_epochs": epochs,
            "flops_per_epoch": self.last_flops_per_epoch,
            "flops_this_round": self.last_flops_round,
            "cumulative_flops": self.cumulative_flops,
            **res.usage.as_record(prefix="train_"),
        }

    def evaluate(self, mask: torch.Tensor | None = None) -> float:
        """Validation score on the held-out edges (macro-F1 by default).

        Caches the full report on ``self.last_report`` and the measured inference
        cost on ``self.last_eval_usage`` for results logging.
        """
        mask = self.val_mask if mask is None else mask
        with track_resources(self.device) as res:
            preds, targets = predict_model(self.model, self.graph, mask)
            self.last_report = report_from_predictions(
                preds, targets, self.num_classes, self.metric, self.benign_class
            )
        self.last_preds, self.last_targets = preds.cpu(), targets.cpu()
        self.last_eval_usage = res.usage
        return self.last_report["score"]

    def evaluate_test(self) -> dict:
        """Full report on the held-out **test** edges; caches it on the client.

        Folds the measured inference cost into the returned report so each client's
        ``results.json["test"]`` records its test-time resource usage.
        """
        with track_resources(self.device) as res:
            preds, targets = predict_model(self.model, self.graph, self.test_mask)
            self.test_report = report_from_predictions(
                preds, targets, self.num_classes, self.metric, self.benign_class
            )
        self.test_preds, self.test_targets = preds.cpu(), targets.cpu()
        self.last_eval_usage = res.usage
        self.test_report.update(res.usage.as_record(prefix="inference_"))
        return self.test_report

    @torch.no_grad()
    def hub_embedding(self) -> torch.Tensor:
        """The embedding of the hub IP node (CPU). Kept for logging/inspection."""
        self.model.eval()
        z = self.model.encode(self.graph.x, self.graph.edge_index)
        return z[self.hub_index].detach().cpu().clone()

    @torch.no_grad()
    def community_embeddings(self) -> torch.Tensor:
        """One mean-pooled node embedding per detected community (CPU).

        Runs the current encoder over the graph, then average-pools the node
        embeddings within each precomputed community. Shape ``[n_communities,
        embed_dim]`` — the *community abstraction* uploaded to the server.
        """
        self.model.eval()
        z = self.model.encode(self.graph.x, self.graph.edge_index)
        pooled = torch.stack([z[idx].mean(dim=0) for idx in self._community_index])
        return pooled.detach().cpu().clone()

    # extraction API

    def extract_payload(self) -> ClientPayload:
        """Bundle weights + community embeddings + scores for the server."""
        val_score = self.evaluate()  # --metric score on val (also fills last_report)
        # The FedAvg weight is driven by attack-detection skill, NOT the --metric:
        # under the non-IID split a benign-only client trivially maxes balanced
        # accuracy and would dominate the merge (pulling the global model to
        # "always benign"). macro_attack_recall is 0 when a client sees no attacks,
        # so such a client correctly falls to the alpha floor weight.
        agg_score = float(self.last_report["macro_attack_recall"])
        return ClientPayload(
            client_id=self.client_id,
            state_dict=self.get_weights(),
            community_embeddings=self.community_embeddings(),
            community_sizes=self.community_sizes,
            val_score=val_score,
            agg_score=agg_score,
            num_train_edges=int(self.train_mask.sum()),
        )

    # artifact persistence (Models/client_NN, Results/client_NN)

    @property
    def name(self) -> str:
        return f"client_{self.client_id:02d}"

    def save_model(
        self, models_dir: str | Path, state_dict: dict[str, torch.Tensor] | None = None
    ) -> Path:
        """Persist the client's local model to ``Models/client_NN/local_model.pt``."""
        out_dir = Path(models_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "local_model.pt"
        torch.save(state_dict if state_dict is not None else self.get_weights(), path)
        return path

    def save_results(self, results_dir: str | Path, rounds: list[dict]) -> Path:
        """client's per-round results to ``Results/client_NN/results.json``."""
        out_dir = Path(results_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "results.json"
        payload = {
            "client_id": self.client_id,
            "hub_ip": self.hub_ip,
            "graph": {
                "n_nodes": int(self.graph.num_nodes),
                "n_edges": int(self.graph.edge_index.shape[1]),
                "n_train_edges": int(self.train_mask.sum()),
                "n_val_edges": int(self.val_mask.sum()),
                "n_test_edges": int(self.test_mask.sum()),
                "split": self.split_kind,
                "n_communities": len(self.communities),
                "community_sizes": self.community_sizes,
            },
            "model_params": self.num_params,
            "rounds": rounds,
        }
        if self.test_report is not None:
            payload["test"] = self.test_report
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load_results(self, results_dir: str | Path) -> list[dict]:
        """Load previously-saved per-round records (for resumption)."""
        path = Path(results_dir) / self.name / "results.json"
        if not path.exists():
            return []
        return json.loads(path.read_text()).get("rounds", [])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FederatedClient(id={self.client_id:02d}, hub={self.hub_ip}, "
            f"nodes={self.graph.num_nodes}, edges={self.graph.edge_index.shape[1]}, "
            f"params={self.num_params})"
        )
