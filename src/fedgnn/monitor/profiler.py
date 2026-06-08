"""Resource profiling for one federated client: RAM, compute, and model size.

Answers "what does it cost to run this client?" by actually **building the client
and running one real training step**, then reporting:

* **model** — parameter count, a per-module breakdown, the fp32 weight size, and
  the training-state footprint (weights + gradients + Adam moments ≈ 16 B/param);
* **data** — the graph's resident tensor size (``x``, ``edge_index``, ``edge_attr``,
  ``y``);
* **memory** — the measured **peak host RAM** the step needed (and **peak GPU
  memory** when run on CUDA, which is exact via the allocator's peak stats);
* **compute** — the FLOPs of one full-batch forward+backward (``flops_per_epoch``)
  and per round (``× local_epochs``), reusing the same counter the trainer logs.

Needs the ``gnn`` extra (torch / torch-geometric). Measurement notes: GPU peak is
exact (``reset_peak_memory_stats`` per call); host-RAM peak is the **increment**
over the process baseline, sampled in a background thread — for an absolute
per-client number profile a single client (``--client``) in a fresh process, since
one process's allocator caches blocks across clients.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

MB = 1024 * 1024
_PAGE = (
    os.sysconf("SC_PAGE_SIZE")
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in getattr(os, "sysconf_names", {})
    else 4096
)


def _rss_bytes() -> int:
    """Resident set size of this process in bytes (Linux /proc; resource fallback)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * _PAGE
    except Exception:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            return 0


class _PeakRSS:
    """Context manager sampling process RSS in a thread; exposes the peak seen."""

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.peak = _rss_bytes()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_bytes())
            time.sleep(self.interval)

    def __enter__(self) -> "_PeakRSS":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, _rss_bytes())


def _tensor_mb(*tensors) -> float:
    total = 0
    for t in tensors:
        if t is not None:
            total += t.element_size() * t.nelement()
    return total / MB


def parameter_breakdown(model) -> list[dict]:
    """Parameter count per top-level sub-module (e.g. convs / norms / classifier)."""
    groups: OrderedDict[str, int] = OrderedDict()
    for pname, p in model.named_parameters():
        top = pname.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    return [{"module": k, "params": v} for k, v in groups.items()]


def profile_client(
    client_dir: str | Path,
    *,
    num_classes: int,
    device: str = "cpu",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split: str = "stratified",
    dropout: float = 0.5,
    hidden_per_head: int = 64,
    heads: int = 4,
    num_layers: int = 3,
    embed_dim: int = 32,
    louvain_resolution: float = 1.0,
    benign_class: int = 0,
    seed: int = 42,
    local_epochs: int = 5,
) -> dict:
    """Build a client, run one training step, and return a resource report dict."""
    import torch

    from fedgnn.train.client import FederatedClient

    dev = torch.device(device)
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    base_rss = _rss_bytes()

    with _PeakRSS() as peak:
        fc = FederatedClient(
            client_dir,
            num_classes=num_classes,
            device=dev,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split=split,
            dropout=dropout,
            hidden_per_head=hidden_per_head,
            heads=heads,
            num_layers=num_layers,
            embed_dim=embed_dim,
            louvain_resolution=louvain_resolution,
            benign_class=benign_class,
            seed=seed,
        )
        info = fc.train(1)  # one real fwd+bwd step → FLOPs + exercises peak memory

    g = fc.graph
    n_params = fc.num_params
    flops_epoch = info["flops_per_epoch"]
    report = {
        "client": fc.name,
        "hub_ip": fc.hub_ip,
        "device": dev.type,
        "n_nodes": int(g.num_nodes),
        "n_edges": int(g.edge_index.shape[1]),
        "in_features": int(g.x.shape[1]),
        "n_train_edges": int(fc.train_mask.sum()),
        "n_val_edges": int(fc.val_mask.sum()),
        "n_test_edges": int(fc.test_mask.sum()),
        "n_communities": len(fc.communities),
        "model_params": n_params,
        "model_mb": round(n_params * 4 / MB, 3),
        "train_state_mb": round(n_params * 16 / MB, 3),  # weights+grads+2×Adam
        "param_breakdown": parameter_breakdown(fc.model),
        "graph_tensors_mb": round(_tensor_mb(g.x, g.edge_index, g.edge_attr, g.y), 3),
        "host_ram_peak_mb": round(max(0.0, (peak.peak - base_rss) / MB), 1),
        "flops_per_epoch": flops_epoch,
        "flops_per_round": flops_epoch * local_epochs,
    }
    if dev.type == "cuda":
        report["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated(dev) / MB, 1)

    del fc
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return report
