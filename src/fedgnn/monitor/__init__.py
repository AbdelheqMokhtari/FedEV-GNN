"""Monitoring layer: per-client resource profiling for FedEV-GNN.

Measures what it costs to run each federated client — peak RAM (host + GPU),
training FLOPs, and model size / parameter breakdown — by building the client and
running one real training step.

* :mod:`fedgnn.monitor.profiler` — :func:`profile_client` + helpers.
* :mod:`fedgnn.monitor.cli` — the ``fedgnn-monitor`` orchestrator.
"""

from fedgnn.monitor.profiler import parameter_breakdown, profile_client

__all__ = ["parameter_breakdown", "profile_client"]
