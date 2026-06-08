"""Validation & visualisation layer for FedEV-GNN.

Reads the JSON artifacts a federated run leaves under ``Results/<exp>/`` and turns
them into evaluation figures under ``Figures/<exp>/``:

* :mod:`fedgnn.validation.loader` — re-assembles ``metadata.json``,
  ``server/training_state.json``, every ``client_NN/results.json`` and
  ``test_results.json`` into one :class:`ExperimentResults` object.
* :mod:`fedgnn.validation.plots` — matplotlib renderers: per-client loss curves,
  per-client/global quality bar charts, global-metric and aggregation-weight
  evolution.
* :mod:`fedgnn.validation.cli` — the ``fedgnn-validate`` orchestrator.
"""

from fedgnn.validation.loader import (
    ClientResults,
    ExperimentResults,
    binary_confusion,
    discover_experiments,
    latest_experiment,
    load_experiment,
)
from fedgnn.validation.plots import render_all

__all__ = [
    "ClientResults",
    "ExperimentResults",
    "binary_confusion",
    "discover_experiments",
    "latest_experiment",
    "load_experiment",
    "render_all",
]
