"""Evaluate a trained graph model on a held-out set of edges.

Centralises the forward+score path so both the client (local validation) and the
CLI (global-model evaluation) produce identical, comparable reports.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from fedgnn.evaluation.metrics import (
    accuracy,
    attack_detection_recall,
    balanced_accuracy,
    macro_attack_recall,
    macro_f1,
    per_class_report,
)

# Metrics that may drive best-model selection (the `score` alias).
SELECTABLE_METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "macro_attack_recall",
    "attack_detection_recall",
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    graph: Data,
    mask: torch.Tensor,
    num_classes: int,
    metric: str = "balanced_accuracy",
    benign_class: int = 0,
) -> dict:
    """Score ``model`` on the masked edges of ``graph``.

    Returns a report with accuracy, macro-F1, balanced accuracy (macro recall),
    the two attack-recall views, a ``per_class`` precision/recall/f1/support
    breakdown, the evaluated-edge count, and a ``score`` alias for the selected
    ``metric`` (the scalar used for weighting / best-model tracking).
    """
    if metric not in SELECTABLE_METRICS:
        raise ValueError(f"metric must be one of {SELECTABLE_METRICS}, got {metric!r}")

    model.eval()
    logits, _ = model(graph.x, graph.edge_index)
    preds = logits[mask].argmax(dim=1)
    targets = graph.y[mask]

    report = {
        "accuracy": accuracy(preds, targets),
        "macro_f1": macro_f1(preds, targets, num_classes),
        "balanced_accuracy": balanced_accuracy(preds, targets, num_classes),
        "macro_attack_recall": macro_attack_recall(
            preds, targets, num_classes, benign_class
        ),
        "attack_detection_recall": attack_detection_recall(
            preds, targets, benign_class
        ),
        "per_class": per_class_report(preds, targets, num_classes),
        "n_eval_edges": int(mask.sum()),
    }
    report["score"] = report[metric]
    return report
