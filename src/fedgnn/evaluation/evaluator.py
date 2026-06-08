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
    confusion_matrix,
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
def predict_model(
    model: nn.Module, graph: Data, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass -> ``(preds, targets)`` for the masked edges (raw class ids).

    Separated from scoring so the caller can **pool** predictions from several
    graphs (e.g. all federated clients) into one set before computing a single,
    honest multi-class report — see :func:`report_from_predictions`.
    """
    model.eval()
    logits, _ = model(graph.x, graph.edge_index)
    return logits[mask].argmax(dim=1), graph.y[mask]


def report_from_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    metric: str = "balanced_accuracy",
    benign_class: int = 0,
) -> dict:
    """Build the full metric report from already-computed predictions/targets.

    Returns accuracy, macro-F1, balanced accuracy (macro recall), the two
    attack-recall views, a ``per_class`` breakdown, the confusion matrix, the
    evaluated-edge count, and a ``score`` alias for the selected ``metric``.
    """
    if metric not in SELECTABLE_METRICS:
        raise ValueError(f"metric must be one of {SELECTABLE_METRICS}, got {metric!r}")

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
        "confusion_matrix": confusion_matrix(preds, targets, num_classes),
        "n_eval_edges": int(preds.numel()),
    }
    report["score"] = report[metric]
    return report


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    graph: Data,
    mask: torch.Tensor,
    num_classes: int,
    metric: str = "balanced_accuracy",
    benign_class: int = 0,
) -> dict:
    """Score ``model`` on the masked edges of ``graph`` (predict + report)."""
    preds, targets = predict_model(model, graph, mask)
    return report_from_predictions(preds, targets, num_classes, metric, benign_class)
