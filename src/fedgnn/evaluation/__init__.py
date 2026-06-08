"""Evaluation utilities for FedEV-GNN."""

from fedgnn.evaluation.evaluator import (
    SELECTABLE_METRICS,
    evaluate_model,
    predict_model,
    report_from_predictions,
)
from fedgnn.evaluation.flops import count_parameters, count_training_flops
from fedgnn.evaluation.metrics import (
    accuracy,
    attack_detection_recall,
    balanced_accuracy,
    confusion_matrix,
    macro_attack_recall,
    macro_f1,
    per_class_report,
    recall_macro,
)
from fedgnn.evaluation.resources import ResourceUsage, rss_mb, track_resources

__all__ = [
    "SELECTABLE_METRICS",
    "ResourceUsage",
    "accuracy",
    "attack_detection_recall",
    "balanced_accuracy",
    "confusion_matrix",
    "count_parameters",
    "count_training_flops",
    "evaluate_model",
    "macro_attack_recall",
    "macro_f1",
    "per_class_report",
    "predict_model",
    "recall_macro",
    "report_from_predictions",
    "rss_mb",
    "track_resources",
]
