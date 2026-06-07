"""Evaluation utilities for FedEV-GNN."""

from fedgnn.evaluation.evaluator import SELECTABLE_METRICS, evaluate_model
from fedgnn.evaluation.flops import count_parameters, count_training_flops
from fedgnn.evaluation.metrics import (
    accuracy,
    attack_detection_recall,
    balanced_accuracy,
    macro_attack_recall,
    macro_f1,
    per_class_report,
    recall_macro,
)

__all__ = [
    "SELECTABLE_METRICS",
    "accuracy",
    "attack_detection_recall",
    "balanced_accuracy",
    "count_parameters",
    "count_training_flops",
    "evaluate_model",
    "macro_attack_recall",
    "macro_f1",
    "per_class_report",
    "recall_macro",
]
