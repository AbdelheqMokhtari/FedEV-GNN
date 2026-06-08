"""Classification metrics computed directly on tensors (no scikit-learn).

All functions take 1-D integer tensors of predictions and targets so the
evaluation package only needs the ``gnn`` extra (torch / torch-geometric).
"""

from __future__ import annotations

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Plain top-1 accuracy."""
    if targets.numel() == 0:
        return 0.0
    return (preds == targets).float().mean().item()


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged F1 over classes with positive support in ``targets``.

    Averaging only over present classes matches scikit-learn's behaviour for
    absent labels — the right default for the heavily imbalanced ToN-IoT
    partitions, where most clients never see several attack classes.
    """
    if targets.numel() == 0:
        return 0.0

    f1s: list[float] = []
    for c in range(num_classes):
        if int((targets == c).sum()) == 0:
            continue
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        f1s.append(_prf(tp, fp, fn)[2])

    return sum(f1s) / len(f1s) if f1s else 0.0


def recall_macro(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    classes: list[int] | None = None,
) -> float:
    """Macro-averaged recall over ``classes`` (default: all present classes).

    Recall = TP / (TP + FN) per class, averaged unweighted — so a rare attack
    class counts as much as the dominant benign traffic.
    """
    if targets.numel() == 0:
        return 0.0
    candidates = range(num_classes) if classes is None else classes

    recalls: list[float] = []
    for c in candidates:
        support = int((targets == c).sum())
        if support == 0:
            continue
        tp = int(((preds == c) & (targets == c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    return sum(recalls) / len(recalls) if recalls else 0.0


def balanced_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    """Balanced accuracy = mean per-class recall over all present classes.

    Unlike plain accuracy it is not dominated by the majority (benign/DDoS)
    flows, so it reflects whether the model handles every class — the right
    headline number for the heavily imbalanced IDS partitions.
    """
    return recall_macro(preds, targets, num_classes)


def macro_attack_recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    benign_class: int = 0,
) -> float:
    """Mean recall across the *attack* classes only (every benign-vs-X excluded).

    This is the metric to watch for an IDS: it asks "of each attack type, what
    fraction did we catch?" and weights every attack type equally, so missing a
    rare attack (mitm, injection) hurts as much as missing a common one.
    """
    classes = [c for c in range(num_classes) if c != benign_class]
    return recall_macro(preds, targets, num_classes, classes)


def attack_detection_recall(
    preds: torch.Tensor, targets: torch.Tensor, benign_class: int = 0
) -> float:
    """Binary attack-detection recall: of all true attack flows, the fraction
    flagged as *some* attack (i.e. not predicted benign).

    The coarse "did we raise an alarm at all?" signal — the cost of a missed
    attack is far higher than misclassifying which attack it was.
    """
    is_attack = targets != benign_class
    n_attacks = int(is_attack.sum())
    if n_attacks == 0:
        return 0.0
    detected = int((is_attack & (preds != benign_class)).sum())
    return detected / n_attacks


def confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> list[list[int]]:
    """Confusion matrix as nested integer lists (JSON-friendly).

    ``cm[i][j]`` = number of edges whose **true** label is ``i`` and whose
    **predicted** label is ``j``. So rows are indexed by truth and columns by
    prediction: each row sums to that class's support, and the diagonal holds the
    correct hits. Built with a single ``bincount`` over the flattened ``i*C+j``
    indices (no scikit-learn, stays on the ``gnn`` extra only).
    """
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long)
    if targets.numel():
        flat = targets.to(torch.long) * num_classes + preds.to(torch.long)
        cm = torch.bincount(flat, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).tolist()


def per_class_report(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> dict[int, dict[str, float]]:
    """Per-class precision / recall / F1 / support (keyed by class index)."""
    report: dict[int, dict[str, float]] = {}
    for c in range(num_classes):
        support = int((targets == c).sum())
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        report[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return report
