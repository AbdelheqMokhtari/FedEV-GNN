"""Recompute held-out confusion matrices for runs trained before CMs were stored.

Experiments trained before the confusion matrix was added to the evaluation report
have no ``confusion_matrix`` block in their ``Results/``. This module reloads the
best global model and **replays each client's deterministic test split** to recover
the matrices on demand, so the validation figures work on legacy runs too.

It is the one place in ``fedgnn.validation`` that needs the ``gnn`` extra (torch /
torch-geometric) and the ``data/clients/`` graphs — imported lazily, so the normal
stored-CM path stays pure-JSON. Nothing is written back to disk; the recomputed
matrices are injected into the in-memory :class:`ExperimentResults` only.
"""

from __future__ import annotations

from pathlib import Path

from fedgnn.validation.loader import ExperimentResults


def recompute_confusions(
    exp: ExperimentResults,
    models_root: str | Path = "Models",
    clients_root: str | Path = "data/clients",
    device: str = "cpu",
) -> bool:
    """Fill in missing per-client + aggregate test CMs by replaying inference.

    Reconstructs each client with the **exact stored config** (so the deterministic
    split — seeded per client — matches training), loads the best (else latest)
    global checkpoint, and scores the held-out test edges. Returns ``True`` if any
    matrices were computed and injected into ``exp``; ``False`` if the inputs
    (config, checkpoint, or client graphs) are unavailable or the ``gnn`` extra is
    not installed.
    """
    config = exp.metadata.get("config", {})
    num_classes = exp.num_classes or int(config.get("num_classes", 0))
    if not num_classes:
        return False

    server_dir = Path(models_root) / exp.name / "server"
    model_path = server_dir / "best_global_model.pt"
    if not model_path.exists():
        model_path = server_dir / "latest_global_model.pt"
    if not model_path.exists():
        return False

    client_dirs = sorted(Path(clients_root).glob("client_*"))
    if not client_dirs:
        return False

    try:  # the only torch-dependent path in the validation package
        import torch

        from fedgnn.train.client import FederatedClient
    except Exception:
        return False

    state = torch.load(model_path, weights_only=False)["model_state"]
    arch = config.get("architecture", {})
    benign_class = exp.benign_class or int(config.get("benign_class", 0))
    have = {c.name for c in exp.clients}

    agg = [[0] * num_classes for _ in range(num_classes)]
    by_name: dict[str, list[list[int]]] = {}
    for d in client_dirs:
        if have and d.name not in have:
            continue
        try:
            fc = FederatedClient(
                d,
                num_classes=num_classes,
                device=device,
                val_ratio=config.get("val_ratio", 0.15),
                test_ratio=config.get("test_ratio", 0.15),
                split=config.get("split", "stratified"),
                dropout=arch.get("dropout", 0.5),
                hidden_per_head=arch.get("hidden_per_head", 64),
                heads=arch.get("heads", 4),
                num_layers=arch.get("num_layers", 3),
                embed_dim=arch.get("embed_dim", 32),
                louvain_resolution=config.get("louvain_resolution", 1.0),
                metric=exp.selection_metric,
                benign_class=benign_class,
                seed=config.get("seed", 42),
            )
            fc.set_weights(state)
            cm = fc.evaluate_test()["confusion_matrix"]
        except Exception:
            continue
        by_name[d.name] = cm
        for i in range(num_classes):
            for j in range(num_classes):
                agg[i][j] += cm[i][j]

    if not by_name:
        return False

    for c in exp.clients:
        if c.name in by_name:
            c.test = dict(c.test or {})
            c.test["confusion_matrix"] = by_name[c.name]
    exp.test_confusion = agg
    return True
