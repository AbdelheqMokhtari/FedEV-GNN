"""Read a finished (or in-progress) experiment back from ``Results/<exp>/``.

The training pipeline (``fedgnn.train``) scatters its record of a run across
several JSON files inside one experiment subtree::

    Results/<exp>/metadata.json              run-level provenance + config
    Results/<exp>/server/training_state.json per-round global history
    Results/<exp>/client_NN/results.json     per-client per-round records
    Results/<exp>/test_results.json          held-out test metrics (if completed)

:class:`ExperimentResults` re-assembles those four sources into a single,
plot-friendly object so the visualisation layer (``fedgnn.validation.plots``)
never has to know the on-disk layout. It is deliberately tolerant of a missing
``test_results.json`` (an interrupted run never wrote one) so partial runs can
still be visualised — :attr:`ExperimentResults.has_test` says whether the
held-out evaluation is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# The four headline metrics logged every round for both the local and global
# views, in a stable display order. ``accuracy`` only exists in the test block.
HEADLINE_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_attack_recall",
    "attack_detection_recall",
)
TEST_METRICS = HEADLINE_METRICS + ("accuracy",)


@dataclass
class ClientResults:
    """One client's record, parsed from ``client_NN/results.json``."""

    name: str  # "client_00"
    client_id: int
    hub_ip: str
    model_params: int
    graph: dict
    rounds: list[dict]
    test: dict | None = None  # filled from test_results.json when present

    @property
    def round_index(self) -> list[int]:
        return [r["round"] for r in self.rounds]

    def series(self, key: str) -> list[float]:
        """Per-round values for ``key`` (e.g. ``final_loss``, ``global_macro_f1``)."""
        return [r.get(key) for r in self.rounds]

    @property
    def confusion_matrix(self) -> list[list[int]] | None:
        """Held-out test confusion matrix (row=true, col=pred), if recorded."""
        if self.test and self.test.get("confusion_matrix"):
            return self.test["confusion_matrix"]
        return None


@dataclass
class ExperimentResults:
    """All artifacts of a single experiment, re-assembled for plotting."""

    name: str
    results_dir: Path
    metadata: dict
    history: list[dict]  # server training_state["history"]
    clients: list[ClientResults]
    test_aggregate: dict = field(default_factory=dict)
    selection_metric: str = "balanced_accuracy"
    evaluated_model: str = ""
    num_classes: int = 0
    benign_class: int = 0
    label_names: list[str] = field(default_factory=list)
    test_confusion: list[list[int]] | None = None  # aggregate, row=true col=pred

    # ── derived convenience views ───────────────────────────────────────────

    @property
    def has_test(self) -> bool:
        """True if the held-out test evaluation ran (clean completion)."""
        return bool(self.test_aggregate)

    @property
    def has_confusion(self) -> bool:
        """True if at least one client (or the aggregate) has a stored CM."""
        return self.test_confusion is not None or any(
            c.confusion_matrix is not None for c in self.clients
        )

    def class_label(self, idx: int) -> str:
        """Display name for a class index (falls back to the index itself)."""
        if 0 <= idx < len(self.label_names):
            return self.label_names[idx]
        return str(idx)

    @property
    def n_rounds(self) -> int:
        return len(self.history)

    @property
    def round_index(self) -> list[int]:
        return [h["round"] for h in self.history]

    def global_series(self, metric: str) -> list[float]:
        """Per-round averaged global metric from the server history."""
        return [h.get("global_metrics", {}).get(metric) for h in self.history]

    @property
    def global_score_series(self) -> list[float]:
        return [h.get("global_score") for h in self.history]

    def client_quality(self) -> dict[str, dict[str, float]]:
        """Best available "how good is each client" table.

        Prefers the honest held-out **test** metrics; falls back to each client's
        **last-round global** metrics for an interrupted run that never reached the
        test pass. Returns ``{client_name: {metric: value}}``.
        """
        quality: dict[str, dict[str, float]] = {}
        for c in self.clients:
            if c.test:
                quality[c.name] = {m: c.test.get(m, 0.0) for m in TEST_METRICS}
            elif c.rounds:
                last = c.rounds[-1]
                quality[c.name] = {
                    m: last.get(f"global_{m}", 0.0) for m in HEADLINE_METRICS
                }
        return quality


def binary_confusion(cm: list[list[int]], benign_class: int = 0) -> list[list[int]]:
    """Collapse a multiclass CM into a 2×2 ``benign`` vs ``attack`` matrix.

    Rows/cols are ``[benign, attack]``::

        [[TN, FP],     TN: benign kept benign      FP: benign flagged as attack
         [FN, TP]]     FN: attack missed (→benign) TP: attack caught as some attack
    """
    n = len(cm)
    tn = cm[benign_class][benign_class]
    fp = sum(cm[benign_class][j] for j in range(n) if j != benign_class)
    fn = sum(cm[i][benign_class] for i in range(n) if i != benign_class)
    tp = sum(
        cm[i][j]
        for i in range(n)
        for j in range(n)
        if i != benign_class and j != benign_class
    )
    return [[tn, fp], [fn, tp]]


def _load_label_names(num_classes: int) -> list[str]:
    """Class index → name from ``data/processed/label_map.json`` (best-effort)."""
    path = Path("data/processed/label_map.json")
    if not path.exists() or num_classes <= 0:
        return []
    label_map = json.loads(path.read_text())  # {name: index}
    names = [str(i) for i in range(num_classes)]
    for name, idx in label_map.items():
        if 0 <= int(idx) < num_classes:
            names[int(idx)] = name
    return names


def discover_experiments(results_root: str | Path) -> list[str]:
    """Names of every experiment under ``results_root`` (have a ``metadata.json``)."""
    root = Path(results_root)
    if not root.exists():
        return []
    names = [
        p.name for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()
    ]
    return sorted(names)


def latest_experiment(results_root: str | Path) -> str | None:
    """Most recently updated experiment under ``results_root`` (by metadata mtime)."""
    root = Path(results_root)
    names = discover_experiments(root)
    if not names:
        return None
    return max(names, key=lambda n: (root / n / "metadata.json").stat().st_mtime)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_experiment(
    name: str, results_root: str | Path = "Results"
) -> ExperimentResults:
    """Re-assemble one experiment from its ``Results/<name>/`` subtree."""
    results_dir = Path(results_root) / name
    if not results_dir.exists():
        available = ", ".join(discover_experiments(results_root)) or "none"
        raise SystemExit(
            f"[validate] experiment '{name}' not found in {results_root}/.\n"
            f"  Available: {available}"
        )

    metadata = _load_json(results_dir / "metadata.json")
    training_state = _load_json(results_dir / "server" / "training_state.json")
    test_doc = _load_json(results_dir / "test_results.json")

    history = training_state.get("history", [])
    selection_metric = (
        test_doc.get("selection_metric")
        or training_state.get("selection_metric")
        or metadata.get("config", {}).get("metric", "balanced_accuracy")
    )
    per_client_test = test_doc.get("per_client", {})

    clients: list[ClientResults] = []
    for client_dir in sorted(results_dir.glob("client_*")):
        doc = _load_json(client_dir / "results.json")
        if not doc:
            continue
        name_ = client_dir.name
        clients.append(
            ClientResults(
                name=name_,
                client_id=doc.get("client_id", -1),
                hub_ip=doc.get("hub_ip", "unknown"),
                model_params=doc.get("model_params", 0),
                graph=doc.get("graph", {}),
                rounds=doc.get("rounds", []),
                # prefer the client's own "test" block; else the aggregated doc
                test=doc.get("test") or per_client_test.get(name_),
            )
        )

    if not clients and not history:
        raise SystemExit(
            f"[validate] experiment '{name}' has no usable results "
            f"(no client records and no server history in {results_dir}/)."
        )

    config = metadata.get("config", {})
    num_classes = int(config.get("num_classes", 0))
    benign_class = int(config.get("benign_class", 0))

    # aggregate CM: prefer the stored one, else sum any per-client test CMs.
    test_confusion = test_doc.get("confusion_matrix")
    if test_confusion is None and num_classes:
        per_client_cms = [c.confusion_matrix for c in clients if c.confusion_matrix]
        if per_client_cms:
            agg = [[0] * num_classes for _ in range(num_classes)]
            for cm in per_client_cms:
                for i in range(num_classes):
                    for j in range(num_classes):
                        agg[i][j] += cm[i][j]
            test_confusion = agg

    return ExperimentResults(
        name=name,
        results_dir=results_dir,
        metadata=metadata,
        history=history,
        clients=clients,
        test_aggregate=test_doc.get("aggregate", {}),
        selection_metric=selection_metric,
        evaluated_model=test_doc.get("evaluated_model", ""),
        num_classes=num_classes,
        benign_class=benign_class,
        label_names=_load_label_names(num_classes),
        test_confusion=test_confusion,
    )
