"""Matplotlib renderers turning an :class:`ExperimentResults` into figures.

Each ``plot_*`` function takes the parsed experiment plus a destination directory,
writes one figure, and returns its :class:`pathlib.Path` (or ``None`` if there was
nothing to draw — e.g. a server figure for a run with no history yet). The CLI
orchestrates them into the on-disk layout::

    Figures/<exp>/
      loss_all_clients.<fmt>          every client's training loss, overlaid
      client_quality.<fmt>            grouped bars — how good each client is
      clients/
        client_NN_loss.<fmt>          one client's loss curve
        client_NN_metrics.<fmt>       one client's local-vs-global metric trends
      server/
        global_metrics_evolution.<fmt>  the four headline metrics per round
        server_quality.<fmt>            bars — how good the global model is
        aggregation_weights.<fmt>       each client's FedAvg weight per round

The module forces matplotlib's non-interactive ``Agg`` backend on import so it
renders headless (CI, servers) without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: render straight to files, never a window

import matplotlib.pyplot as plt  # noqa: E402

from fedgnn.validation.loader import (  # noqa: E402
    HEADLINE_METRICS,
    TEST_METRICS,
    ClientResults,
    ExperimentResults,
    binary_confusion,
)

# Pretty labels for the metric keys used across the figures.
METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "balanced_accuracy": "Balanced Acc.",
    "macro_attack_recall": "Attack Recall",
    "attack_detection_recall": "Detection Recall",
    "accuracy": "Accuracy",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _save(fig: plt.Figure, out_dir: Path, stem: str, fmt: str, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _clean(values: list) -> tuple[list[int], list[float]]:
    """Drop ``None`` holes, returning aligned (index, value) lists."""
    idx, vals = [], []
    for i, v in enumerate(values):
        if v is not None:
            idx.append(i)
            vals.append(v)
    return idx, vals


# ── per-client figures ──────────────────────────────────────────────────────


def plot_client_loss(
    client: ClientResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """One client's training loss (``final_loss``) across rounds."""
    rounds = client.round_index
    losses = client.series("final_loss")
    pos, vals = _clean(losses)
    if not vals:
        return None
    xs = [rounds[i] for i in pos]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, vals, marker="o", ms=4, color="#c0392b", lw=1.8)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Training loss (final epoch)")
    ax.set_title(f"{client.name} — local training loss  (hub {client.hub_ip})")
    return _save(fig, out_dir, f"{client.name}_loss", fmt, dpi)


def plot_client_metrics(
    client: ClientResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """One client's local vs global headline metrics across rounds."""
    rounds = client.round_index
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    drew = False
    for ax, metric in zip(axes.flat, HEADLINE_METRICS):
        lpos, lvals = _clean(client.series(f"local_{metric}"))
        gpos, gvals = _clean(client.series(f"global_{metric}"))
        if lvals:
            ax.plot(
                [rounds[i] for i in lpos],
                lvals,
                marker="o",
                ms=3,
                lw=1.5,
                label="local",
                color="#2980b9",
            )
            drew = True
        if gvals:
            ax.plot(
                [rounds[i] for i in gpos],
                gvals,
                marker="s",
                ms=3,
                lw=1.5,
                label="global",
                color="#27ae60",
            )
            drew = True
        ax.set_title(_label(metric))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="best")
    if not drew:
        plt.close(fig)
        return None
    for ax in axes[-1]:
        ax.set_xlabel("Federated round")
    fig.suptitle(
        f"{client.name} — local vs global metrics per round  (hub {client.hub_ip})"
    )
    return _save(fig, out_dir, f"{client.name}_metrics", fmt, dpi)


# ── experiment-level figures ────────────────────────────────────────────────


def plot_loss_all_clients(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """Every client's training-loss curve overlaid on one axis."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    drew = False
    for k, client in enumerate(exp.clients):
        rounds = client.round_index
        pos, vals = _clean(client.series("final_loss"))
        if not vals:
            continue
        ax.plot(
            [rounds[i] for i in pos],
            vals,
            marker="o",
            ms=3,
            lw=1.5,
            color=cmap(k % 10),
            label=client.name,
        )
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Training loss (final epoch)")
    ax.set_title(f"{exp.name} — local training loss across clients")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    return _save(fig, out_dir, "loss_all_clients", fmt, dpi)


def plot_client_quality(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """Grouped bars: each client's quality across the headline metrics.

    Uses held-out **test** metrics when available, else last-round **global**
    metrics (noted in the title so the two are never confused).
    """
    quality = exp.client_quality()
    if not quality:
        return None
    names = list(quality)
    metrics = TEST_METRICS if exp.has_test else HEADLINE_METRICS
    source = "held-out test" if exp.has_test else "last-round global (no test pass)"

    n_groups = len(names)
    n_bars = len(metrics)
    width = 0.8 / n_bars
    x = range(n_groups)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(max(9, n_groups * 1.1), 5.5))
    for b, metric in enumerate(metrics):
        offsets = [i + (b - (n_bars - 1) / 2) * width for i in x]
        vals = [quality[name].get(metric, 0.0) for name in names]
        ax.bar(
            offsets,
            vals,
            width=width,
            label=_label(metric),
            color=cmap(b / max(1, n_bars - 1)),
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{exp.name} — per-client model quality  ({source})")
    ax.legend(ncol=n_bars, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _save(fig, out_dir, "client_quality", fmt, dpi)


# ── server figures ──────────────────────────────────────────────────────────


def plot_global_metrics_evolution(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """The four headline global metrics per round, from the server history."""
    if not exp.history:
        return None
    rounds = exp.round_index
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    drew = False
    for k, metric in enumerate(HEADLINE_METRICS):
        pos, vals = _clean(exp.global_series(metric))
        if not vals:
            continue
        ax.plot(
            [rounds[i] for i in pos],
            vals,
            marker="o",
            ms=4,
            lw=1.8,
            color=cmap(k),
            label=_label(metric),
        )
        drew = True
    if not drew:
        plt.close(fig)
        return None
    best_round = exp.metadata.get("best_round")
    if best_round is not None and best_round in rounds:
        ax.axvline(
            best_round,
            color="grey",
            ls="--",
            lw=1,
            label=f"best ({exp.selection_metric}) @ r{best_round}",
        )
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Score (mean over clients' val splits)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{exp.name} — global model metrics per round")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, out_dir, "global_metrics_evolution", fmt, dpi)


def plot_server_quality(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """Bars: the global model's aggregate quality across metrics."""
    if exp.has_test:
        scores = exp.test_aggregate
        metrics = [m for m in TEST_METRICS if m in scores]
        source = f"held-out test ({exp.evaluated_model})"
    elif exp.history:
        last = exp.history[-1].get("global_metrics", {})
        scores = last
        metrics = [m for m in HEADLINE_METRICS if m in scores]
        source = f"last-round global (round {exp.history[-1].get('round')})"
    else:
        return None
    if not metrics:
        return None

    labels = [_label(m) for m in metrics]
    vals = [scores[m] for m in metrics]
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(
        labels,
        vals,
        color=[cmap(i / max(1, len(metrics) - 1)) for i in range(len(metrics))],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{exp.name} — global (server) model quality\n{source}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out_dir, "server_quality", fmt, dpi)


def plot_aggregation_weights(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """Each client's normalised FedAvg weight per round (stacked area).

    Shows how the Adaptive Weighted FedAvg influence shifts between clients over
    training — bands summing to 1.0 each round.
    """
    if not exp.history:
        return None
    rounds = exp.round_index
    client_ids = exp.history[0].get("client_ids", [])
    if not client_ids:
        return None
    # weights[round][client] -> series per client, aligned to client_ids order
    series = {cid: [] for cid in client_ids}
    for h in exp.history:
        ids = h.get("client_ids", [])
        w = h.get("w_k_norm", [])
        lookup = dict(zip(ids, w))
        for cid in client_ids:
            series[cid].append(lookup.get(cid, 0.0))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    ax.stackplot(
        rounds,
        *[series[cid] for cid in client_ids],
        labels=[f"client_{cid:02d}" for cid in client_ids],
        colors=[cmap(i % 10) for i in range(len(client_ids))],
        alpha=0.85,
    )
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Normalised aggregation weight")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min(rounds), max(rounds))
    ax.set_title(f"{exp.name} — Adaptive FedAvg weight share per client")
    ax.legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(False)
    return _save(fig, out_dir, "aggregation_weights", fmt, dpi)


# ── confusion matrices ──────────────────────────────────────────────────────


def _involved_classes(cm: list[list[int]]) -> list[int]:
    """Class indices that appear as a true row *or* a predicted column (non-empty).

    Keeps each matrix compact: a client that only ever sees 4 classes draws a 4×4,
    not a sparse 10×10. Always square (same indices on both axes) and never empty.
    """
    arr = np.asarray(cm)
    rows = set(np.where(arr.sum(axis=1) > 0)[0].tolist())
    cols = set(np.where(arr.sum(axis=0) > 0)[0].tolist())
    idxs = sorted(rows | cols)
    return idxs or list(range(len(cm)))


def _draw_cm(ax, cm: list[list[int]], labels: list[str], title: str) -> None:
    """Heatmap of one confusion matrix: colour = row-normalised recall, text = count.

    Colouring by recall (each row ÷ its support) makes the diagonal readable even
    when class sizes span orders of magnitude — a rare attack with 100% recall is
    as dark as the benign block. The raw counts are annotated in each cell.
    """
    arr = np.asarray(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    norm = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0)

    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall (row-norm.)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            count = int(arr[i, j])
            if count == 0:
                continue
            ax.text(
                j,
                i,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.grid(False)


def plot_client_confusion(
    client: ClientResults,
    exp: ExperimentResults,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 120,
) -> Path | None:
    """One client's held-out **multiclass** confusion matrix (per attack type)."""
    cm = client.confusion_matrix
    if not cm:
        return None
    idxs = _involved_classes(cm)
    sub = [[cm[i][j] for j in idxs] for i in idxs]
    labels = [exp.class_label(i) for i in idxs]

    fig, ax = plt.subplots(figsize=(max(6, len(idxs) * 0.9), max(5, len(idxs) * 0.8)))
    _draw_cm(ax, sub, labels, f"{client.name} — confusion (hub {client.hub_ip})")
    return _save(fig, out_dir, f"{client.name}_confusion", fmt, dpi)


def plot_client_confusion_binary(
    client: ClientResults,
    exp: ExperimentResults,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 120,
) -> Path | None:
    """One client's **binary** (benign vs attack) confusion matrix."""
    cm = client.confusion_matrix
    if not cm:
        return None
    bcm = binary_confusion(cm, exp.benign_class)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    _draw_cm(ax, bcm, ["benign", "attack"], f"{client.name} — benign vs attack")
    return _save(fig, out_dir, f"{client.name}_confusion_binary", fmt, dpi)


def plot_server_confusion(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """Aggregate (all clients summed) **multiclass** confusion matrix."""
    cm = exp.test_confusion
    if not cm:
        return None
    idxs = _involved_classes(cm)
    sub = [[cm[i][j] for j in idxs] for i in idxs]
    labels = [exp.class_label(i) for i in idxs]

    fig, ax = plt.subplots(figsize=(max(6, len(idxs) * 0.9), max(5, len(idxs) * 0.8)))
    _draw_cm(ax, sub, labels, f"{exp.name} — global model confusion (all clients)")
    return _save(fig, out_dir, "confusion", fmt, dpi)


def plot_server_confusion_binary(
    exp: ExperimentResults, out_dir: Path, fmt: str = "png", dpi: int = 120
) -> Path | None:
    """Aggregate **binary** (benign vs attack) confusion matrix."""
    cm = exp.test_confusion
    if not cm:
        return None
    bcm = binary_confusion(cm, exp.benign_class)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    _draw_cm(ax, bcm, ["benign", "attack"], f"{exp.name} — global benign vs attack")
    return _save(fig, out_dir, "confusion_binary", fmt, dpi)


# ── orchestration ───────────────────────────────────────────────────────────


def render_all(
    exp: ExperimentResults,
    figures_root: str | Path = "Figures",
    fmt: str = "png",
    dpi: int = 120,
) -> list[Path]:
    """Render the full figure set for ``exp`` into ``Figures/<exp>/``.

    Returns every path written. Sub-folders ``clients/`` and ``server/`` hold the
    per-client and server figures; experiment-wide summaries sit at the top level.
    """
    base = Path(figures_root) / exp.name
    clients_dir = base / "clients"
    server_dir = base / "server"
    written: list[Path] = []

    def _add(path: Path | None) -> None:
        if path is not None:
            written.append(path)

    # experiment-level summaries
    _add(plot_loss_all_clients(exp, base, fmt, dpi))
    _add(plot_client_quality(exp, base, fmt, dpi))

    # per-client
    for client in exp.clients:
        _add(plot_client_loss(client, clients_dir, fmt, dpi))
        _add(plot_client_metrics(client, clients_dir, fmt, dpi))
        _add(plot_client_confusion(client, exp, clients_dir, fmt, dpi))
        _add(plot_client_confusion_binary(client, exp, clients_dir, fmt, dpi))

    # server / global model
    _add(plot_global_metrics_evolution(exp, server_dir, fmt, dpi))
    _add(plot_server_quality(exp, server_dir, fmt, dpi))
    _add(plot_aggregation_weights(exp, server_dir, fmt, dpi))
    _add(plot_server_confusion(exp, server_dir, fmt, dpi))
    _add(plot_server_confusion_binary(exp, server_dir, fmt, dpi))

    return written
