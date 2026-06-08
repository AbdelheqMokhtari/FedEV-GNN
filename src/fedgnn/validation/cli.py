"""Visualisation CLI for finished (or in-progress) federated experiments.

Reads the JSON artifacts a training run leaves under ``Results/<exp>/`` and draws
the evaluation figures into ``Figures/<exp>/`` — loss curves per client, bar
charts of how good each client and the global (server) model are, and the global
metric / aggregation-weight evolution::

    fedgnn-validate list                       # show available experiments
    fedgnn-validate plot                       # latest experiment -> Figures/<latest>/
    fedgnn-validate plot fedgnn05              # a specific experiment
    fedgnn-validate plot --all                 # every experiment under Results/
    fedgnn-validate plot fedgnn05 --format pdf --dpi 200

Output layout per experiment ``<exp>``::

    Figures/<exp>/loss_all_clients.png      Figures/<exp>/client_quality.png
    Figures/<exp>/clients/client_NN_loss.png   …_metrics.png
    Figures/<exp>/server/global_metrics_evolution.png  server_quality.png
                          aggregation_weights.png

(also available as the ``fedgnn-validate`` console script, or
``python -m fedgnn.validation``.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fedgnn.validation.loader import (
    discover_experiments,
    latest_experiment,
    load_experiment,
)
from fedgnn.validation.plots import render_all


def _cmd_list(args: argparse.Namespace) -> None:
    """List every experiment found under the results root."""
    names = discover_experiments(args.results_dir)
    if not names:
        raise SystemExit(
            f"[validate] no experiments under {args.results_dir}/ — "
            "run `fedgnn-train run …` first."
        )
    latest = latest_experiment(args.results_dir)
    print(f"[validate] experiments in {args.results_dir}/ ({len(names)}):")
    for name in names:
        try:
            exp = load_experiment(name, args.results_dir)
        except SystemExit:
            print(f"    {name:<20} (unreadable)")
            continue
        status = exp.metadata.get("status", "?")
        rounds = exp.metadata.get("rounds_completed", exp.n_rounds)
        test = "test✓" if exp.has_test else "no-test"
        flag = "  <- latest" if name == latest else ""
        print(
            f"    {name:<20} {status:<11} rounds={rounds:<4} "
            f"clients={len(exp.clients):<3} {test}{flag}"
        )


def _render_one(name: str, args: argparse.Namespace) -> None:
    exp = load_experiment(name, args.results_dir)

    # Confusion matrices: prefer the ones stored at train time; for legacy runs
    # that predate CM storage, replay inference to recover them (needs gnn extra).
    if not exp.has_confusion and not args.no_recompute and exp.has_test:
        from fedgnn.validation.confusion import recompute_confusions

        print(f"[validate] {name}: no stored confusion matrix — recomputing …")
        if not recompute_confusions(
            exp, args.models_dir, args.clients_dir, device=args.device
        ):
            print(
                f"[validate] {name}: could not recompute (need the gnn extra + "
                f"{args.models_dir}/{name}/server checkpoint + {args.clients_dir}/); "
                "confusion figures skipped"
            )

    paths = render_all(exp, args.figures_dir, fmt=args.format, dpi=args.dpi)
    out_dir = Path(args.figures_dir) / name
    print(
        f"[validate] {name}: {len(paths)} figures -> {out_dir}/ "
        f"(clients={len(exp.clients)}, rounds={exp.n_rounds}, "
        f"test={'yes' if exp.has_test else 'no'}, "
        f"confusion={'yes' if exp.has_confusion else 'no'})"
    )
    if args.verbose:
        for p in paths:
            print(f"    {p}")


def _cmd_plot(args: argparse.Namespace) -> None:
    """Render figures for one experiment, a named one, or all of them."""
    if args.all:
        names = discover_experiments(args.results_dir)
        if not names:
            raise SystemExit(
                f"[validate] no experiments under {args.results_dir}/ — "
                "run `fedgnn-train run …` first."
            )
    elif args.experiment:
        names = [args.experiment]
    else:
        latest = latest_experiment(args.results_dir)
        if latest is None:
            raise SystemExit(
                f"[validate] no experiments under {args.results_dir}/ — "
                "run `fedgnn-train run …` first."
            )
        names = [latest]

    for name in names:
        _render_one(name, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedgnn-validate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list experiments available under Results/")
    p_list.add_argument(
        "--results_dir", type=str, default="Results", help="root of experiment results"
    )
    p_list.set_defaults(func=_cmd_list)

    p_plot = sub.add_parser(
        "plot", help="render evaluation figures for an experiment into Figures/"
    )
    p_plot.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="experiment name (default: the latest run; ignored with --all)",
    )
    p_plot.add_argument(
        "--all", action="store_true", help="render every experiment under Results/"
    )
    p_plot.add_argument(
        "--results_dir", type=str, default="Results", help="root of experiment results"
    )
    p_plot.add_argument(
        "--figures_dir",
        type=str,
        default="Figures",
        help="root for output figures (per-experiment subfolders)",
    )
    p_plot.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="figure file format",
    )
    p_plot.add_argument("--dpi", type=int, default=120, help="raster output DPI")
    p_plot.add_argument(
        "--no-recompute",
        action="store_true",
        help="don't replay inference for legacy runs without stored confusion "
        "matrices (skip those figures instead)",
    )
    p_plot.add_argument(
        "--models_dir",
        type=str,
        default="Models",
        help="root of experiment checkpoints (for confusion-matrix recompute)",
    )
    p_plot.add_argument(
        "--clients_dir",
        type=str,
        default="data/clients",
        help="federated client graphs (for confusion-matrix recompute)",
    )
    p_plot.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device for confusion-matrix recompute (cpu or cuda)",
    )
    p_plot.add_argument(
        "--verbose", action="store_true", help="print every figure path written"
    )
    p_plot.set_defaults(func=_cmd_plot)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
