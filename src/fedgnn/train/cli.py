"""Orchestration CLI for the single-machine federated training simulation.

Runs the full FedAvg loop over the ``data/clients/*`` partitions, with each run
isolated in its own experiment folder so results are never overwritten::

    fedgnn-train run --rounds 50 --local_epochs 5                 # -> experiment_001
    fedgnn-train run --rounds 50 --local_epochs 5                 # -> experiment_002
    fedgnn-train run --rounds 50 --local_epochs 5 --experiment ablation_alpha0
    fedgnn-train run --rounds 80 --local_epochs 5 --resume        # continue latest

Each round: distribute the global weights -> every client trains locally
(recording its FLOPs) -> collect payloads -> server builds the overlay graph and
runs Adaptive Weighted FedAvg -> evaluate the new global model -> persist.

Per experiment ``<exp>``::

    Models/<exp>/  server/{latest,best}_global_model.pt  +  client_NN/local_model.pt
    Results/<exp>/ server/training_state.json            +  client_NN/results.json
    Results/<exp>/ metadata.json   (start time, config, resume events, progress)

(also available as the ``fedgnn-train`` console script.)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fedgnn.evaluation import SELECTABLE_METRICS
from fedgnn.train.client import CLIENTS_DIR, PROCESSED_DIR, FederatedClient
from fedgnn.train.experiment import Experiment
from fedgnn.train.server import CheckpointManager, FederatedServer

# Metrics averaged across clients and reported every round (beyond the selected).
REPORTED_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_attack_recall",
    "attack_detection_recall",
)


def _resolve_device(cpu: bool) -> torch.device:
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _label_map() -> dict[str, int]:
    label_map_path = PROCESSED_DIR / "label_map.json"
    if not label_map_path.exists():
        raise SystemExit(
            f"[train] missing {label_map_path} — run `fedgnn-data clean` first"
        )
    return json.loads(label_map_path.read_text())


def _discover_clients(n_clients: int | None) -> list[Path]:
    dirs = sorted(CLIENTS_DIR.glob("client_*"))
    if not dirs:
        raise SystemExit(
            f"[train] no clients in {CLIENTS_DIR} — run `fedgnn-data split && build`"
        )
    return dirs[:n_clients] if n_clients is not None else dirs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _final_test_evaluation(
    clients: list[FederatedClient],
    ckpt: CheckpointManager,
    experiment: Experiment,
    client_records: dict[int, list[dict]],
    metric: str,
) -> dict:
    """Evaluate the **best** global model on every client's held-out test set.

    Run once, after training. The test split never influenced selection, so this
    is the closest thing to an honest generalisation estimate the pipeline offers
    (still transductive — see docs). Writes ``Results/<exp>/test_results.json`` and
    folds the test metrics back into each client's ``results.json``.
    """
    source = ckpt.best_path if ckpt.best_path.exists() else ckpt.latest_path
    state = torch.load(source, weights_only=False)["model_state"]

    per_client = {}
    headline = (
        "macro_f1",
        "balanced_accuracy",
        "macro_attack_recall",
        "attack_detection_recall",
        "accuracy",
    )
    for client in clients:
        client.set_weights(state)
        rep = client.evaluate_test()
        per_client[client.name] = {m: round(rep[m], 4) for m in headline}
        # persist the test block into the client's own results.json
        client.save_results(experiment.results_dir, client_records[client.client_id])

    aggregate = {
        m: round(_mean([client.test_report[m] for client in clients]), 4)
        for m in headline
    }
    test_doc = {
        "experiment": experiment.name,
        "evaluated_model": source.name,
        "selection_metric": metric,
        "aggregate": aggregate,
        "per_client": per_client,
    }
    (experiment.results_dir / "test_results.json").write_text(
        json.dumps(test_doc, indent=2)
    )
    return aggregate


def _cmd_run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = _resolve_device(args.cpu)
    label_map = _label_map()
    num_classes = len(label_map)
    benign_class = label_map.get("benign", 0)
    client_dirs = _discover_clients(args.n_clients)

    experiment = Experiment.resolve(
        args.models_dir, args.results_dir, name=args.experiment, resume=args.resume
    )
    config = {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "alpha": args.alpha,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split": args.split,
        "overlay": {
            "start": args.overlay_start,
            "min": args.overlay_min,
            "step": args.overlay_step,
        },
        "metric": args.metric,
        "architecture": {
            "hidden_per_head": args.hidden,
            "heads": args.heads,
            "num_layers": args.layers,
            "embed_dim": args.embed_dim,
            "dropout": args.dropout,
        },
        "louvain_resolution": args.louvain_resolution,
        "n_clients": len(client_dirs),
        "num_classes": num_classes,
        "benign_class": benign_class,
        "seed": args.seed,
        "device": device.type,
    }

    print(
        f"[train] experiment={experiment.name} | device={device.type} | "
        f"clients={len(client_dirs)} | classes={num_classes} | "
        f"rounds={args.rounds} | local_epochs={args.local_epochs} | "
        f"alpha={args.alpha} | metric={args.metric}"
    )
    print(f"[train] models -> {experiment.models_dir}/")
    print(f"[train] results -> {experiment.results_dir}/")

    clients = [
        FederatedClient(
            d,
            num_classes=num_classes,
            device=device,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split=args.split,
            lr=args.lr,
            dropout=args.dropout,
            hidden_per_head=args.hidden,
            heads=args.heads,
            num_layers=args.layers,
            embed_dim=args.embed_dim,
            louvain_resolution=args.louvain_resolution,
            metric=args.metric,
            benign_class=benign_class,
            seed=args.seed,
        )
        for d in client_dirs
    ]
    total_communities = sum(len(c.communities) for c in clients)
    print(
        f"[train] local model: {clients[0].num_params:,} parameters/client | "
        f"{total_communities} communities across {len(clients)} clients"
    )
    server = FederatedServer(
        alpha=args.alpha,
        overlay_start=args.overlay_start,
        overlay_min=args.overlay_min,
        overlay_step=args.overlay_step,
        embed_dim=args.embed_dim,
        device=device,
    )
    ckpt = CheckpointManager(experiment.models_dir, experiment.results_dir)

    # Global model starts from a fresh client template (architectures identical).
    global_state = clients[0].get_weights()
    start_round = 0
    best_metric = float("-inf")
    best_round = -1
    history: list[dict] = []
    client_records: dict[int, list[dict]] = {c.client_id: [] for c in clients}

    resumed = False
    if args.resume:
        loaded = ckpt.load_latest()
        if loaded is None:
            print(
                f"[train] --resume set but no checkpoint in {experiment.models_dir}; "
                "starting fresh"
            )
        else:
            resumed = True
            global_state = loaded["model_state"]
            start_round = int(loaded["round"]) + 1
            best_metric = float(loaded.get("best_metric", float("-inf")))
            state = ckpt.load_state()
            history = state.get("history", [])
            best_round = state.get("best_round", -1)
            for client in clients:
                recs = client.load_results(experiment.results_dir)
                client_records[client.client_id] = recs
                if recs:
                    client.cumulative_flops = recs[-1].get("cumulative_flops", 0)
            print(
                f"[train] resumed from {ckpt.latest_path} @ round {loaded['round']} "
                f"(best={best_metric:.4f}); continuing at round {start_round}"
            )

    experiment.init_metadata(config, resuming=resumed)

    if start_round >= args.rounds:
        print(f"[train] nothing to do — already at round {start_round}/{args.rounds}")
        experiment.update_metadata(status="completed", rounds_completed=start_round)
        return

    run_start = time.time()
    try:
        for r in range(start_round, args.rounds):
            t0 = time.time()

            # 1. distribute global weights, train each client locally, collect payloads
            payloads = []
            round_records: dict[int, dict] = {}
            for client in clients:
                client.set_weights(global_state)
                train_info = client.train(args.local_epochs)
                payload = client.extract_payload()
                payloads.append(payload)
                client.save_model(experiment.models_dir, payload.state_dict)
                local_rep = client.last_report
                round_records[client.client_id] = {
                    "round": r,
                    "local_epochs": args.local_epochs,
                    "final_loss": round(train_info["final_loss"], 4),
                    "local_score": round(payload.val_score, 4),
                    "local_macro_f1": round(local_rep["macro_f1"], 4),
                    "local_balanced_accuracy": round(local_rep["balanced_accuracy"], 4),
                    "local_macro_attack_recall": round(
                        local_rep["macro_attack_recall"], 4
                    ),
                    "local_attack_detection_recall": round(
                        local_rep["attack_detection_recall"], 4
                    ),
                    "flops_per_epoch": train_info["flops_per_epoch"],
                    "flops_this_round": train_info["flops_this_round"],
                    "cumulative_flops": train_info["cumulative_flops"],
                }

            # 2. server: overlay graph + Adaptive Weighted FedAvg
            global_state, info = server.aggregate(payloads)

            # 3. evaluate the *aggregated* global model on every client's val split
            global_reports = []
            for client in clients:
                client.set_weights(global_state)
                client.evaluate()
                rep = client.last_report
                global_reports.append(rep)
                rec = round_records[client.client_id]
                rec["global_score"] = round(rep["score"], 4)
                rec["global_macro_f1"] = round(rep["macro_f1"], 4)
                rec["global_balanced_accuracy"] = round(rep["balanced_accuracy"], 4)
                rec["global_macro_attack_recall"] = round(rep["macro_attack_recall"], 4)
                rec["global_attack_detection_recall"] = round(
                    rep["attack_detection_recall"], 4
                )
                rec["aggregation_weight"] = round(
                    info.weights[info.client_ids.index(client.client_id)], 4
                )

            global_means = {
                m: _mean([rep[m] for rep in global_reports]) for m in REPORTED_METRICS
            }
            avg_metric = _mean([rep["score"] for rep in global_reports])
            round_flops = sum(rec["flops_this_round"] for rec in round_records.values())

            # 4. persist client artifacts (model already saved; append + write results)
            for client in clients:
                client_records[client.client_id].append(round_records[client.client_id])
                client.save_results(
                    experiment.results_dir, client_records[client.client_id]
                )

            # 5. server checkpoints (latest always; best on improvement)
            improved = avg_metric > best_metric
            if improved:
                best_metric = avg_metric
                best_round = r
                ckpt.save_best(global_state, r, avg_metric)
            ckpt.save_latest(global_state, r, best_metric)

            history.append(
                {
                    "round": r,
                    "client_ids": info.client_ids,
                    "local_scores": [round(s, 4) for s in info.scores],
                    "w_k_raw": [round(w, 4) for w in info.raw_weights],
                    "w_k_norm": [round(w, 4) for w in info.weights],
                    "communities_per_client": info.communities_per_client,
                    "overlay_nodes": info.n_communities,
                    "overlay_edges": info.overlay_edges,
                    "overlay_threshold": round(info.overlay_threshold, 4),
                    "overlay_isolated": info.overlay_isolated,
                    "global_score": round(avg_metric, 4),
                    "global_metrics": {k: round(v, 4) for k, v in global_means.items()},
                    "best_metric": round(best_metric, 4),
                    "round_flops": round_flops,
                }
            )
            ckpt.save_state(
                {
                    "experiment": experiment.name,
                    "round": r,
                    "rounds_total": args.rounds,
                    "selection_metric": args.metric,
                    "best_metric": best_metric,
                    "best_round": best_round,
                    "config": config,
                    "history": history,
                }
            )
            experiment.update_metadata(
                status="running",
                rounds_completed=r + 1,
                best_metric=round(best_metric, 6),
                best_round=best_round,
                last_global_metrics={k: round(v, 4) for k, v in global_means.items()},
                elapsed_seconds=round(time.time() - run_start, 1),
            )

            flag = "  <- best" if improved else ""
            print(
                f"\n[round {r:>3}/{args.rounds - 1}] "
                f"global {args.metric}={avg_metric:.4f}{flag} | "
                f"bal_acc={global_means['balanced_accuracy']:.4f} | "
                f"atk_recall={global_means['macro_attack_recall']:.4f} | "
                f"detect={global_means['attack_detection_recall']:.4f} | "
                f"flops={round_flops / 1e9:.2f}G | {time.time() - t0:.1f}s"
            )
            print(
                f"    overlay: {info.n_communities} communities | "
                f"{info.overlay_edges} edges @ thr={info.overlay_threshold:.2f} | "
                f"{info.overlay_isolated} isolated"
            )
            if not args.quiet:
                for client in clients:
                    rec = round_records[client.client_id]
                    n_comm = len(client.communities)
                    print(
                        f"    {client.name} hub={client.hub_ip:<15} "
                        f"comm={n_comm:>3} loss={rec['final_loss']:>9.3f} | "
                        f"val f1={rec['local_macro_f1']:.3f} "
                        f"bal={rec['local_balanced_accuracy']:.3f} "
                        f"atk={rec['local_macro_attack_recall']:.3f} "
                        f"det={rec['local_attack_detection_recall']:.3f} | "
                        f"w={rec['aggregation_weight']:.3f}"
                    )
    except KeyboardInterrupt:
        experiment.update_metadata(
            status="interrupted", elapsed_seconds=round(time.time() - run_start, 1)
        )
        print(
            f"\n[train] interrupted — progress saved to experiment "
            f"'{experiment.name}'. Resume with --resume."
        )
        return

    # Final held-out TEST evaluation on the best global model (never seen in rounds).
    print("[train] evaluating best global model on held-out test sets …")
    test_metrics = _final_test_evaluation(
        clients, ckpt, experiment, client_records, args.metric
    )
    experiment.update_metadata(
        status="completed",
        rounds_completed=args.rounds,
        elapsed_seconds=round(time.time() - run_start, 1),
        test_metrics=test_metrics,
    )
    print(
        f"[train] done — experiment '{experiment.name}' | "
        f"best val {args.metric}={best_metric:.4f} @ round {best_round}"
    )
    if not args.quiet:
        for client in clients:
            t = client.test_report
            print(
                f"    {client.name} hub={client.hub_ip:<15} "
                f"f1={t['macro_f1']:.3f} bal={t['balanced_accuracy']:.3f} "
                f"atk={t['macro_attack_recall']:.3f} "
                f"det={t['attack_detection_recall']:.3f} acc={t['accuracy']:.3f}"
            )
    print(
        f"[test ] AGG macro_f1={test_metrics['macro_f1']:.4f} | "
        f"bal_acc={test_metrics['balanced_accuracy']:.4f} | "
        f"atk_recall={test_metrics['macro_attack_recall']:.4f} | "
        f"detect={test_metrics['attack_detection_recall']:.4f} | "
        f"acc={test_metrics['accuracy']:.4f}"
    )
    print(
        f"[train] models in {experiment.models_dir}/  "
        f"results in {experiment.results_dir}/ (test_results.json)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedgnn-train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the federated training simulation")
    p_run.add_argument("--rounds", type=int, default=50, help="federated rounds")
    p_run.add_argument(
        "--local_epochs", type=int, default=5, help="local epochs per round per client"
    )
    p_run.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="experiment name (default: auto-numbered experiment_NNN)",
    )
    p_run.add_argument(
        "--models_dir",
        type=str,
        default="Models",
        help="root for experiment model checkpoints",
    )
    p_run.add_argument(
        "--results_dir",
        type=str,
        default="Results",
        help="root for experiment results + metadata",
    )
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="resume the latest (or --experiment named) run from its latest checkpoint",
    )
    p_run.add_argument(
        "--alpha", type=float, default=0.2, help="floor weight for Adaptive FedAvg"
    )
    p_run.add_argument(
        "--lr", type=float, default=1e-3, help="local Adam learning rate"
    )
    p_run.add_argument(
        "--val_ratio", type=float, default=0.15, help="per-client edge val fraction"
    )
    p_run.add_argument(
        "--test_ratio", type=float, default=0.15, help="per-client edge test fraction"
    )
    p_run.add_argument(
        "--split",
        choices=["temporal", "random"],
        default="temporal",
        help="edge split: temporal (past->future, default) or random",
    )
    # --- local GAT architecture ---
    p_run.add_argument(
        "--hidden", type=int, default=64, help="GAT hidden units per attention head"
    )
    p_run.add_argument(
        "--heads", type=int, default=4, help="number of GAT attention heads"
    )
    p_run.add_argument(
        "--layers", type=int, default=3, help="number of GAT layers (>=2)"
    )
    p_run.add_argument(
        "--embed_dim",
        type=int,
        default=32,
        help="node embedding width = hub fingerprint width (shared with server)",
    )
    p_run.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate (GAT + classifier)"
    )
    p_run.add_argument(
        "--louvain_resolution",
        type=float,
        default=1.0,
        help="Louvain resolution (>1 => more, smaller communities per client)",
    )
    p_run.add_argument(
        "--overlay_start",
        type=float,
        default=0.7,
        help="initial cosine threshold for the community overlay graph",
    )
    p_run.add_argument(
        "--overlay_min",
        type=float,
        default=0.3,
        help="floor the overlay threshold is lowered to while nodes are isolated",
    )
    p_run.add_argument(
        "--overlay_step",
        type=float,
        default=0.05,
        help="decrement applied to the overlay threshold each iteration",
    )
    p_run.add_argument(
        "--metric",
        choices=list(SELECTABLE_METRICS),
        default="balanced_accuracy",
        help="metric driving best-model selection (all metrics are still logged)",
    )
    p_run.add_argument(
        "--n_clients", type=int, default=None, help="use only the first N clients"
    )
    p_run.add_argument("--seed", type=int, default=42, help="global random seed")
    p_run.add_argument(
        "--cpu", action="store_true", help="force CPU even if CUDA exists"
    )
    p_run.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the per-client breakdown (print only the round summary)",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
