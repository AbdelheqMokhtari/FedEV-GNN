"""Monitoring CLI — inspect the per-client resource cost of a federated run.

Builds each client and runs one real training step to report, per client, the
**RAM** it needs (peak host memory, and peak GPU memory on CUDA), its **compute**
cost (FLOPs per epoch / per round), and the **model** size (parameters, per-module
breakdown, training-state footprint)::

    fedgnn-monitor profile                      # all clients, default architecture
    fedgnn-monitor profile --cpu                # force CPU (host-RAM peaks)
    fedgnn-monitor profile --client 0           # just client_00 (clean absolute RAM)
    fedgnn-monitor profile --hidden 128 --heads 8 --layers 4   # size a bigger model
    fedgnn-monitor profile --json monitor.json  # also dump the raw report

The architecture flags mirror ``fedgnn-train run`` so you can size any candidate
model before committing to a training run. Needs the ``gnn`` extra (torch /
torch-geometric) and the client graphs under ``data/clients/`` (run
``fedgnn-data split && fedgnn-data build`` first).

(also available as the ``fedgnn-monitor`` console script, or
``python -m fedgnn.monitor``.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLIENTS_DIR = Path("data/clients")
LABEL_MAP = Path("data/processed/label_map.json")


def _resolve_device(cpu: bool) -> str:
    if cpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _label_map() -> dict[str, int]:
    if not LABEL_MAP.exists():
        raise SystemExit(
            f"[monitor] missing {LABEL_MAP} — run `fedgnn-data clean` first"
        )
    return json.loads(LABEL_MAP.read_text())


def _discover_clients(token: str | None, n_clients: int | None) -> list[Path]:
    if not CLIENTS_DIR.exists():
        raise SystemExit(
            f"[monitor] no clients in {CLIENTS_DIR} — run `fedgnn-data split && build`"
        )
    if token is not None:
        name = token if token.startswith("client_") else f"client_{int(token):02d}"
        path = CLIENTS_DIR / name
        if not path.exists():
            raise SystemExit(f"[monitor] client folder not found: {name}")
        return [path]
    dirs = sorted(CLIENTS_DIR.glob("client_*"))
    if not dirs:
        raise SystemExit(f"[monitor] no client_* folders in {CLIENTS_DIR}")
    return dirs[:n_clients] if n_clients is not None else dirs


def _g(flops: int) -> str:
    return f"{flops / 1e9:.2f}G"


def _print_architecture(rep: dict, args: argparse.Namespace, num_classes: int) -> None:
    """Model architecture + parameter breakdown (identical across clients)."""
    print("\n── Model architecture (shared by every client) " + "─" * 26)
    print(
        f"  in_features={rep.get('in_features', '?')}  classes={num_classes}  "
        f"hidden/head={args.hidden}  heads={args.heads}  layers={args.layers}  "
        f"embed_dim={args.embed_dim}  dropout={args.dropout}"
    )
    print(f"\n  {'module':<16}{'params':>14}")
    print(f"  {'-'*16}{'-'*14:>14}")
    for row in rep["param_breakdown"]:
        print(f"  {row['module']:<16}{row['params']:>14,}")
    print(f"  {'-'*16}{'-'*14:>14}")
    print(f"  {'TOTAL':<16}{rep['model_params']:>14,}")
    print(
        f"\n  weights (fp32): {rep['model_mb']:.3f} MB   |   "
        f"training state (weights+grads+Adam): {rep['train_state_mb']:.3f} MB"
    )


def _print_table(reports: list[dict], device: str) -> None:
    """Per-client resource table + federation totals."""
    gpu = device == "cuda"
    mem_col = "gpu_peak_MB" if gpu else "host_ram_MB"
    print("\n── Per-client resource cost " + "─" * 45)
    header = (
        f"  {'client':<11}{'hub':<16}{'nodes':>7}{'edges':>10}"
        f"{'graph_MB':>10}{mem_col:>13}{'FLOPs/epoch':>13}{'FLOPs/round':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in reports:
        mem = r.get("gpu_peak_mb") if gpu else r.get("host_ram_peak_mb")
        print(
            f"  {r['client']:<11}{r['hub_ip']:<16}{r['n_nodes']:>7,}"
            f"{r['n_edges']:>10,}{r['graph_tensors_mb']:>10.2f}{mem:>13.1f}"
            f"{_g(r['flops_per_epoch']):>13}{_g(r['flops_per_round']):>13}"
        )

    total_round = sum(r["flops_per_round"] for r in reports)
    peak_mem = max(
        (r.get("gpu_peak_mb") if gpu else r.get("host_ram_peak_mb")) or 0.0
        for r in reports
    )
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TOTAL/PEAK':<11}{'':<16}{'':>7}{'':>10}{'':>10}"
        f"{peak_mem:>13.1f}{'':>13}{_g(total_round):>13}"
    )
    label = "peak GPU memory" if gpu else "peak host RAM (increment)"
    print(
        f"\n  device={device}  |  heaviest client {label}={peak_mem:.1f} MB  |  "
        f"federation compute/round={_g(total_round)} FLOPs"
    )
    if not gpu:
        print(
            "  note: host-RAM peaks are increments over the process baseline; for "
            "an absolute number profile one client with --client N."
        )


def _cmd_profile(args: argparse.Namespace) -> None:
    try:
        from fedgnn.monitor.profiler import profile_client
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit(
            f"[monitor] could not import profiler ({exc}). "
            'Install the gnn extra: pip install -e ".[gnn]"'
        )

    label_map = _label_map()
    num_classes = len(label_map)
    benign_class = label_map.get("benign", 0)
    device = _resolve_device(args.cpu)
    client_dirs = _discover_clients(args.client, args.n_clients)

    print(
        f"[monitor] profiling {len(client_dirs)} client(s) on {device} | "
        f"classes={num_classes} | model "
        f"{args.hidden}/{args.heads}/{args.layers} (hidden/heads/layers)"
    )

    reports: list[dict] = []
    for d in client_dirs:
        rep = profile_client(
            d,
            num_classes=num_classes,
            device=device,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split=args.split,
            dropout=args.dropout,
            hidden_per_head=args.hidden,
            heads=args.heads,
            num_layers=args.layers,
            embed_dim=args.embed_dim,
            louvain_resolution=args.louvain_resolution,
            benign_class=benign_class,
            seed=args.seed,
            local_epochs=args.local_epochs,
        )
        reports.append(rep)
        mem = rep.get("gpu_peak_mb") if device == "cuda" else rep["host_ram_peak_mb"]
        print(
            f"  ✓ {rep['client']} ({rep['n_edges']:,} edges, "
            f"{mem:.1f} MB, {_g(rep['flops_per_epoch'])}/epoch)"
        )

    _print_architecture(reports[0], args, num_classes)
    _print_table(reports, device)

    if args.json:
        out = Path(args.json)
        out.write_text(
            json.dumps(
                {"device": device, "num_classes": num_classes, "clients": reports},
                indent=2,
            )
        )
        print(f"\n[monitor] wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedgnn-monitor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="measure per-client RAM, FLOPs, and model size")
    p.add_argument(
        "--client",
        type=str,
        default=None,
        help="profile only this client (e.g. 0 or client_00); best for absolute RAM",
    )
    p.add_argument(
        "--n_clients", type=int, default=None, help="profile only the first N clients"
    )
    p.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="local epochs per round (scales the reported FLOPs/round)",
    )
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA exists")
    p.add_argument(
        "--json", type=str, default=None, help="also dump the report to JSON"
    )
    # architecture (mirrors fedgnn-train run, so you can size any candidate model)
    p.add_argument("--hidden", type=int, default=64, help="GAT hidden units per head")
    p.add_argument("--heads", type=int, default=4, help="GAT attention heads")
    p.add_argument("--layers", type=int, default=3, help="GAT layers (>=2)")
    p.add_argument("--embed_dim", type=int, default=32, help="node embedding width")
    p.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    p.add_argument(
        "--louvain_resolution", type=float, default=1.0, help="Louvain resolution"
    )
    p.add_argument(
        "--split",
        choices=["stratified", "stratified_temporal", "temporal", "random"],
        default="stratified",
        help="edge split used when building the client (affects train-edge count)",
    )
    p.add_argument("--val_ratio", type=float, default=0.15, help="val edge fraction")
    p.add_argument("--test_ratio", type=float, default=0.15, help="test edge fraction")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.set_defaults(func=_cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
