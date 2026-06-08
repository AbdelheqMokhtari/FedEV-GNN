"""Command-line interface for the data pipeline.

A single entrypoint with subcommands so each stage can be run consistently::

    fedgnn-data load [--sample | --processed | --split 0]
    fedgnn-data sample --rows 1000000    # raw -> data/samples/
    fedgnn-data clean                    # sample -> data/processed/
    fedgnn-data select                   # validate & write the engineered feature set
    fedgnn-data split [--csv]            # hub partitioning -> data/clients/
    fedgnn-data build [--window-ms N]    # PyG graphs -> data/clients/*/graph*.pt

(also available as the ``fedgnn-data`` console script).
"""

import argparse
import json
import textwrap
from pathlib import Path

import polars as pl

from fedgnn.data import graph_builder
from fedgnn.data.federated_split import (
    dirichlet_splitter,
    hub_splitter,
    threat_splitter,
)
from fedgnn.data.loaders import ToNIoTLoader
from fedgnn.data.preprocessing import cleaning, feature_selection
from fedgnn.data.preprocessing.cleaning import REFERENCE_COLS, TIME_COLS, TOPOLOGY_COLS
from fedgnn.data.preprocessing.feature_selection import GROUPS

PROCESSED_DIR = ToNIoTLoader.PROJECT_ROOT / "data" / "processed"
LABEL_COLS = ["Label", "Attack", "y_binary", "y_multiclass"]

_GROUP_LABELS = {
    "protocol": "Protocol",
    "application": "Application (L7)",
    "service": "Service / Port",
    "tcp_state": "TCP State (flag bits)",
    "volume": "Volume",
    "volume_ratio": "Volume Ratios",
    "size": "Packet Size",
    "temporal": "Temporal",
    "reachability": "Reachability (TTL)",
}
_W = 72
_I = "    "


def _section(title: str) -> None:
    print(f"\n  ── {title} " + "─" * max(0, _W - len(title) - 6))


def _wrapped_list(items: list[str]) -> None:
    for line in textwrap.wrap(", ".join(items), width=_W - len(_I)):
        print(_I + line)


def _print_features(df_cols: list[str], processed: bool) -> None:
    """Print a grouped feature catalogue for the loaded dataset."""
    all_features: list[str] = []
    selected_set: set[str] = set()
    label_map: dict[str, int] = {}

    if processed:
        feat_path = PROCESSED_DIR / "feature_columns.json"
        sel_path = PROCESSED_DIR / "selected_features.json"
        lmap_path = PROCESSED_DIR / "label_map.json"
        if feat_path.exists():
            all_features = json.loads(feat_path.read_text())
        if sel_path.exists():
            selected_set = set(json.loads(sel_path.read_text()))
        if lmap_path.exists():
            label_map = json.loads(lmap_path.read_text())

    col_set = set(df_cols)
    fixed_set = set(TOPOLOGY_COLS + TIME_COLS + REFERENCE_COLS)

    print()
    _section("Graph structure")
    print(f"{_I}Topology  : {', '.join(c for c in TOPOLOGY_COLS if c in col_set)}")
    print(f"{_I}Time      : {', '.join(c for c in TIME_COLS if c in col_set)}")
    print(f"{_I}Reference : {', '.join(c for c in REFERENCE_COLS if c in col_set)}")
    labels_present = [c for c in LABEL_COLS if c in col_set]
    if labels_present:
        print(f"{_I}Labels    : {', '.join(labels_present)}")

    if processed and all_features:
        unselected = [f for f in all_features if f not in selected_set]

        _section(f"Selected features ({len(selected_set)})")
        for group, feats in GROUPS.items():
            present = [f for f in feats if f in col_set]
            if present:
                label = _GROUP_LABELS.get(group, group)
                print(f"\n{_I}{label} ({len(present)})")
                for line in textwrap.wrap(", ".join(present), width=_W - len(_I) - 2):
                    print(_I + "  " + line)

        if unselected:
            _section(f"Not selected ({len(unselected)})")
            _wrapped_list(unselected)

        if label_map:
            _section(f"Classes ({len(label_map)})")
            pairs = sorted(label_map.items(), key=lambda x: x[1])
            col_w = max(len(k) for k, _ in pairs) + 2
            cols = 3
            for i, (name, idx) in enumerate(pairs):
                end = "\n" if (i + 1) % cols == 0 or i == len(pairs) - 1 else ""
                print(f"{_I}{idx}  {name:<{col_w}}", end=end)

        print(
            f"\n\n  {len(df_cols)} total cols | {len(all_features)} features"
            f" | {len(selected_set)} selected"
        )
    else:
        feature_cols = [c for c in df_cols if c not in fixed_set | set(LABEL_COLS)]
        _section(f"Features ({len(feature_cols)})")
        _wrapped_list(feature_cols)
        print(f"\n  {len(df_cols)} total cols | {len(feature_cols)} features")


CLIENTS_DIR = ToNIoTLoader.PROJECT_ROOT / "data" / "clients"


def _resolve_client(token: str) -> Path:
    """Accept '0', '00', 'client_00', etc. and return the client folder path."""
    # Normalise to 'client_NN'
    if token.startswith("client_"):
        name = token
    else:
        try:
            name = f"client_{int(token):02d}"
        except ValueError:
            raise SystemExit(
                f"[load] unrecognised client id '{token}' — use 0-9 or client_00"
            )
    path = CLIENTS_DIR / name
    if not path.exists():
        available = (
            sorted(p.name for p in CLIENTS_DIR.glob("client_*"))
            if CLIENTS_DIR.exists()
            else []
        )
        hint = (
            f"  Available: {', '.join(available)}"
            if available
            else "  Run `fedgnn-data split` first."
        )
        raise SystemExit(f"[load] client folder not found: {path.name}\n{hint}")
    return path


def _print_client(client_dir: Path, head: int) -> None:
    """Print meta, preview, attack distribution, and feature catalogue for a client."""
    meta_path = client_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    df = pl.read_parquet(client_dir / "flows.parquet")

    print(
        f"[load] source    :"
        f"{client_dir.relative_to(ToNIoTLoader.PROJECT_ROOT)}/flows.parquet"
    )
    print(f"[load] hub IP    : {meta.get('hub_ip', 'unknown')}")
    print(f"[load] shape     : {df.height:,} rows × {df.width} cols")

    hub_stats = meta.get("hub_stats", {})
    if hub_stats:
        print(
            f"[load] hub stats : score={hub_stats.get('score', '?'):,}  "
            f"src_flows={hub_stats.get('src_flows', '?'):,}  "
            f"dst_flows={hub_stats.get('dst_flows', '?'):,}"
        )
    print()
    print(df.head(head))

    if "Attack" in df.columns:
        print()
        print(df.group_by("Attack").len().sort("len", descending=True))

    _print_features(df.columns, processed=True)


def _cmd_load(args: argparse.Namespace) -> None:
    """Inspect raw data, a sample, a processed dataset, or a federated client."""
    if args.split is not None:
        client_dir = _resolve_client(args.split)
        _print_client(client_dir, args.head)
        return

    if args.processed:
        # Pick cleaned_<rows>.parquet or the newest available.
        if args.rows is not None:
            path = PROCESSED_DIR / f"cleaned_{args.rows}.parquet"
            if not path.exists():
                available = sorted(PROCESSED_DIR.glob("cleaned_*.parquet"))
                names = ", ".join(p.name for p in available) or "none"
                raise SystemExit(
                    f"[load] no cleaned file for {args.rows} rows. Available: {names}"
                )
        else:
            candidates = sorted(PROCESSED_DIR.glob("cleaned_*.parquet"))
            if not candidates:
                raise SystemExit(
                    "[load] no processed data found — run `fedgnn-data clean` first"
                )
            path = max(candidates, key=lambda p: p.stat().st_mtime)
        df = pl.read_parquet(path)
        print(f"[load] source    : {path.relative_to(ToNIoTLoader.PROJECT_ROOT)}")
        print(f"[load] shape     : {df.height:,} rows × {df.width} cols")
        print(df.head(args.head))
        if "Attack" in df.columns:
            print()
            print(df.group_by("Attack").len().sort("len", descending=True))
        _print_features(df.columns, processed=True)
    else:
        loader = ToNIoTLoader(sample=args.sample)
        lf = loader.load(nrows=args.rows, stratify=args.stratify)
        schema = lf.collect_schema()
        print(f"[load] source    : {loader.data_path}")
        print(
            f"[load] shape     : {lf.select(pl.len()).collect().item():,}"
            "rows × {len(schema)} cols"
        )
        print(lf.head(args.head).collect())
        if "Attack" in schema.names():
            print()
            print(lf.group_by("Attack").len().sort("len", descending=True).collect())
        _print_features(schema.names(), processed=False)


def _cmd_sample(args: argparse.Namespace) -> None:
    """Write a stratified sample of the raw dataset to data/samples/."""
    loader = ToNIoTLoader(sample=False)
    print(
        f"[sample] drawing {args.rows:,} stratified rows from {loader.data_path.name}"
    )
    df = loader.load(nrows=args.rows, stratify=True, seed=args.seed).collect()

    ToNIoTLoader.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = ToNIoTLoader.SAMPLES_DIR / f"ton_iot_{df.height}.parquet"
    df.write_parquet(out, compression="snappy")
    print(
        f"[sample] wrote {out.relative_to(ToNIoTLoader.PROJECT_ROOT)}"
        "({df.height:,} rows)"
    )

    # Parquet is the format the pipeline uses; CSV is an optional human-readable copy.
    if args.csv:
        csv_out = out.with_suffix(".csv")
        df.write_csv(csv_out)
        print(
            f"[sample] wrote {csv_out.relative_to(ToNIoTLoader.PROJECT_ROOT)} (--csv)"
        )


def _cmd_clean(args: argparse.Namespace) -> None:
    cleaning.run(sample_rows=args.rows)


def _cmd_select(_args: argparse.Namespace) -> None:
    feature_selection.run()


def _cmd_split(args: argparse.Namespace) -> None:
    if args.strategy == "dirichlet":
        dirichlet_splitter.run(
            n_clients=args.n_clients, alpha=args.alpha, seed=args.seed, csv=args.csv
        )
    elif args.strategy == "threat":
        if args.n_clients is not None and args.n_clients != threat_splitter.N_CLIENTS:
            print(
                f"[split] note: threat strategy uses {threat_splitter.N_CLIENTS}"
                f" role-segments; ignoring --n-clients {args.n_clients}"
            )
        threat_splitter.run(csv=args.csv, seed=args.seed)
    else:
        hub_splitter.run(
            n_clients=args.n_clients or hub_splitter.N_CLIENTS, csv=args.csv
        )


def _cmd_build(args: argparse.Namespace) -> None:
    graph_builder.run(window_ms=args.window_ms, n_clients=args.n_clients)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedgnn-data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="inspect raw, sample, or processed data")
    src = p_load.add_mutually_exclusive_group()
    src.add_argument("--sample", action="store_true", help="load from data/samples/")
    src.add_argument(
        "--processed", action="store_true", help="load from data/processed/"
    )
    src.add_argument(
        "--split",
        type=str,
        metavar="CLIENT",
        help="load a federated client (e.g. 0, client_00) from data/clients/",
    )
    p_load.add_argument(
        "--rows",
        type=int,
        default=None,
        help="limit rows (raw/sample) or pick cleaned_{N}.parquet (processed)",
    )
    p_load.add_argument(
        "--stratify", action="store_true", help="stratified sampling (raw/sample only)"
    )
    p_load.add_argument("--head", type=int, default=5, help="preview rows to print")
    p_load.set_defaults(func=_cmd_load)

    p_sample = sub.add_parser(
        "sample", help="draw a stratified sample from the raw data"
    )
    p_sample.add_argument("--rows", type=int, default=1_000_000, help="number of rows")
    p_sample.add_argument("--seed", type=int, default=42, help="random seed")
    p_sample.add_argument("--csv", action="store_true", help="also write a .csv copy")
    p_sample.set_defaults(func=_cmd_sample)

    p_clean = sub.add_parser("clean", help="clean the sample into data/processed/")
    p_clean.add_argument(
        "--rows",
        type=int,
        default=None,
        help="pick sample by row count (e.g. 10000 -> ton_iot_10000.parquet); "
        "default: newest sample",
    )
    p_clean.set_defaults(func=_cmd_clean)

    p_select = sub.add_parser(
        "select", help="curate the engineered feature set -> selected_features.json"
    )
    p_select.set_defaults(func=_cmd_select)

    p_split = sub.add_parser(
        "split", help="partition flows into federated clients → data/clients/"
    )
    p_split.add_argument(
        "--strategy",
        choices=["dirichlet", "threat", "hub"],
        default="dirichlet",
        help="dirichlet = non-IID label skew, all classes per client (default); "
        "threat = one-attack-per-client segments; hub = traffic-ranked (near-IID)",
    )
    p_split.add_argument(
        "--n-clients",
        type=int,
        default=None,
        help="number of clients (dirichlet/hub; threat uses a fixed 8 segments)",
    )
    p_split.add_argument(
        "--alpha",
        type=float,
        default=dirichlet_splitter.DEFAULT_ALPHA,
        help="Dirichlet concentration (lower = sharper label skew; default 0.5)",
    )
    p_split.add_argument(
        "--seed", type=int, default=42, help="random seed for the partition"
    )
    p_split.add_argument(
        "--csv",
        action="store_true",
        help="also write flows.csv alongside flows.parquet in each client folder",
    )
    p_split.set_defaults(func=_cmd_split)

    p_build = sub.add_parser(
        "build",
        help="build PyG static + temporal graphs for T-GAT → data/clients/*/graph*.pt",
    )
    p_build.add_argument(
        "--window-ms",
        type=int,
        default=3_600_000,
        help="temporal snapshot window width in milliseconds (default 1 hour)",
    )
    p_build.add_argument(
        "--n-clients",
        type=int,
        default=None,
        help="build only the first N client folders (default: all)",
    )
    p_build.set_defaults(func=_cmd_build)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
