"""Command-line interface for the data pipeline.

A single entrypoint with subcommands so each stage can be run consistently::

    python -m fedgnn.data load [--sample | --processed] [--rows N]
    python -m fedgnn.data sample --rows 1000000    # raw -> data/samples/
    python -m fedgnn.data clean                    # sample -> data/processed/
    python -m fedgnn.data select                   # validate & write 26-feature set

(also available as the ``fedgnn-data`` console script).
"""

import argparse
import json
import textwrap

import polars as pl

from fedgnn.data.loaders import ToNIoTLoader
from fedgnn.data.preprocessing import cleaning, feature_selection
from fedgnn.data.preprocessing.cleaning import REFERENCE_COLS, TIME_COLS, TOPOLOGY_COLS
from fedgnn.data.preprocessing.feature_selection import GROUPS

PROCESSED_DIR = ToNIoTLoader.PROJECT_ROOT / "data" / "processed"
LABEL_COLS = ["Label", "Attack", "y_binary", "y_multiclass"]

_GROUP_LABELS = {
    "behavioral": "Behavioral",
    "temporal": "Temporal",
    "content_volume": "Content & Volume",
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
            f"\n\n  {len(df_cols)} total cols | {len(all_features)}"
            "features | {len(selected_set)} selected"
        )
    else:
        feature_cols = [c for c in df_cols if c not in fixed_set | set(LABEL_COLS)]
        _section(f"Features ({len(feature_cols)})")
        _wrapped_list(feature_cols)
        print(f"\n  {len(df_cols)} total cols | {len(feature_cols)} features")


def _cmd_load(args: argparse.Namespace) -> None:
    """Inspect raw data, a sample, or the processed dataset."""
    if args.split is not None:
        splits_dir = ToNIoTLoader.PROJECT_ROOT / "data" / "splits"
        if not splits_dir.exists():
            raise SystemExit(
                "[load] federated splits not yet created — run `fedgnn-data split`"
            )
        raise SystemExit(
            "[load] --split not yet implemented " "(coming with the federated stage)"
        )

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
        help="load a federated split by client id (coming soon)",
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

    p_select = sub.add_parser("select", help="write the fixed 26-feature set to JSON")
    p_select.set_defaults(func=_cmd_select)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
