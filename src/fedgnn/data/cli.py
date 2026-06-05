"""Command-line interface for the data pipeline.

A single entrypoint with subcommands so each stage can be run consistently::

    python -m fedgnn.data load --sample            # inspect raw or sample data
    python -m fedgnn.data sample --rows 1000000    # raw -> data/samples/
    python -m fedgnn.data clean                    # sample -> data/processed/
    python -m fedgnn.data select                   # edge-optimized feature subset
    python -m fedgnn.data graph --window-ms 3600000  # build temporal snapshots

(also available as the ``fedgnn-data`` console script).
"""

import argparse

import polars as pl

from fedgnn.data.loaders import ToNIoTLoader
from fedgnn.data.preprocessing import cleaning, feature_selection

PROCESSED_DIR = ToNIoTLoader.PROJECT_ROOT / "data" / "processed"


def _cmd_load(args: argparse.Namespace) -> None:
    """Load the raw or sample dataset through the loader and print a preview."""
    loader = ToNIoTLoader(sample=args.sample)
    lf = loader.load(nrows=args.rows, stratify=args.stratify)
    schema = lf.collect_schema()

    print(f"[load] source: {loader.data_path}")
    print(
        f"[load] rows: {lf.select(pl.len()).collect().item():,} | cols: {len(schema)}"
    )
    print(lf.head(args.head).collect())
    if "Attack" in schema.names():
        print(lf.group_by("Attack").len().sort("len", descending=True).collect())


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

    p_load = sub.add_parser("load", help="load raw/sample data and print a preview")
    p_load.add_argument("--sample", action="store_true", help="load from data/samples/")
    p_load.add_argument("--rows", type=int, default=None, help="limit rows loaded")
    p_load.add_argument("--stratify", action="store_true", help="stratified sampling")
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
