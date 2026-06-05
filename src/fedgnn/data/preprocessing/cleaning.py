import json
from pathlib import Path

import polars as pl

from fedgnn.data.loaders import ToNIoTLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# IP endpoints define the graph topology only.
TOPOLOGY_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

# Flow timestamps drive the temporal snapshot slicing in the graph stage.
TIME_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]

# L4_SRC_PORT is OS-assigned / ephemeral -> kept for reference but NOT a feature.
# (L4_DST_PORT is left in the feature set: it identifies the service, e.g. 80/443.)
REFERENCE_COLS = ["L4_SRC_PORT"]
RAW_LABEL_COLS = ["Label", "Attack"]

BENIGN = "benign"  # normal traffic, always class 0
UNSPECIFIED_ADDR = "0.0.0.0"


def drop_unroutable(df: pl.DataFrame) -> pl.DataFrame:
    """Drop flows that cannot map to a meaningful graph edge.

    * 0.0.0.0 endpoints are the unspecified address (no real node).
    * self-loops (src IP == dst IP) would become self-edges in the flow graph.
    """
    return df.filter(
        (pl.col("IPV4_SRC_ADDR") != UNSPECIFIED_ADDR)
        & (pl.col("IPV4_DST_ADDR") != UNSPECIFIED_ADDR)
        & (pl.col("IPV4_SRC_ADDR") != pl.col("IPV4_DST_ADDR"))
    )


def normalize_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase the Attack label (the dataset mixes casing, e.g. DDoS vs ddos)."""
    return df.with_columns(pl.col("Attack").str.to_lowercase())


def fill_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Replace inf/-inf with null, then fill numeric nulls with 0.

    These are NetFlow counters: a missing value means the event did not occur,
    so 0 is the correct, meaningful fill.
    """
    float_cols = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns(
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_cols
        )
    numeric_cols = [c for c, t in df.schema.items() if t.is_numeric()]
    return df.with_columns(pl.col(c).fill_null(0) for c in numeric_cols)


def drop_constant_columns(
    df: pl.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    """Drop feature columns with <= 1 unique value (zero variance == no signal)."""
    n_unique = df.select(pl.col(c).n_unique().alias(c) for c in feature_cols).row(0)
    dropped = [c for c, k in zip(feature_cols, n_unique) if k <= 1]
    return [c for c in feature_cols if c not in dropped], dropped


def build_label_map(df: pl.DataFrame) -> dict[str, int]:
    """Global-stable map: benign -> 0, attacks alphabetically -> 1..N.

    Every client shares this index space so FedAvg averages matching neurons.
    """
    attacks = sorted(c for c in df["Attack"].unique().to_list() if c != BENIGN)
    return {BENIGN: 0, **{a: i for i, a in enumerate(attacks, start=1)}}


def encode_labels(df: pl.DataFrame, label_map: dict[str, int]) -> pl.DataFrame:
    """Add y_binary (Int8) and y_multiclass (Int) from the raw labels."""
    return df.with_columns(
        pl.col("Label").cast(pl.Int8).alias("y_binary"),
        pl.col("Attack").replace_strict(label_map).cast(pl.Int64).alias("y_multiclass"),
    )


def clean(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str], dict[str, int], dict]:
    """Run the cleaning pipeline; return (df, feature_cols, label_map, report)."""
    rows_before = df.height

    df = drop_unroutable(df)
    df = normalize_labels(df)
    df = fill_missing(df)

    # Features = everything that is not topology, time, a reference id, or a label.
    # (L4_DST_PORT is intentionally NOT reserved, so it stays an edge feature.)
    reserved = set(TOPOLOGY_COLS + TIME_COLS + REFERENCE_COLS + RAW_LABEL_COLS)
    feature_cols = [c for c in df.columns if c not in reserved]
    feature_cols, dropped_constant = drop_constant_columns(df, feature_cols)

    label_map = build_label_map(df)
    df = encode_labels(df, label_map)

    keep = (
        TOPOLOGY_COLS
        + REFERENCE_COLS
        + TIME_COLS
        + RAW_LABEL_COLS
        + ["y_binary", "y_multiclass"]
    )
    df = df.select(keep + feature_cols)

    report = {
        "rows_before": rows_before,
        "rows_after": df.height,
        "dropped_constant": dropped_constant,
        "n_features": len(feature_cols),
    }
    return df, feature_cols, label_map, report


def run(sample_rows: int | None = None) -> None:
    """CLI entry: load the sample, clean it, and write the processed outputs.

    ``sample_rows`` picks a specific ``data/samples/ton_iot_{n}.parquet`` by row
    count.  If omitted, the newest sample file is used (default behaviour).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if sample_rows is not None:
        path = ToNIoTLoader.SAMPLES_DIR / f"ton_iot_{sample_rows}.parquet"
        if not path.exists():
            available = sorted(ToNIoTLoader.SAMPLES_DIR.glob("ton_iot_*.parquet"))
            names = ", ".join(p.name for p in available) or "none"
            raise FileNotFoundError(
                f"No sample with {sample_rows} rows found. Available: {names}"
            )
        loader = ToNIoTLoader(data_path=path)
    else:
        loader = ToNIoTLoader(sample=True)
    n_rows = loader.data_path.stem.split("_")[-1]  # ton_iot_1000000 -> 1000000
    print(f"[clean] loading sample: {loader.data_path.name}")
    df = loader.load().collect()

    df, feature_cols, label_map, report = clean(df)

    out_parquet = PROCESSED_DIR / f"cleaned_{n_rows}.parquet"
    df.write_parquet(out_parquet, compression="snappy")
    (PROCESSED_DIR / "feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2)
    )
    (PROCESSED_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2))

    dropped = report["rows_before"] - report["rows_after"]
    print(
        f"[clean] rows {report['rows_before']:,} -> {report['rows_after']:,} "
        f"(dropped {dropped:,})"
    )
    print(f"[clean] constant cols dropped: {report['dropped_constant'] or 'none'}")
    print(f"[clean] features kept: {report['n_features']} | classes: {len(label_map)}")
    print(f"[clean] wrote {out_parquet.relative_to(PROJECT_ROOT)}")
