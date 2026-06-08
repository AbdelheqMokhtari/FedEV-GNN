import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLIENTS_DIR = PROJECT_ROOT / "data" / "clients"

SRC = "IPV4_SRC_ADDR"
DST = "IPV4_DST_ADDR"
LABEL_COL = "y_multiclass"

DEFAULT_N_CLIENTS = 8
DEFAULT_ALPHA = 0.5


def _dirichlet_counts(n_items: int, proportions: np.ndarray) -> np.ndarray:
    """Split ``n_items`` into per-client integer counts matching ``proportions``.

    Floors then hands the rounding remainder to the clients with the largest
    fractional parts, so the counts always sum back to ``n_items`` exactly.
    """
    exact = proportions * n_items
    counts = np.floor(exact).astype(np.int64)
    remainder = n_items - int(counts.sum())
    if remainder > 0:
        for j in np.argsort(-(exact - counts))[:remainder]:
            counts[j] += 1
    return counts


def partition(
    df: pl.DataFrame, n_clients: int, alpha: float, seed: int
) -> list[pl.DataFrame]:
    """Dirichlet label-skew assignment of flows to clients. Returns one df each."""
    rng = np.random.default_rng(seed)
    y = df[LABEL_COL].to_numpy()
    cid = np.empty(len(y), dtype=np.int64)

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        counts = _dirichlet_counts(len(idx), proportions)
        bounds = np.cumsum(counts)[:-1]
        for client_id, chunk in enumerate(np.split(idx, bounds)):
            cid[chunk] = client_id

    tagged = df.with_columns(pl.Series("cid", cid).cast(pl.Int32))
    return [tagged.filter(pl.col("cid") == i).drop("cid") for i in range(n_clients)]


def _class_distribution(df: pl.DataFrame) -> dict[str, int]:
    if "Attack" not in df.columns:
        return {}
    return dict(df.group_by("Attack").len().sort("Attack").iter_rows())


def save_splits(
    splits: list[pl.DataFrame], alpha: float, seed: int, csv: bool = False
) -> Path:
    """Write one folder per client plus a manifest documenting the skew."""
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Drop orphan client folders from a previous run with more clients.
    for stale in CLIENTS_DIR.glob("client_*"):
        sfx = stale.name.split("_")[-1]
        if sfx.isdigit() and int(sfx) >= len(splits):
            shutil.rmtree(stale)

    per_client: list[dict] = []
    for i, df in enumerate(splits):
        client_dir = CLIENTS_DIR / f"client_{i:02d}"
        client_dir.mkdir(exist_ok=True)
        df.write_parquet(client_dir / "flows.parquet", compression="snappy")
        if csv:
            df.write_csv(client_dir / "flows.csv")

        class_dist = _class_distribution(df)
        attack_flows = sum(v for k, v in class_dist.items() if k != "benign")
        # busiest dst IP stands in as the hub anchor (downstream reads hub_ip).
        hub_ip = "?"
        if df.height:
            hub_ip = df.group_by(DST).len().sort("len", descending=True).row(0)[0]
        dominant = sorted(
            ((k, v) for k, v in class_dist.items() if k != "benign"),
            key=lambda kv: -kv[1],
        )[:3]

        meta = {
            "client_id": i,
            "hub_ip": hub_ip,
            "n_flows": df.height,
            "n_attack_flows": attack_flows,
            "attack_ratio": round(attack_flows / df.height, 4) if df.height else 0.0,
            "dominant_attacks": [k for k, _ in dominant],
            "n_unique_src": df[SRC].n_unique(),
            "n_unique_dst": df[DST].n_unique(),
            "class_distribution": class_dist,
        }
        (client_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        per_client.append(meta)

    manifest = {
        "strategy": "dirichlet (non-IID label skew)",
        "dirichlet_alpha": alpha,
        "seed": seed,
        "n_clients": len(splits),
        "total_flows": sum(s.height for s in splits),
        "clients": per_client,
    }
    manifest_path = CLIENTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def run(
    n_clients: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
    csv: bool = False,
) -> None:
    """CLI entry: load the cleaned parquet, Dirichlet-partition, save client folders."""
    n_clients = n_clients or DEFAULT_N_CLIENTS
    candidates = sorted(PROCESSED_DIR.glob("cleaned_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No cleaned parquet found — run `fedgnn-data clean` first"
        )
    cleaned = max(candidates, key=lambda p: p.stat().st_mtime)

    print(
        f"[split] strategy : dirichlet label-skew | {n_clients} clients | alpha={alpha}"
    )
    print(f"[split] loading  : {cleaned.name}")
    df = pl.read_parquet(cleaned)
    print(f"[split] {df.height:,} flows | {df[SRC].n_unique()} unique src IPs\n")

    splits = partition(df, n_clients, alpha, seed)
    n_total_classes = int(df[LABEL_COL].n_unique())

    print("[split] per-client class mix (dominant attacks shown):")
    total = sum(s.height for s in splits)
    for i, df_i in enumerate(splits):
        dist = _class_distribution(df_i)
        attacks = sorted(
            ((k, v) for k, v in dist.items() if k != "benign"), key=lambda kv: -kv[1]
        )
        mix = ", ".join(f"{k} {100 * v / df_i.height:.0f}%" for k, v in attacks[:3])
        n_classes = sum(1 for v in dist.values() if v > 0)
        print(
            f"  client_{i:02d}  {df_i.height:>8,} ({100 * df_i.height / total:4.1f}%)  "
            f"{n_classes:>2}/{n_total_classes} classes  |  top attacks: {mix or 'none'}"
        )

    manifest_path = save_splits(splits, alpha, seed, csv=csv)
    fmt = "parquet + CSV" if csv else "parquet"
    print(f"\n[split] wrote {n_clients} client folders ({fmt}) → data/clients/")
    print(f"[split] manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
