"""Hub-centered graph partitioning for federated IDS on NF-ToN-IoT-v3.

Strategy
--------
1.  Score every IP: score = src_flow_count + dst_flow_count.
2.  The top N IPs by score become hub anchors — one per federated client.
    High-traffic IPs in IoT networks (switches, attacked servers, gateways)
    concentrate most of the graph's communication, so anchoring partitions
    on them preserves local graph neighbourhoods and produces realistic
    non-IID splits without any clustering overhead.
3.  Assign each flow by hub affinity (source priority):
        src IP is a hub  → that hub's client
        dst IP is a hub  → that hub's client  (src wins on tie)
        neither          → least-loaded client so far
4.  Save each partition to data/clients/client_{i:02d}/.

Notes
-----
* 192.168.1.1 is expected to be the network gateway. It receives its own
  client (typically client_05) and is not given special treatment beyond
  what the score-based ranking already implies — the per-flow affinity
  rules keep it from dominating other clients.
* IPs are treated as logical communication entities, not guaranteed
  unique physical devices.

Invoked via the data CLI::

    python -m fedgnn.data split [--csv] [--n-clients 10]
"""

import json
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLIENTS_DIR = PROJECT_ROOT / "data" / "clients"

SRC = "IPV4_SRC_ADDR"
DST = "IPV4_DST_ADDR"
N_CLIENTS = 10


def compute_hub_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Return (ip, src_flows, dst_flows, score) sorted by score descending."""
    src = df.group_by(SRC).len().rename({SRC: "ip", "len": "src_flows"})
    dst = df.group_by(DST).len().rename({DST: "ip", "len": "dst_flows"})
    return (
        src.join(dst, on="ip", how="full", coalesce=True)
        .fill_null(0)
        .with_columns((pl.col("src_flows") + pl.col("dst_flows")).alias("score"))
        .sort("score", descending=True)
    )


def select_hubs(scores: pl.DataFrame, n: int = N_CLIENTS) -> list[str]:
    """Pick top-n IPs by traffic score as client hub anchors."""
    return scores.head(n)["ip"].to_list()


def partition(df: pl.DataFrame, hubs: list[str]) -> list[pl.DataFrame]:
    """Assign flows to clients by hub affinity; unmatched go to least-loaded.

    Returns one DataFrame per client in hub order.
    """
    n = len(hubs)
    hub_df = pl.DataFrame(
        {
            "ip": hubs,
            "cid": list(range(n)),
        }
    ).with_columns(pl.col("cid").cast(pl.Int32))

    # Vectorised affinity join: src hub wins over dst hub.
    tagged = (
        df.join(hub_df.rename({"ip": SRC, "cid": "src_cid"}), on=SRC, how="left")
        .join(hub_df.rename({"ip": DST, "cid": "dst_cid"}), on=DST, how="left")
        .with_columns(pl.coalesce(["src_cid", "dst_cid"]).alias("cid"))
    )

    matched = tagged.filter(pl.col("cid").is_not_null())
    unmatched = tagged.filter(pl.col("cid").is_null())

    # Seed per-client counts from already-matched flows for load balancing.
    client_counts = [0] * n
    for cid, count in matched.group_by("cid").len().iter_rows():
        client_counts[int(cid)] = count

    # Assign each unmatched flow to the current least-loaded client.
    assignments: list[int] = []
    for _ in range(unmatched.height):
        c = int(min(range(n), key=lambda i: client_counts[i]))
        assignments.append(c)
        client_counts[c] += 1

    if unmatched.height > 0:
        unmatched = unmatched.with_columns(
            pl.Series("cid", assignments, dtype=pl.Int32)
        )
        tagged = pl.concat([matched, unmatched])
    else:
        tagged = matched

    return [
        tagged.filter(pl.col("cid") == i).drop("src_cid", "dst_cid", "cid")
        for i in range(n)
    ]


def save_splits(
    splits: list[pl.DataFrame],
    hubs: list[str],
    scores: pl.DataFrame,
    csv: bool = False,
) -> Path:
    """Write one client folder per split plus a top-level manifest.

    Layout::

        data/clients/
          client_00/  flows.parquet  [flows.csv]  meta.json
          client_01/  ...
          manifest.json
    """
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

    score_lookup: dict[str, dict] = {
        row[0]: {"src_flows": row[1], "dst_flows": row[2], "score": row[3]}
        for row in scores.iter_rows()
    }

    per_client: list[dict] = []
    for i, (df, hub) in enumerate(zip(splits, hubs)):
        client_dir = CLIENTS_DIR / f"client_{i:02d}"
        client_dir.mkdir(exist_ok=True)

        df.write_parquet(client_dir / "flows.parquet", compression="snappy")
        if csv:
            df.write_csv(client_dir / "flows.csv")

        class_dist: dict[str, int] = {}
        if "Attack" in df.columns:
            class_dist = dict(df.group_by("Attack").len().sort("Attack").iter_rows())

        meta = {
            "client_id": i,
            "hub_ip": hub,
            "hub_stats": score_lookup.get(hub, {}),
            "n_flows": df.height,
            "n_unique_src": df[SRC].n_unique(),
            "n_unique_dst": df[DST].n_unique(),
            "class_distribution": class_dist,
        }
        (client_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        per_client.append(meta)

    manifest = {
        "n_clients": len(splits),
        "total_flows": sum(s.height for s in splits),
        "hub_ips": hubs,
        "clients": per_client,
    }
    manifest_path = CLIENTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def run(n_clients: int = N_CLIENTS, csv: bool = False) -> None:
    """CLI entry: load the cleaned parquet, partition, and save client folders."""
    candidates = sorted(PROCESSED_DIR.glob("cleaned_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No cleaned parquet found — run `fedgnn-data clean` first"
        )
    cleaned = max(candidates, key=lambda p: p.stat().st_mtime)

    print(f"[split] loading: {cleaned.name}")
    df = pl.read_parquet(cleaned)
    print(f"[split] {df.height:,} flows | {df[SRC].n_unique()} unique IPs\n")

    scores = compute_hub_scores(df)
    hubs = select_hubs(scores, n_clients)

    print(f"[split] top-{n_clients} hub anchors:")
    for i, hub in enumerate(hubs):
        r = scores.filter(pl.col("ip") == hub).row(0)
        _, src_f, dst_f, score = r
        print(
            f"  client_{i:02d}  {hub:<18}  score={score:>8,}  "
            f"(src={src_f:,}  dst={dst_f:,})"
        )

    print(f"\n[split] assigning {df.height:,} flows to {n_clients} clients ...")
    splits = partition(df, hubs)

    print("\n[split] partition summary:")
    total = sum(s.height for s in splits)
    for i, (split, hub) in enumerate(zip(splits, hubs)):
        pct = 100.0 * split.height / total
        print(f"  client_{i:02d}  {hub:<18}  {split.height:>8,} flows  ({pct:.1f}%)")

    manifest_path = save_splits(splits, hubs, scores, csv=csv)
    fmt = "parquet + CSV" if csv else "parquet"
    print(f"\n[split] wrote {n_clients} client folders ({fmt}) → data/clients/")
    print(f"[split] manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
