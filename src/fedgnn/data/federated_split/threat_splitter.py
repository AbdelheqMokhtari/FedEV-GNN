import json
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLIENTS_DIR = PROJECT_ROOT / "data" / "clients"

SRC = "IPV4_SRC_ADDR"
DST = "IPV4_DST_ADDR"
BENIGN = "benign"

# Fraction of benign flows handed out as an equal floor to every client (the rest
# is load-balanced toward the lighter clients). Guarantees each segment a benign
# baseline so edge classification has a negative class to contrast against.
BENIGN_FLOOR_FRAC = 0.5

# (segment role name, signature attack classes). Order defines the client id.
# The last role is the benign-only segment (no signature attacks).
ROLE_PROFILES: list[tuple[str, list[str]]] = [
    ("charging_controllers", ["ddos"]),
    ("charging_gateways", ["dos"]),
    ("vehicle_telematics", ["scanning"]),
    ("ocpp_backend", ["password", "injection"]),
    ("driver_mobile_app", ["xss"]),
    ("v2g_comms_link", ["mitm"]),
    ("firmware_ota", ["backdoor", "ransomware"]),
    ("field_benign_edge", []),
]

N_CLIENTS = len(ROLE_PROFILES)
ROLE_NAMES = [name for name, _ in ROLE_PROFILES]


def class_to_client() -> dict[str, int]:
    """Map every signature attack class to the client id of the role that owns it."""
    mapping: dict[str, int] = {}
    for cid, (_, classes) in enumerate(ROLE_PROFILES):
        for cls in classes:
            mapping[cls] = cid
    return mapping


def _balance_benign_counts(
    attack_load: list[int], n_benign: int, n_clients: int
) -> list[int]:
    """How many benign flows each client receives: equal floor + deficit fill.

    Every client first gets an equal share of ``BENIGN_FLOOR_FRAC`` of the benign
    pool; the remainder is distributed in proportion to how far each client sits
    below the per-client target size, so heavy (attack-rich) clients get little
    extra benign and light clients get more. Leftover rounding goes to the last
    (benign-edge) client.
    """
    floor_each = int(BENIGN_FLOOR_FRAC * n_benign / n_clients)
    counts = [floor_each] * n_clients

    total_flows = sum(attack_load) + n_benign
    target = total_flows / n_clients
    deficit = [max(0.0, target - attack_load[i] - floor_each) for i in range(n_clients)]
    remaining = n_benign - floor_each * n_clients
    total_deficit = sum(deficit)
    if total_deficit > 0 and remaining > 0:
        for i in range(n_clients):
            counts[i] += int(remaining * deficit[i] / total_deficit)

    counts[-1] += n_benign - sum(counts)  # absorb rounding leftover
    return counts


def partition(df: pl.DataFrame, seed: int = 42) -> list[pl.DataFrame]:
    """Route flows into ``N_CLIENTS`` threat-profile segments. Returns one df each."""
    cmap = class_to_client()

    is_benign = pl.col("Attack") == BENIGN
    attack = df.filter(~is_benign).with_columns(
        pl.col("Attack").replace_strict(cmap, default=-1).cast(pl.Int32).alias("cid")
    )
    benign = df.filter(is_benign)

    attack_load = [attack.filter(pl.col("cid") == i).height for i in range(N_CLIENTS)]
    counts = _balance_benign_counts(attack_load, benign.height, N_CLIENTS)

    # Build a shuffled cid column for the benign flows matching the target counts.
    rng = np.random.default_rng(seed)
    benign_cids = np.repeat(np.arange(N_CLIENTS), counts)
    rng.shuffle(benign_cids)
    benign = benign.with_columns(pl.Series("cid", benign_cids).cast(pl.Int32))

    tagged = pl.concat([attack, benign])
    return [tagged.filter(pl.col("cid") == i).drop("cid") for i in range(N_CLIENTS)]


def _class_distribution(df: pl.DataFrame) -> dict[str, int]:
    if "Attack" not in df.columns:
        return {}
    return dict(df.group_by("Attack").len().sort("Attack").iter_rows())


def save_splits(splits: list[pl.DataFrame], csv: bool = False) -> Path:
    """Write one folder per segment plus a manifest documenting the heterogeneity."""
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove orphan client folders from a previous run with more clients, so the
    # set of client_* folders always matches exactly this split.
    import shutil

    for stale in CLIENTS_DIR.glob("client_*"):
        idx = stale.name.split("_")[-1]
        if idx.isdigit() and int(idx) >= len(splits):
            shutil.rmtree(stale)

    per_client: list[dict] = []
    for i, df in enumerate(splits):
        role, signature = ROLE_PROFILES[i]
        client_dir = CLIENTS_DIR / f"client_{i:02d}"
        client_dir.mkdir(exist_ok=True)

        df.write_parquet(client_dir / "flows.parquet", compression="snappy")
        if csv:
            df.write_csv(client_dir / "flows.csv")

        class_dist = _class_distribution(df)
        attack_flows = sum(v for k, v in class_dist.items() if k != BENIGN)
        # hub_ip kept for downstream compatibility (graph_meta/training read it):
        # the segment's busiest IP stands in as the anchor.
        hub_ip = "?"
        if df.height:
            hub_ip = df.group_by(DST).len().sort("len", descending=True).row(0)[0]

        meta = {
            "client_id": i,
            "role": role,
            "signature_attacks": signature,
            "hub_ip": hub_ip,
            "n_flows": df.height,
            "n_attack_flows": attack_flows,
            "attack_ratio": round(attack_flows / df.height, 4) if df.height else 0.0,
            "n_unique_src": df[SRC].n_unique(),
            "n_unique_dst": df[DST].n_unique(),
            "class_distribution": class_dist,
        }
        (client_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        per_client.append(meta)

    manifest = {
        "strategy": "threat-profile (non-IID label skew)",
        "n_clients": len(splits),
        "total_flows": sum(s.height for s in splits),
        "roles": {i: ROLE_PROFILES[i][0] for i in range(len(splits))},
        "clients": per_client,
    }
    manifest_path = CLIENTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def run(csv: bool = False, seed: int = 42) -> None:
    """CLI entry: load the cleaned parquet, route into segments, save client folders."""
    candidates = sorted(PROCESSED_DIR.glob("cleaned_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No cleaned parquet found — run `fedgnn-data clean` first"
        )
    cleaned = max(candidates, key=lambda p: p.stat().st_mtime)

    print(f"[split] strategy : threat-profile (non-IID) | {N_CLIENTS} segments")
    print(f"[split] loading  : {cleaned.name}")
    df = pl.read_parquet(cleaned)
    print(f"[split] {df.height:,} flows | {df[SRC].n_unique()} unique src IPs\n")

    splits = partition(df, seed=seed)

    print("[split] segment summary (role — flows — threat profile):")
    total = sum(s.height for s in splits)
    for i, df_i in enumerate(splits):
        role, sig = ROLE_PROFILES[i]
        dist = _class_distribution(df_i)
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
        mix = ", ".join(f"{k} {100 * v / df_i.height:.0f}%" for k, v in top)
        sig_s = "+".join(sig) if sig else "benign-only"
        print(
            f"  client_{i:02d}  {role:<22} {df_i.height:>8,} "
            f"({100 * df_i.height / total:4.1f}%)  [{sig_s}]  {mix}"
        )

    manifest_path = save_splits(splits, csv=csv)
    fmt = "parquet + CSV" if csv else "parquet"
    print(f"\n[split] wrote {N_CLIENTS} client folders ({fmt}) → data/clients/")
    print(f"[split] manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
