"""Explicit, domain-driven feature selection over the *engineered* feature set.

The cleaning stage (:mod:`fedgnn.data.preprocessing.engineering`) decomposes raw
NetFlow codes/bitmasks/counters into model-ready signals. This module curates
which of them feed the GAT, organised into eight semantic groups. Every selected
feature is non-negative (``log1p``-safe) and either bounded (flags/fractions) or
heavy-tailed-but-positive (volumes/IATs) — see the engineering module for why.

  Protocol (3)          — one-hot transport protocol (TCP/UDP/ICMP)
  Application (11)       — one-hot top-10 nDPI L7 protocols + l7_other
  Service (5)           — destination-port service buckets (http/https/dns/…)
  TCP state (10)        — decomposed client+server TCP-flag bits
  Volume (4)            — raw in/out byte & packet counts (log-compressed)
  Volume ratios (7)     — asymmetry, retransmission rate, throughput, duration
  Packet size (6)       — per-size-bucket fractions + packet-length spread
  Temporal (6)          — flow duration, duration ratio, inter-arrival timing
  Reachability (3)      — TTL min/max and hop-count spread

These features are chosen by design; the choice is advisory (the raw columns are
all retained in the cleaned parquet) and meant to be revisited with EDA.

Invoked via the data CLI::

    python -m fedgnn.data select
"""

import json
import textwrap
from pathlib import Path

from fedgnn.data.preprocessing.engineering import L7_TOP

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Feature groups (engineered names are lower-case; retained raw NetFlow are UPPER-case)

PROTOCOL: list[str] = [
    "is_tcp",
    "is_udp",
    "is_icmp",
]

# Application protocol: top-10 nDPI ids (see engineering.L7_TOP) + catch-all.
APPLICATION: list[str] = [f"l7_{v}" for v in L7_TOP] + ["l7_other"]

SERVICE: list[str] = [
    "port_http",
    "port_https",
    "port_dns",
    "port_wellknown",
    "port_ephemeral",
]

TCP_STATE: list[str] = [
    "cf_syn",
    "cf_ack",
    "cf_fin",
    "cf_rst",
    "cf_psh",
    "sf_syn",
    "sf_ack",
    "sf_fin",
    "sf_rst",
    "sf_psh",
]

VOLUME: list[str] = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
]

VOLUME_RATIO: list[str] = [
    "bytes_per_pkt_in",
    "bytes_per_pkt_out",
    "byte_ratio",
    "pkt_ratio",
    "retrans_rate_in",
    "retrans_rate_out",
    "throughput_ratio",
]

SIZE: list[str] = [
    "frac_pkt_128",
    "frac_pkt_256_512",
    "frac_pkt_512_1024",
    "frac_pkt_1024_1514",
    "MAX_IP_PKT_LEN",
    "pkt_len_range",
]

TEMPORAL: list[str] = [
    "FLOW_DURATION_MILLISECONDS",
    "dur_ratio",
    "SRC_TO_DST_IAT_AVG",
    "DST_TO_SRC_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_STDDEV",
]

REACHABILITY: list[str] = [
    "MIN_TTL",
    "MAX_TTL",
    "ttl_range",
]

GROUPS: dict[str, list[str]] = {
    "protocol": PROTOCOL,
    "application": APPLICATION,
    "service": SERVICE,
    "tcp_state": TCP_STATE,
    "volume": VOLUME,
    "volume_ratio": VOLUME_RATIO,
    "size": SIZE,
    "temporal": TEMPORAL,
    "reachability": REACHABILITY,
}

SELECTED: list[str] = [f for feats in GROUPS.values() for f in feats]


def run() -> None:
    """Validate the fixed feature set against the cleaned parquet and write the JSON."""
    available = json.loads((PROCESSED_DIR / "feature_columns.json").read_text())
    available_set = set(available)

    missing = [f for f in SELECTED if f not in available_set]
    if missing:
        print(
            f"[select] WARNING — {len(missing)} feature(s) not found in cleaned data:"
        )
        for f in missing:
            print(f"           - {f}")

    kept = [f for f in SELECTED if f in available_set]
    (PROCESSED_DIR / "selected_features.json").write_text(json.dumps(kept, indent=2))

    GROUP_LABELS = {
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
    WIDTH = 72
    INDENT = "    "

    print(f"\n[select] {len(kept)}/{len(SELECTED)} features confirmed\n")
    for group, feats in GROUPS.items():
        confirmed = [f for f in feats if f in available_set]
        n_total = len(feats)
        n_ok = len(confirmed)
        label = GROUP_LABELS.get(group, group)
        missing_note = f"  ⚠ {n_total - n_ok} missing" if n_ok < n_total else ""
        header = f"  ── {label} ({n_ok}){missing_note} "
        print(header + "─" * max(0, WIDTH - len(header) + 2))
        lines = textwrap.wrap(", ".join(confirmed), width=WIDTH - len(INDENT))
        for line in lines:
            print(INDENT + line)
        print()
    print("[select] wrote data/processed/selected_features.json")
