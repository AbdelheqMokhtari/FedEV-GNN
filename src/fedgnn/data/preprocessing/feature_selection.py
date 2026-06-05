"""Explicit, domain-driven feature selection.

Feature set derived from the NF-ToN-IoT-v3 literature and EDA. Three semantic
groups cover the signals needed to distinguish all attack classes:

  Behavioral (6)        — protocol identity and reachability (ports, TTL, ICMP)
  Temporal (10)         — flow timing and TCP state (IAT, duration, flags)
  Content & Volume (10) — byte/packet volume and size distribution

These 26 features are fixed by design, not learned.

Invoked via the data CLI::

    python -m fedgnn.data select
"""

import json
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Feature groups

BEHAVIORAL: list[str] = [
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "MIN_TTL",
    "MAX_TTL",
    "ICMP_TYPE",
]

TEMPORAL: list[str] = [
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "SRC_TO_DST_IAT_AVG",
    "DST_TO_SRC_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_STDDEV",
]

CONTENT_VOLUME: list[str] = [
    "IN_BYTES",
    "OUT_BYTES",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
]

GROUPS: dict[str, list[str]] = {
    "behavioral": BEHAVIORAL,
    "temporal": TEMPORAL,
    "content_volume": CONTENT_VOLUME,
}

SELECTED: list[str] = BEHAVIORAL + TEMPORAL + CONTENT_VOLUME


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
        "behavioral": "Behavioral",
        "temporal": "Temporal",
        "content_volume": "Content & Volume",
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
