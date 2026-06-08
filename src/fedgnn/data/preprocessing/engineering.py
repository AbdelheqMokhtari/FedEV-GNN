"""Feature engineering for NF-ToN-IoT-v3 flows.

Turns raw NetFlow columns into model-ready signals *before* selection. Raw flow
records mix three kinds of column that a GNN cannot consume as-is:

  * **categorical codes** — ``PROTOCOL`` (6=TCP, 17=UDP, …), ``L7_PROTO`` (nDPI
    app protocol: HTTP/TLS/DNS/…) and ``L4_DST_PORT`` (80, 443, 53, …). Their
    integer value carries no ordinal meaning, so feeding them as continuous numbers
    is wrong. We one-hot them into binary indicators (``L7_PROTO`` is high-cardinality
    so we keep only the top-10 protocols and bucket the rest into ``l7_other``).
  * **bitmasks** — ``CLIENT_TCP_FLAGS`` / ``SERVER_TCP_FLAGS`` are *summed* TCP
    flag bits (SYN=2, ACK=16, …). The integer ``18`` literally means "SYN+ACK";
    a model can only recover that if we decompose it into the individual bits.
  * **raw counters** — byte/packet/retransmission counts that only become
    discriminative as *ratios* (asymmetry, retransmission rate, avg packet size)
    and as *distributions* (per-size-bucket fractions) rather than raw volumes.

Non-negativity contract
-----------------------
Every engineered column is **non-negative by construction**. Training applies
``log1p(clip(x, 0, None))`` then a z-score (see :mod:`fedgnn.train.client`), so a
feature that could go negative would be silently clipped to 0 and lose its signal.
Ratios therefore use a ``+1`` denominator (finite and ``>= 0``); ranges are
``max - min`` (``>= 0`` since max ``>=`` min); flags / one-hots are ``0``/``1``.

Why this works for GAT
----------------------
Edge features are mean-aggregated onto graph nodes
(:func:`fedgnn.data.graph_builder.edge_graph.aggregate_node_features`). A ``0``/``1``
flag thus becomes *"the fraction of a node's flows that had this flag set"* — a
smooth, bounded signal that GAT attention can compare across nodes that mostly
send, mostly receive, or do both.
"""

import polars as pl

# TCP control-flag bits (RFC 9293). Decomposing the summed flag bitmask into these
# recovers the per-flow behaviour: SYN-only = scan, SYN+ACK = handshake, RST = reset.
TCP_BITS: dict[str, int] = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16}

# IANA protocol numbers worth distinguishing (the dataset only carries 1/2/6/17).
PROTO_TCP, PROTO_UDP, PROTO_ICMP = 6, 17, 1

# L7_PROTO is the nDPI application protocol: 123 distinct ids, but the 10 below
# cover ~98% of flows. Fixed list (computed from the 1M sample, ordered by
# frequency) so the engineered column set stays identical across clients/runs —
# FedAvg only works if every client shares the same feature space.
# Known names: 0=Unknown, 7=HTTP, 91=TLS, 5=DNS.
L7_TOP: list[int] = [0, 7, 91, 5, 1, 81, 131, 12, 3, 92]

# Service ports: identity, not magnitude. 80/443/53 dominate; ephemeral = client side.
HTTP_PORTS = [80, 8080]
HTTPS_PORTS = [443, 8443]
DNS_PORT = 53
WELL_KNOWN_MAX = 1024
EPHEMERAL_MIN = 49152


def _protocol_onehot() -> list[pl.Expr]:
    """``PROTOCOL`` code -> binary protocol indicators (IGMP falls through to all-0)."""
    p = pl.col("PROTOCOL")
    return [
        (p == PROTO_TCP).cast(pl.Int8).alias("is_tcp"),
        (p == PROTO_UDP).cast(pl.Int8).alias("is_udp"),
        (p == PROTO_ICMP).cast(pl.Int8).alias("is_icmp"),
    ]


def _l7_proto_onehot() -> list[pl.Expr]:
    """``L7_PROTO`` -> one-hot of the top-10 app prt + an ``l7_other`` catch-all."""
    p = pl.col("L7_PROTO")
    exprs = [(p == v).cast(pl.Int8).alias(f"l7_{v}") for v in L7_TOP]
    exprs.append((~p.is_in(L7_TOP)).cast(pl.Int8).alias("l7_other"))
    return exprs


def _service_buckets() -> list[pl.Expr]:
    """``L4_DST_PORT`` -> service-category indicators (overlapping, not exclusive)."""
    p = pl.col("L4_DST_PORT")
    return [
        p.is_in(HTTP_PORTS).cast(pl.Int8).alias("port_http"),
        p.is_in(HTTPS_PORTS).cast(pl.Int8).alias("port_https"),
        (p == DNS_PORT).cast(pl.Int8).alias("port_dns"),
        (p < WELL_KNOWN_MAX).cast(pl.Int8).alias("port_wellknown"),
        (p >= EPHEMERAL_MIN).cast(pl.Int8).alias("port_ephemeral"),
    ]


def _flag_bits(col: str, prefix: str) -> list[pl.Expr]:
    """Decompose a summed TCP-flag bitmask column into one 0/1 column per flag."""
    return [
        (((pl.col(col) // bit) % 2).cast(pl.Int8)).alias(f"{prefix}_{name}")
        for name, bit in TCP_BITS.items()
    ]


def _ratios() -> list[pl.Expr]:
    """Asymmetry / rate features from raw counters (``+1`` keeps them finite, >= 0)."""
    return [
        (pl.col("IN_BYTES") / (pl.col("IN_PKTS") + 1)).alias("bytes_per_pkt_in"),
        (pl.col("OUT_BYTES") / (pl.col("OUT_PKTS") + 1)).alias("bytes_per_pkt_out"),
        (pl.col("IN_BYTES") / (pl.col("OUT_BYTES") + 1)).alias("byte_ratio"),
        (pl.col("IN_PKTS") / (pl.col("OUT_PKTS") + 1)).alias("pkt_ratio"),
        (pl.col("RETRANSMITTED_IN_BYTES") / (pl.col("IN_BYTES") + 1)).alias(
            "retrans_rate_in"
        ),
        (pl.col("RETRANSMITTED_OUT_BYTES") / (pl.col("OUT_BYTES") + 1)).alias(
            "retrans_rate_out"
        ),
        (
            pl.col("SRC_TO_DST_AVG_THROUGHPUT")
            / (pl.col("DST_TO_SRC_AVG_THROUGHPUT") + 1)
        ).alias("throughput_ratio"),
        (pl.col("DURATION_IN") / (pl.col("DURATION_OUT") + 1)).alias("dur_ratio"),
    ]


def _size_distribution() -> list[pl.Expr]:
    """Packet-size *fractions* (scale-free) from the raw per-bucket packet counts."""
    total = (
        pl.col("NUM_PKTS_UP_TO_128_BYTES")
        + pl.col("NUM_PKTS_128_TO_256_BYTES")
        + pl.col("NUM_PKTS_256_TO_512_BYTES")
        + pl.col("NUM_PKTS_512_TO_1024_BYTES")
        + pl.col("NUM_PKTS_1024_TO_1514_BYTES")
        + 1
    )
    return [
        (pl.col("NUM_PKTS_UP_TO_128_BYTES") / total).alias("frac_pkt_128"),
        (pl.col("NUM_PKTS_256_TO_512_BYTES") / total).alias("frac_pkt_256_512"),
        (pl.col("NUM_PKTS_512_TO_1024_BYTES") / total).alias("frac_pkt_512_1024"),
        (pl.col("NUM_PKTS_1024_TO_1514_BYTES") / total).alias("frac_pkt_1024_1514"),
    ]


def _ranges() -> list[pl.Expr]:
    """Spread features: max - min (>= 0). Hop-count and packet-length variability."""
    return [
        (pl.col("MAX_TTL") - pl.col("MIN_TTL")).alias("ttl_range"),
        (pl.col("MAX_IP_PKT_LEN") - pl.col("MIN_IP_PKT_LEN")).alias("pkt_len_range"),
    ]


def engineer_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Append all engineered columns; return ``(df, new_column_names)``.

    Source raw columns are kept (non-destructive) so selection stays advisory and
    later runs can revisit the choice. ``new_column_names`` is computed by diffing
    the schema so it always matches what was actually added.
    """
    exprs = (
        _protocol_onehot()
        + _l7_proto_onehot()
        + _service_buckets()
        + _flag_bits("CLIENT_TCP_FLAGS", "cf")
        + _flag_bits("SERVER_TCP_FLAGS", "sf")
        + _ratios()
        + _size_distribution()
        + _ranges()
    )
    before = set(df.columns)
    df = df.with_columns(exprs)
    new_cols = [c for c in df.columns if c not in before]
    # Float features default to Float64 -> downcast to match the raw NetFlow columns
    # and keep the parquet compact (the model casts everything to float32 anyway).
    float_new = [c for c in new_cols if df.schema[c] == pl.Float64]
    if float_new:
        df = df.with_columns(pl.col(c).cast(pl.Float32) for c in float_new)
    return df, new_cols
