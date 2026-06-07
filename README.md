# FedEV-GNN

Federated Graph Neural Networks for Intrusion Detection in Connected Electric Mobility.

The framework models network traffic as a directed multigraph — IP addresses
as nodes, NetFlow records as edges — and trains a Temporal GNN across
federated clients without sharing raw traffic data.

---

## Dataset

**NF-ToN-IoT-v3** — 27.5 M NetFlow records, 10 attack classes + benign.

| Property | Value |
|---|---|
| Source | University of Queensland — ML-Based NIDS Datasets |
| Total flows | 27,520,260 |
| Features | 53 NetFlow features + 2 labels = 55 columns |
| Format | CSV (original) → Parquet (auto-converted on first load) |

Download: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

Place the CSV at:
```
data/raw/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eda]"   # core + notebook/vis extras
```

---

## Data Pipeline CLI — `fedgnn-data`

All pipeline stages are driven by a single console script.

```
fedgnn-data <command> [options]
```

Equivalent to `python -m fedgnn.data <command>`.

### Full pipeline (run in order)

```bash
# 1. Draw a 1 M stratified sample from the raw 5.3 GB CSV
fedgnn-data sample --rows 1000000

# 2. Clean the sample (drop invalid flows, encode labels)
fedgnn-data clean

# 3. Validate and write the 26-feature set
fedgnn-data select

# 4. Partition into 10 federated client folders
fedgnn-data split

# 5. Build PyG graphs (static + temporal) for T-GAT training
fedgnn-data build
```

---

### `load` — inspect any dataset stage

```bash
fedgnn-data load                           # raw dataset
fedgnn-data load --sample                  # latest sample
fedgnn-data load --processed               # latest cleaned parquet
fedgnn-data load --processed --rows 10000  # specific cleaned file
fedgnn-data load --split 3                 # federated client 3
fedgnn-data load --split client_00         # same, by folder name
fedgnn-data load --head 10                 # show 10 preview rows (default 5)
```

Prints: source path · shape · row preview · attack distribution · grouped
feature catalogue (topology / selected features by group / not selected /
class map).

---

### `sample` — stratified subsample

```bash
fedgnn-data sample --rows 1000000          # default
fedgnn-data sample --rows 50000 --seed 0
fedgnn-data sample --rows 1000000 --csv    # also write .csv copy
```

Writes `data/samples/ton_iot_{n}.parquet`. The CSV copy is for inspection
only; the pipeline always reads parquet.

---

### `clean` — non-destructive cleaning

```bash
fedgnn-data clean               # newest sample
fedgnn-data clean --rows 50000  # ton_iot_50000.parquet specifically
```

Steps: drop 0.0.0.0 / self-loops · lowercase Attack · fill nulls · drop
constant columns · add `y_binary` + `y_multiclass` with a global-stable
label map (`benign=0`, attacks alphabetical `1..N`).

Writes `data/processed/cleaned_{n}.parquet` + `feature_columns.json` +
`label_map.json`.

---

### `select` — fixed 26-feature set

```bash
fedgnn-data select
```

Validates and writes the domain-driven 26-feature set to
`data/processed/selected_features.json`.

| Group | Count | Purpose |
|---|---|---|
| Behavioral | 6 | protocol identity, ports, TTL, ICMP |
| Temporal | 10 | flow duration, IAT statistics, TCP flags |
| Content & Volume | 10 | byte/packet volume, size distribution |

---

### `split` — federated hub partitioning

```bash
fedgnn-data split                  # 10 clients, parquet only
fedgnn-data split --csv            # also write flows.csv per client
fedgnn-data split --n-clients 5    # fewer clients
```

**Strategy:** scores every IP by `src_flows + dst_flows`, selects the
top-N as hub anchors, then assigns each flow to the hub it communicates
with most directly (source hub > destination hub > least-loaded client).
Preserves local graph neighbourhoods without any clustering overhead.

Writes one folder per client:

```
data/clients/
  client_00/  flows.parquet  [flows.csv]  meta.json
  client_01/  ...
  manifest.json
```

Each `meta.json` records the hub IP, traffic score, flow count, unique
src/dst count, and per-class distribution.

---

### `build` — PyG graph construction for T-GAT

```bash
fedgnn-data build                       # all clients, 1-hour snapshot windows
fedgnn-data build --window-ms 1800000   # 30-minute windows
fedgnn-data build --n-clients 2         # only the first 2 clients (debug)
```

> Requires the `gnn` extra: `pip install -e ".[gnn]"` (`torch`,
> `torch-geometric`, `scipy`). Every other `fedgnn-data` command works
> without it; `build` exits with an install hint if it's missing.

Turns each client's `flows.parquet` into the PyG graphs T-GAT trains on.
**Nodes = unique device IPs, edges = flows** — a normal communication graph;
the only "temporal" twist is that flows are bucketed by
`FLOW_START_MILLISECONDS` into fixed-width windows and **one graph snapshot
is built per window**, because T-GAT needs that ordered sequence to learn how
the graph evolves, not just a single frozen view.

Each snapshot is a `torch_geometric.data.Data` with `x` (the 26 selected
features, mean-aggregated over each node's incident in+out flows), `edge_index`,
`edge_attr` (the same 26 raw features kept per-edge), and `y` (`y_multiclass`).

Writes, alongside each client's `flows.parquet`:

| File | Contents |
|---|---|
| `graph.pt` | one whole-client static `Data` graph — training-ready single graph |
| `graphs.pt` | ordered `list[Data]` snapshot sequence — what T-GAT unrolls over |
| `graph.mat` | the same static graph as `graph.pt`, mirrored to `.mat` for MATLAB/Octave/scipy inspection |
| `graph_meta.json` | `graph` block (static-graph nodes/edges/class distribution, the one you'll likely read first) + `temporal` block (sequence stats) |

It also prints — and writes to the global `data/processed/tgat_param_count.json`
(shared by every client, since the architecture doesn't vary per client) — the
theoretical parameter count of the local T-GAT model with a per-layer
breakdown: `GAT(26→64, heads=2)` → `GAT(128→32, heads=1)` →
`Linear(32→num_classes)`.

---

## Validation CLI — `fedgnn-validate`

Turns the JSON a training run leaves under `Results/<exp>/` into evaluation
**figures** under `Figures/<exp>/` — loss curves per client, bar charts of how
good each client and the global (server) model are, and the global-metric /
aggregation-weight evolution. Read-only: it re-draws what training recorded, never
re-runs a model, so it works on a finished *or* interrupted run.

```bash
pip install -e ".[viz]"        # matplotlib

fedgnn-validate list                       # show available experiments
fedgnn-validate plot                       # latest experiment -> Figures/<latest>/
fedgnn-validate plot fedgnn05              # a specific experiment
fedgnn-validate plot --all                 # every experiment under Results/
fedgnn-validate plot fedgnn02 --format pdf --dpi 200 --verbose
```

Output per experiment `<exp>`:

```
Figures/<exp>/
├── loss_all_clients.png               every client's training loss, overlaid
├── client_quality.png                 grouped bars — how good each client is
├── clients/
│   ├── client_NN_loss.png             one client's loss curve
│   └── client_NN_metrics.png          its local-vs-global metric trends
└── server/
    ├── global_metrics_evolution.png   the four headline metrics per round
    ├── server_quality.png             bars — how good the global model is
    └── aggregation_weights.png        each client's FedAvg weight share per round
```

Quality bars use the held-out **test** metrics when present, and fall back to the
last-round **global** metrics for an interrupted run (noted in the title). See
[docs/validation-pipeline.md](docs/validation-pipeline.md) for the full reference.

---

## Output Layout

```
data/
├── raw/NF-ToN-IoT-v3/
│   └── NF-ToN-IoT-v3.csv          ← download manually
├── samples/
│   └── ton_iot_1000000.parquet     ← fedgnn-data sample
├── processed/
│   ├── cleaned_1000000.parquet     ← fedgnn-data clean
│   ├── feature_columns.json        ← all 48 edge features
│   ├── label_map.json              ← class → index (shared)
│   ├── selected_features.json      ← fixed 26-feature subset
│   └── tgat_param_count.json       ← fedgnn-data build (global, shared by all clients)
└── clients/
    ├── client_00/                  ← fedgnn-data split (+ build)
    │   ├── flows.parquet
    │   ├── meta.json
    │   ├── graph.pt                ← fedgnn-data build (whole-client static graph)
    │   ├── graphs.pt               ← fedgnn-data build (temporal snapshot sequence)
    │   ├── graph.mat               ← fedgnn-data build (same static graph, .mat)
    │   └── graph_meta.json         ← fedgnn-data build
    ├── ...
    └── manifest.json
```

---

## Python API

```python
from fedgnn.data.loaders import ToNIoTLoader

# Load full dataset lazily
loader = ToNIoTLoader()
lf = loader.load()                              # Polars LazyFrame

# Load latest sample
loader = ToNIoTLoader(sample=True)
lf = loader.load(nrows=100_000, stratify=True)
df = lf.collect()
```

```python
from fedgnn.data.preprocessing.cleaning import clean
from fedgnn.data.preprocessing.feature_selection import SELECTED, GROUPS
from fedgnn.data.federated_split.hub_splitter import compute_hub_scores, partition
```

```python
# Requires the `gnn` extra (torch, torch-geometric, scipy)
from fedgnn.data.graph_builder import build_snapshot, build_snapshot_sequence

import torch
graph = torch.load("data/clients/client_00/graph.pt", weights_only=False)
sequence = torch.load("data/clients/client_00/graphs.pt", weights_only=False)
```

---

## Project Structure

```
src/fedgnn/
├── data/
│   ├── loaders/            ToNIoTLoader (Polars lazy/streaming)
│   ├── preprocessing/      cleaning.py · feature_selection.py
│   ├── graph_builder/      edge_graph.py · temporal_graph.py
│   ├── federated_split/    hub_splitter.py
│   └── cli.py              fedgnn-data entry point
├── train/                 federated FedAvg simulation (fedgnn-train)
├── evaluation/            metrics · evaluator · FLOPs accounting
└── validation/            results → figures (fedgnn-validate)
    ├── loader.py          re-assembles Results/<exp>/ into ExperimentResults
    ├── plots.py           loss curves · quality bars · server dynamics
    └── cli.py             fedgnn-validate entry point
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

Pipeline documentation:
[docs/data_pipeline.md](docs/data_pipeline.md) ·
[docs/train-pipeline.md](docs/train-pipeline.md) ·
[docs/validation-pipeline.md](docs/validation-pipeline.md)
