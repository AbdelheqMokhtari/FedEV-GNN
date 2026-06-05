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
│   └── selected_features.json      ← fixed 26-feature subset
└── clients/
    ├── client_00/                  ← fedgnn-data split
    │   ├── flows.parquet
    │   └── meta.json
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
└── ...
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

Full pipeline documentation: [docs/data_pipeline.md](docs/data_pipeline.md)
