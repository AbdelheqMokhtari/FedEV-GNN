# FedEV-GNN

Federated Graph Neural Networks for Intrusion Detection in Connected Electric Mobility.


## Dataset

This project uses the **NF-ToN-IoT-v3** dataset for intrusion detection experiments.

| Property | Value |
|---|---|
| Source | University of Queensland — ML-Based NIDS Datasets |
| Total flows | 27,520,260 |
| Features | 53 NetFlow features + 2 labels = 55 columns |
| Format | CSV (original) → Parquet (auto-converted on first load) |

Official dataset page: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

### Download Instructions

Create the raw dataset directory:

```bash
mkdir -p data/raw/NF-ToN-IoT-v3
cd data/raw/NF-ToN-IoT-v3
```

Download the CSV from the official UQ repository and place it here:

```text
data/raw/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv
```

### Parquet Conversion

The loader uses **Polars** for memory-efficient data handling. On first run it
automatically converts the CSV to Parquet format (compressed from ~5.3 GB to ~0.66 GB)
and saves it alongside the CSV. This conversion runs once and is skipped on all
subsequent runs.

```python
from fedgnn.data.loaders.ton_iot_loader import ToNIoTLoader

loader = ToNIoTLoader()   # converts CSV → Parquet automatically on first run
lf     = loader.load()    # returns a Polars LazyFrame
```

> Parquet is not committed to the repository — it is generated locally from the
> original CSV. See `.gitignore` for details.

### Expected Structure

```text
data/
├── raw/
│   └── NF-ToN-IoT-v3/
│       ├── NF-ToN-IoT-v3.csv        ← download manually
│       └── NF-ToN-IoT-v3.parquet    ← generated automatically on first run
├── processed/
└── graphs/
```

### Stratified Random Subsampling

The full NF-ToN-IoT-v3 dataset contains more than **27 million flows**, which
can be expensive to process during experimentation and rapid prototyping.

To support scalable development workflows, the loader includes an optional
**stratified random subsampling pipeline** that preserves the original attack
class distribution while generating smaller representative subsets.

By default, enabling stratified sampling without specifying `nrows`
creates a balanced subset of:

```python
1_000_000 rows
````

This approach preserves attack distribution 


### Example Usage

```python
from fedgnn.data.loaders.ton_iot_loader import ToNIoTLoader

loader = ToNIoTLoader()

# Load full dataset lazily
lf = loader.load()

# Load first 100k rows
lf = loader.load(nrows=100_000)

# Load stratified subset preserving attack distribution
lf = loader.load(
    stratify=True,
    nrows=1_000_000,
)
```

### Sample Persistence

Generated subsets can be saved locally for reuse during experiments.

Instead of recomputing the stratified sample every time, sampled datasets are
stored under:

```text
data/samples/
```

Example structure:

```text
data/
├── raw/
├── processed/
├── samples/
│   └── NF-ToN-IoT-v3/
│       └── stratified_1M.parquet
└── graphs/
```

### Lazy Loading Architecture

The dataset loader is built using **Polars LazyFrame** execution.

This enables:

* memory-efficient processing,
* deferred execution,
* scalable filtering,
* efficient column projection,
* large-scale dataset handling.

Unlike traditional eager pandas loading, LazyFrames avoid loading the entire
dataset into RAM immediately, making the framework more suitable for large
intrusion detection datasets.

## Setup

## Project Structure

## Development Workflow