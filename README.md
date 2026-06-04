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

## Setup

## Project Structure

## Development Workflow