from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from fedgnn.data.loaders.base_loader import BaseLoader


class ToNIoTLoader(BaseLoader):
    """Loader for NF-ToN-IoT-v3 dataset."""

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DEFAULT_PATH = (
        PROJECT_ROOT / "data" / "raw" / "NF-ToN-IoT-v3" / "NF-ToN-IoT-v3.parquet"
    )

    def __init__(self, data_path: str | Path | None = None):
        super().__init__(data_path or self.DEFAULT_PATH)
        if not self.data_path.exists():
            self._convert()

    def _convert(self) -> None:
        """Convert the CSV to Parquet (runs automatically once)."""
        csv_path = self.data_path.with_suffix(".csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        print("[ToNIoTLoader] First run — converting CSV → Parquet...")
        pl.scan_csv(csv_path).sink_parquet(self.data_path, compression="snappy")
        print(f"[ToNIoTLoader] Done → {self.data_path}")

    def load(
        self,
        nrows: int | None = None,
        stratify: bool = False,
        label_col: str = "Attack",
        seed: int = 42,
        return_type: Literal["lazy", "polars", "pandas"] = "lazy",
    ) -> pl.LazyFrame:
        """Load NF-ToN-IoT-v3 dataset as a LazyFrame.


        Parameters
        ----------
        nrows : int, optional
            Number of rows to load. If `stratify=False`,
            loads the first `nrows` rows. If `stratify=True`,
            loads a stratified sample preserving class distribution.

        stratify : bool, default False
            Whether to perform stratified sampling based on
            the target label distribution.

        label_col : str, default "Attack"
            Column used for stratified sampling.

        seed : int, default 42
            Random seed for reproducibility.



        Returns
        -------
        pl.LazyFrame
            Lazy frame. Data not loaded into memory until .collect().

        Examples
        --------
        >>> loader = ToNIoTLoader()
        >>>
        >>> # Load all data (lazy, no memory cost):
        >>> lf_all = loader.load()
        >>>
        >>> # Load first 100k rows (simple head, fast, backward-compatible):
        >>> lf_head = loader.load(nrows=100_000)
        >>>
        >>> # Load stratified 1M sample (preserves class distribution):
        >>> lf_sample = loader.load(nrows=1_000_000, stratify=True)
        >>>
        >>> # Load stratified 500k sample:
        >>> lf_small = loader.load(nrows=500_000, stratify=True, seed=42)
        """

        self.validate()
        lf = pl.scan_parquet(self.data_path)

        if not stratify:
            if nrows is not None:
                lf = lf.head(nrows)
            return lf

        if nrows is None:
            nrows = 1_000_000

        return self._stratified_sample(lf, nrows, label_col, seed, return_type)

    def _stratified_sample(
        self,
        lf: pl.LazyFrame,
        nrows: int,
        label_col: str,
        seed: int,
        return_type: str,
    ) -> pl.LazyFrame:

        # Check label column exists
        schema = lf.collect_schema()
        if label_col not in schema.names():
            raise ValueError(
                f"Label column '{label_col}' not found in dataset. "
                f"Available columns: {schema.names()}"
            )

        # Early exit: if sample >= full dataset, return all
        total = lf.select(pl.len()).collect().item()
        if nrows >= total:
            return lf

        # Compute fraction to sample from each class
        frac = nrows / total
        rng = np.random.default_rng(seed)

        # Materialize only the row index + label column (2 columns, very lightweight)
        labels_df = (
            lf.with_row_index("__row__").select(["__row__", label_col]).collect()
        )

        # Per-class stratified selection: sample from each class proportionally
        selected_rows = []
        for cls in labels_df[label_col].unique().to_list():
            cls_rows = labels_df.filter(pl.col(label_col) == cls)["__row__"].to_numpy()
            # Number to sample: ceil(frac * count), at least 1 per class
            k = max(1, int(np.ceil(len(cls_rows) * frac)))
            k = min(k, len(cls_rows))  # cap at available
            selected = rng.choice(cls_rows, size=k, replace=False)
            selected_rows.extend(selected.tolist())

        # Filter original lazy frame to only the selected rows
        lf_result = (
            lf.with_row_index("__row__")
            .filter(pl.col("__row__").is_in(selected_rows))
            .drop("__row__")
        )
        return self._convert_result(lf_result, return_type)

    def _convert_result(data, return_type: str):
        """Convert result to the requested type."""
        if return_type == "lazy":
            return data
        elif return_type == "polars":
            return data.collect() if isinstance(data, pl.LazyFrame) else data
        elif return_type == "pandas":
            df = data.collect() if isinstance(data, pl.LazyFrame) else data
            return df.to_pandas()
        else:
            raise ValueError(
                f"return_type must be 'lazy', 'polars', or 'pandas', got {return_type}"
            )

    def save(
        self,
        data,
        output_dir: str | Path = "data/sampled",
        name: str = "ton_iot",
        formats: list[str] | None = None,
    ) -> dict:
        """Save data after Sampling."""
        if formats is None:
            formats = ["parquet", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize input to polars DataFrame
        if isinstance(data, pl.LazyFrame):
            df = data.collect()
        elif hasattr(data, "__module__") and "pandas" in data.__module__:
            # pd.DataFrame detected
            df = pl.from_pandas(data)
        elif isinstance(data, pl.DataFrame):
            df = data
        else:
            raise TypeError(
                f"data must be pl.LazyFrame, pl.DataFrame, or pd.DataFrame, "
                f"got {type(data)}"
            )

        n_rows = df.height
        n_cols = df.width
        results = {}

        # Save in requested formats
        for fmt in formats:
            if fmt == "parquet":
                path = output_dir / f"{name}_{n_rows}.parquet"
                df.write_parquet(path)
                results["parquet_path"] = str(path)
                print(f"✓ Saved {n_rows:,} rows, {n_cols:,} cols → {path}")

            elif fmt == "csv":
                path = output_dir / f"{name}_{n_rows}.csv"
                df.write_csv(path)
                results["csv_path"] = str(path)
                print(f"✓ Saved {n_rows:,} rows, {n_cols:,} cols → {path}")

            elif fmt == "json":
                path = output_dir / f"{name}_{n_rows}.json"
                df.write_json(path)
                results["json_path"] = str(path)
                print(f"✓ Saved {n_rows:,} rows, {n_cols:,} cols → {path}")

            else:
                raise ValueError(
                    f"format {fmt} not supported. Choose: parquet, csv, json"
                )

        results["n_rows"] = n_rows
        results["n_cols"] = n_cols

        return results
