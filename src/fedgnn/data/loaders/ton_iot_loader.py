from pathlib import Path

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

        return self.__setattr__

    def _stratified_sample(
        self, lf: pl.LazyFrame, nrows: int, label_col: str, seed: int
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
        return (
            lf.with_row_index("__row__")
            .filter(pl.col("__row__").is_in(selected_rows))
            .drop("__row__")
        )
