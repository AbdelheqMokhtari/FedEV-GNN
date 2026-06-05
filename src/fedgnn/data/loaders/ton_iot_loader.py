from pathlib import Path

import numpy as np
import polars as pl

from fedgnn.data.loaders.base_loader import BaseLoader


class ToNIoTLoader(BaseLoader):
    """Simple loader for NF-ToN-IoT-v3 dataset."""

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DEFAULT_PATH = (
        PROJECT_ROOT / "data" / "raw" / "NF-ToN-IoT-v3" / "NF-ToN-IoT-v3.parquet"
    )
    SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

    def __init__(self, data_path: str | Path | None = None, sample: bool = False):
        """Create a loader.

        Parameters
        ----------
        data_path : str | Path, optional
            Explicit path to load. Overrides ``sample`` when given.
        sample : bool, default False
            If True, load the prepared parquet from ``data/samples/`` instead of
            the full raw dataset (no 5.3GB CSV conversion is triggered).
        """
        if data_path is None:
            data_path = self._latest_sample() if sample else self.DEFAULT_PATH
        super().__init__(data_path)
        self.sample = sample
        # Only the raw dataset needs the one-off CSV -> Parquet conversion.
        if not sample and not self.data_path.exists():
            self._convert()

    def _latest_sample(self) -> Path:
        """Return the most recent sample parquet (pattern: ton_iot_{n}.parquet)."""
        matches = sorted(self.SAMPLES_DIR.glob("ton_iot_*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No sample parquet found in {self.SAMPLES_DIR}")
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _convert(self) -> None:
        """Convert CSV to Parquet (runs once)."""
        csv_path = self.data_path.with_suffix(".csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        print("[ToNIoTLoader] Converting CSV → Parquet...")
        pl.scan_csv(csv_path).sink_parquet(self.data_path, compression="snappy")
        print(f"[ToNIoTLoader] Done → {self.data_path}")

    def load(
        self, nrows: int | None = None, stratify: bool = False, seed: int = 42
    ) -> pl.LazyFrame:
        """Load dataset as LazyFrame.

        Parameters
        ----------
        nrows : int, optional
            If stratify=False: load first nrows (simple head).
            If stratify=True: load stratified sample of nrows (default 1_000_000).
        stratify : bool, default False
            If True, preserve class distribution when sampling.
        seed : int, default 42
            Random seed for reproducibility.
        """
        self.validate()
        lf = pl.scan_parquet(self.data_path)

        if not stratify:
            # Simple head (original behavior)
            if nrows is not None:
                lf = lf.head(nrows)
            return lf

        # Stratified sampling
        if nrows is None:
            nrows = 1_000_000

        return self._stratified_sample(lf, nrows, seed)

    def _stratified_sample(
        self, lf: pl.LazyFrame, nrows: int, seed: int
    ) -> pl.LazyFrame:
        """Stratified sample preserving class distribution."""
        total = lf.select(pl.len()).collect().item()
        if nrows >= total:
            return lf

        frac = nrows / total
        rng = np.random.default_rng(seed)

        # Collect only label column + index (lightweight)
        labels_df = lf.with_row_index("__row__").select(["__row__", "Attack"]).collect()

        # Per-class sampling (allocate exact total)
        selected_rows = []
        total_allocated = 0
        classes = labels_df["Attack"].unique().to_list()

        for i, cls in enumerate(classes):
            cls_rows = labels_df.filter(pl.col("Attack") == cls)["__row__"].to_numpy()

            # Fixing the issue of rounding
            if i == len(classes) - 1:
                k = nrows - total_allocated
            else:
                k = max(1, int(round(len(cls_rows) * frac)))

            k = min(k, len(cls_rows))  # don't exceed available
            selected = rng.choice(cls_rows, size=k, replace=False)
            selected_rows.extend(selected.tolist())
            total_allocated += k

        # Filter and return lazy
        return (
            lf.with_row_index("__row__")
            .filter(pl.col("__row__").is_in(selected_rows))
            .drop("__row__")
        )
