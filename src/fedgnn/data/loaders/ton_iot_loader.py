from pathlib import Path

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

    def load(self, nrows: int | None = None) -> pl.LazyFrame:
        """Load NF-ToN-IoT-v3 dataset as a LazyFrame."""
        self.validate()
        lf = pl.scan_parquet(self.data_path)
        if nrows is not None:
            lf = lf.head(nrows)
        return lf
