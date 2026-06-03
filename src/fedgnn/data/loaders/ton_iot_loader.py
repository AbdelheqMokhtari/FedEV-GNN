from pathlib import Path

import pandas as pd

from fedgnn.data.loaders.base_loader import BaseLoader


class ToNIoTLoader(BaseLoader):
    """Loader for NF-ToN-IoT-v3 dataset."""

    PROJECT_ROOT = Path(__file__).resolve().parents[4]

    DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "NF-ToN-IoT-v3" / "NF-ToN-IoT-v3.csv"

    def __init__(self, data_path: str | Path | None = None):
        super().__init__(data_path or self.DEFAULT_PATH)

    def load(
        self,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """Load NF-ToN-IoT-v3 dataset."""

        self.validate()

        df = pd.read_csv(
            self.data_path,
            nrows=nrows,
        )

        return df
