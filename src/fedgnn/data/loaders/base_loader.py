from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def exists(self) -> bool:
        """Check whether dataset exists."""
        return self.data_path.exists()

    def validate(self) -> None:
        """Validate dataset presence."""
        if not self.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load dataset into pandas DataFrame."""
        pass
