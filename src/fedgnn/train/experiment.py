"""Experiment isolation + metadata for federated runs.

Every run lives in its own ``<root>/<experiment>/`` subtree so a new run never
clobbers a previous one::

    Models/experiment_001/server/…   Models/experiment_001/client_NN/…
    Results/experiment_001/server/…  Results/experiment_001/client_NN/…
    Results/experiment_001/metadata.json   ← run-level provenance

An experiment is created fresh (auto-numbered, or a name via ``--experiment``) or
re-opened for resumption. ``metadata.json`` captures when it started, its config,
whether/when it was resumed from a checkpoint, progress, and the best score.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_PREFIX = "experiment_"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Experiment:
    """One isolated run: its model/result directories and ``metadata.json``."""

    def __init__(self, models_root: str | Path, results_root: str | Path, name: str):
        self.name = name
        self.models_dir = Path(models_root) / name
        self.results_dir = Path(results_root) / name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.results_dir / "metadata.json"

    # resolution

    @staticmethod
    def _auto_name(results_root: Path) -> str:
        """Next ``experiment_NNN`` not already present under ``results_root``."""
        existing = []
        if results_root.exists():
            for p in results_root.glob(f"{EXPERIMENT_PREFIX}*"):
                suffix = p.name[len(EXPERIMENT_PREFIX) :]
                if p.is_dir() and suffix.isdigit():
                    existing.append(int(suffix))
        return f"{EXPERIMENT_PREFIX}{(max(existing) + 1) if existing else 1:03d}"

    @staticmethod
    def _latest_name(results_root: Path) -> str | None:
        """Most recently modified *experiment* dir under ``results_root``, if any.

        Only directories carrying a ``metadata.json`` count as experiments, so
        stray folders (e.g. an older flat ``server``/``client_NN`` layout) are
        never mistaken for a resumable run.
        """
        if not results_root.exists():
            return None
        candidates = [
            p
            for p in results_root.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p / "metadata.json").stat().st_mtime).name

    @classmethod
    def resolve(
        cls,
        models_root: str | Path,
        results_root: str | Path,
        name: str | None = None,
        resume: bool = False,
    ) -> "Experiment":
        """Pick the experiment to use.

        * explicit ``name`` -> use it (created if missing).
        * ``resume`` with no name -> the latest existing experiment (or a fresh
          auto-numbered one if none exist yet).
        * otherwise -> a fresh auto-numbered experiment.
        """
        results_root = Path(results_root)
        if name:
            return cls(models_root, results_root, name)
        if resume:
            latest = cls._latest_name(results_root)
            if latest is not None:
                return cls(models_root, results_root, latest)
        return cls(models_root, results_root, cls._auto_name(results_root))

    def exists_on_disk(self) -> bool:
        """True if this experiment already has metadata (i.e. has been run before)."""
        return self.metadata_path.exists()

    # metadata

    def load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text())

    def save_metadata(self, metadata: dict) -> None:
        self.metadata_path.write_text(json.dumps(metadata, indent=2))

    def init_metadata(self, config: dict, resuming: bool) -> dict:
        """Create or update metadata at the start of a run; returns the dict.

        On a fresh start, stamps ``created_at``/``config``. On a resume, preserves
        the original creation info and appends a resume event.
        """
        meta = self.load_metadata()
        if not meta:
            meta = {
                "experiment": self.name,
                "created_at": _now(),
                "status": "running",
                "config": config,
                "resumed": False,
                "resume_events": [],
                "rounds_completed": 0,
                "best_metric": None,
                "best_round": None,
            }
        if resuming and meta:
            meta["resumed"] = True
            meta.setdefault("resume_events", []).append(
                {"at": _now(), "from_round": meta.get("rounds_completed", 0)}
            )
            # Record the config actually used for this resumed segment.
            meta["config"] = config
        meta["status"] = "running"
        meta["updated_at"] = _now()
        self.save_metadata(meta)
        return meta

    def update_metadata(self, **fields) -> dict:
        """Patch fields (e.g. progress, status) and re-stamp ``updated_at``."""
        meta = self.load_metadata()
        meta.update(fields)
        meta["updated_at"] = _now()
        self.save_metadata(meta)
        return meta
