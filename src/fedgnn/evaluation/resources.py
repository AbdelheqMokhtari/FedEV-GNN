"""Lightweight runtime resource accounting for a code block.

Complements the FLOPs counter (:mod:`fedgnn.evaluation.flops`, a *theoretical*
cost) with the *measured* cost: wall-clock time, CPU time / utilisation, process
memory, and peak GPU memory. It is cheap enough to wrap **every federated round**
— no sampling threads, just start/stop deltas — so the trainer can record what each
client's local training and inference actually cost into ``Results/``.

Measurement notes:

* **CPU %** is ``process CPU time / wall time × 100`` over the block. It can exceed
  100 % when PyTorch uses several CPU threads, and is low on CUDA (the CPU mostly
  waits on the GPU) — both are honest signals of where the work happens.
* **`rss_mb`** is the process resident memory *snapshot* at the end of the block
  (the whole simulation shares one process, so it is the footprint at that moment,
  not a per-client increment).
* **`gpu_peak_mb`** is the peak CUDA memory *during* the block (the allocator's
  peak stats are reset on entry), so it **is** a clean per-block figure.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

MB = 1024 * 1024
_PAGE = (
    os.sysconf("SC_PAGE_SIZE")
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in getattr(os, "sysconf_names", {})
    else 4096
)


def rss_mb() -> float:
    """Resident set size of this process in MB (Linux /proc; resource fallback)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * _PAGE / MB
    except Exception:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


def _cuda_device(device):
    """Return the torch CUDA device for ``device`` (or ``None`` if not CUDA)."""
    if device is None:
        return None
    try:
        import torch

        dev = device if isinstance(device, torch.device) else torch.device(device)
        return dev if dev.type == "cuda" else None
    except Exception:
        return None


@dataclass
class ResourceUsage:
    """Measured cost of a code block."""

    wall_s: float
    cpu_s: float
    cpu_percent: float
    rss_mb: float
    gpu_peak_mb: float | None = None

    def as_record(self, prefix: str = "") -> dict:
        """Flatten into round JSON keys, e.g. ``train_time_s`` / ``train_cpu_percent``."""  # noqa: E501
        rec = {
            f"{prefix}time_s": round(self.wall_s, 4),
            f"{prefix}cpu_percent": round(self.cpu_percent, 1),
            f"{prefix}rss_mb": round(self.rss_mb, 1),
        }
        if self.gpu_peak_mb is not None:
            rec[f"{prefix}gpu_peak_mb"] = round(self.gpu_peak_mb, 1)
        return rec


class track_resources:
    """Context manager measuring wall/CPU time, end RSS, and peak GPU mem of a block.

    ::

        with track_resources(device) as r:
            ...  # work
        r.usage  # -> ResourceUsage
    """

    def __init__(self, device=None) -> None:
        self.device = device
        self.usage: ResourceUsage | None = None

    def __enter__(self) -> "track_resources":
        self._gpu = _cuda_device(self.device)
        if self._gpu is not None:
            import torch

            torch.cuda.reset_peak_memory_stats(self._gpu)
        self._w0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, *exc) -> None:
        gpu_peak = None
        if self._gpu is not None:
            import torch

            # CUDA kernels are async — wait for them so wall time + peak are real.
            torch.cuda.synchronize(self._gpu)
            gpu_peak = torch.cuda.max_memory_allocated(self._gpu) / MB
        wall = time.perf_counter() - self._w0
        cpu = time.process_time() - self._c0
        self.usage = ResourceUsage(
            wall_s=wall,
            cpu_s=cpu,
            cpu_percent=(cpu / wall * 100.0) if wall > 0 else 0.0,
            rss_mb=rss_mb(),
            gpu_peak_mb=gpu_peak,
        )
