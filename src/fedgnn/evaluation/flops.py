"""Computational-cost accounting for local training.

Quantifies "how much calculation each client performs" so the heterogeneous
compute load across the federated partitions can be measured and reported — the
per-client graphs differ by orders of magnitude in node/edge count, so their
per-round FLOPs differ accordingly.

FLOPs are measured with PyTorch's ``FlopCounterMode``, which traces the actual
tensor ops executed in a forward **and** backward pass. It captures the
matmul/attention-projection work that dominates GAT cost; sparse scatter
(message-passing) gathers are not all counted, so treat the figure as a
matmul-dominant lower bound that is exact for the dense layers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total number of (trainable) parameters in ``model``."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_training_flops(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> int:
    """FLOPs for a single forward + backward training step on the full graph.

    Because every local epoch runs on the same (full-batch) graph, the per-epoch
    cost is constant: multiply this by the number of local epochs to get the
    per-round cost. Grads are zeroed before and after so the measurement does not
    perturb the subsequent real training step.
    """
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    counter = FlopCounterMode(display=False)
    with counter:
        logits, _ = model(x, edge_index)
        loss = criterion(logits[mask], y[mask])
        loss.backward()
    total = int(counter.get_total_flops())

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return total
