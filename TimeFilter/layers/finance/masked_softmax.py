"""Strict adjacency softmax for the optional financial graph variant."""

import torch


def strict_masked_softmax(logits, valid):
    """Normalize only routed edges; callers must retain at least a self-loop."""
    return torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)
