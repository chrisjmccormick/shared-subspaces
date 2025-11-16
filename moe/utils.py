"""
# Utility Functions for Sparse MoE Decoder

Contains helper routines shared between configuration objects and layer
implementations. Mirrors the helpers offered in `subspace_decoder/utils.py`.
"""

from __future__ import annotations

from typing import Iterable, Literal, Tuple

import torch
import torch.nn as nn


def create_norm_layer(hidden_size: int, norm_type: Literal["layernorm", "rmsnorm"], eps: float) -> nn.Module:
    """
    Factory for normalization layers mirroring the behavior in the subspace
    decoder.

    Args:
        hidden_size: Dimension to normalize.
        norm_type: Either `"layernorm"` or `"rmsnorm"`.
        eps: Numerical stability epsilon.

    Returns:
        Instantiated normalization module.
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps)
    if norm_type == "rmsnorm":
        return DeepseekV3RMSNorm(hidden_size, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class DeepseekV3RMSNorm(nn.Module):
    """
    Reusable RMSNorm variant also used inside the original subspace decoder.

    We duplicate the implementation here so the MoE stack does not take a hard
    dependency on the original code paths.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def format_size(num: int) -> str:
    """
    Pretty-print large integers using common suffixes.

    Mirrors the helper from the subspace decoder so training scripts can report
    parameter counts in a compact form.
    """
    suffixes = [" ", "K", "M", "B"]
    base = 1024
    value = float(num)
    for suffix in suffixes:
        if abs(value) < base:
            return f"{value:.2f}{suffix}" if value % 1 else f"{int(value)}{suffix}"
        value /= base
    return f"{value:.2f}T" if value % 1 else f"{int(value)}T"


def summarize_parameters(model: nn.Module, display_bias: bool = True) -> int:
    """
    Print a compact table of parameter shapes/counts and return the total.
    """
    params: Iterable[Tuple[str, nn.Parameter]] = list(model.named_parameters())
    print(f"The model has {len(params)} different named parameters.\n")

    total = 0
    for _, tensor in params:
        total += tensor.numel()

    print(f"Total elements: {format_size(total)}\n")
    header = (
        "Parameter Name                                              "
        "Dimensions       Total Values    Trainable"
    )
    print(header)

    for name, tensor in params:
        shape = list(tensor.size())
        compact = [dim for dim in shape if dim != 1]
        if len(compact) == 1:
            if not display_bias:
                continue
            dims = f"{tensor.size()[0]:>10,} x {'-':<10}"
        elif len(compact) == 2:
            dims = f"{tensor.size()[0]:>10,} x {tensor.size()[1]:<10,}"
        elif len(compact) == 3:
            dims = (
                f"{tensor.size()[0]:>10,} x {tensor.size()[1]:,} x "
                f"{tensor.size()[2]:<10}"
            )
        elif len(compact) == 4:
            dims = (
                f"{tensor.size()[0]:>10,} x {tensor.size()[1]:,} x "
                f"{tensor.size()[2]:,} x {tensor.size()[3]:<10}"
            )
        else:
            dims = " x ".join(str(dim) for dim in tensor.size())
        print(
            f"{name:<55} {dims}    {format_size(tensor.numel()):>6}    {tensor.requires_grad}"
        )

    print(f"\nTotal elements: {format_size(total)}\n")
    return total
