
"""# `feedforward.py`

MoE-aware feed-forward building blocks mirroring the structure of the dense
`SubspaceFeedForward` module from `subspace_decoder`. Each expert implements a
SwiGLU block with optional low-rank decomposition while the router performs a
noisy top-k dispatch with explicit capacity enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..models.shared_space_config import SparseMoEDecoderConfig
from ..utils import create_norm_layer


# --------------------------------------------------------------------------- #
#                                Router Output                                #
# --------------------------------------------------------------------------- #

@dataclass
class RouterOutput:
    """
    Container storing router decisions.

    Attributes:
        probs:      Softmax-normalized routing weights `[B, T, K]`.
        indices:    Expert indices selected for each token `[B, T, K]`.
        logits:     Raw logits before top-k selection `[B, T, E]` (optional).
    """

    probs: torch.Tensor
    indices: torch.Tensor
    logits: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------- #
#                              Expert SwiGLU Block                            #
# --------------------------------------------------------------------------- #


class ExpertSwiGLU(nn.Module):
    """
    SwiGLU feed-forward expert with optional low-rank decomposition.

    Dense experts follow the standard pattern:
        x -> W_in -> silu -> ⊙ gate -> W_out

    Decomposed experts match the Shared Subspace implementation:
        x -> W_in_shared -> norm -> W_in
          -> W_gate_shared -> norm -> W_gate
          -> W_out -> W_out_shared
    """

    def __init__(self, config: 'SparseMoEDecoderConfig', layer_idx: int) -> None:
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size

        self.is_dense = (not config.ffn_decompose) or (layer_idx < config.num_dense_layers)

        if self.is_dense:
            # === Dense Expert Projections ===
            self.W_in = nn.Linear(self.hidden_dim, self.intermediate_dim)
            self.W_gate = nn.Linear(self.hidden_dim, self.intermediate_dim)
            self.W_out = nn.Linear(self.intermediate_dim, self.hidden_dim)
        else:
            # === Decomposed Expert Projections ===
            rank = config.ffn_rank

            # Input branch
            self.W_in_shared = nn.Linear(self.hidden_dim, rank, bias=False)
            norm_eps = config.layer_norm_eps if config.norm_type == "layernorm" else config.rms_norm_eps
            self.W_in_shared_norm = create_norm_layer(rank, config.norm_type, norm_eps)
            self.W_in = nn.Linear(rank, self.intermediate_dim)

            # Gate branch
            self.W_gate_shared = nn.Linear(self.hidden_dim, rank, bias=False)
            self.W_gate_shared_norm = create_norm_layer(rank, config.norm_type, norm_eps)
            self.W_gate = nn.Linear(rank, self.intermediate_dim)

            # Output branch
            self.W_out = nn.Linear(self.intermediate_dim, rank, bias=False)
            self.W_out_shared = nn.Linear(rank, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `[tokens, hidden_dim]` or `[B, T, hidden_dim]`
        Returns:
            Processed tensor with the same shape.
        """
        original_shape = x.shape
        if x.dim() == 3:
            # Collapse batch + sequence dimensions for per-token processing.
            x = x.view(-1, self.hidden_dim)

        if self.is_dense:
            # ==============================
            #       Dense SwiGLU Path
            # ==============================
            x_proj = self.W_in(x)
            gate = self.W_gate(x)
            x = F.silu(x_proj) * gate
            x = self.W_out(x)
        else:
            # ==============================
            #    Decomposed SwiGLU Path
            # ==============================
            x_proj = self.W_in(self.W_in_shared_norm(self.W_in_shared(x)))
            gate = self.W_gate(self.W_gate_shared_norm(self.W_gate_shared(x)))
            x = F.silu(x_proj) * gate
            x = self.W_out_shared(self.W_out(x))

        if len(original_shape) == 3:
            x = x.view(original_shape)
        return x


# --------------------------------------------------------------------------- #
#                                Noisy Top-K Router                           #
# --------------------------------------------------------------------------- #


class NoisyTopKRouter(nn.Module):
    """
    Gating network that selects a sparse subset of experts per token.

    The implementation mirrors the router sketched in `moe/sparse_moe.py` but
    exposes the logits/probabilities explicitly to simplify debugging and loss
    computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        noise_std: float,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Args:
            x: `[B, T, D]` hidden states.

        Returns:
            RouterOutput with probabilities `[B, T, K]` and indices `[B, T, K]`.
        """
        # Linear projection that produces one logit per expert.
        # Shape: `[B, T, E]` where `E` is the number of experts.
        logits = self.router(x)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Keep only the top-k experts per token to achieve sparsity.
        # Shapes:
        #   topk_logits  -> `[B, T, K]`
        #   topk_indices -> `[B, T, K]`
        topk_logits, topk_indices = logits.topk(self.top_k, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1, dtype=logits.dtype)

        return RouterOutput(probs=topk_probs, indices=topk_indices, logits=logits)


# --------------------------------------------------------------------------- #
#                             Sparse MoE Feed Forward                         #
# --------------------------------------------------------------------------- #


class SparseMoEFeedForward(nn.Module):
    """
    Drop-in replacement for the dense feed-forward layer.

    Pipeline:
        1. Router picks top-k experts per token (with optional noise).
        2. Tokens are dispatched to experts up to capacity.
        3. Expert outputs are weighted by router probabilities and gathered.
    """

    def __init__(self, config: 'SparseMoEDecoderConfig', layer_idx: int) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.use_sparse = config.uses_sparse_experts
        self.hidden_dim = config.hidden_size

        self.dense_fallback = ExpertSwiGLU(config, layer_idx)

        if self.use_sparse:
            self.router = NoisyTopKRouter(
                hidden_dim=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.router_top_k,
                noise_std=config.router_noise_std,
            )
            self.experts = nn.ModuleList(
                ExpertSwiGLU(config, layer_idx) for _ in range(config.num_experts)
            )
            self.dropout = nn.Dropout(config.expert_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `[B, T, D]` hidden states.

        Returns:
            `[B, T, D]` tensor matching the dense FFN interface.
        """
        if not self.use_sparse:
            return self.dense_fallback(x)

        B, T, D = x.shape
        tokens = B * T

        router_out = self.router(x)

        # =========================
        #   Flatten Token Batch
        # =========================
        # All routing computations operate on `[tokens, ...]` to simplify
        # scatter/gather operations.
        flat_hidden = x.view(tokens, D)
        flat_probs = router_out.probs.view(tokens, self.config.router_top_k)
        flat_indices = router_out.indices.view(tokens, self.config.router_top_k)

        # === Expert Capacity ===
        # Compute the maximum number of routed tokens each expert is allowed
        # to process. This mirrors the behavior of the sketch in
        # `moe/sparse_moe.py`.
        capacity = self.config.compute_capacity(tokens)

        updates = torch.zeros_like(flat_hidden)

        for expert_idx, expert in enumerate(self.experts):
            mask = flat_indices == expert_idx
            matches = mask.nonzero(as_tuple=False)

            if matches.numel() == 0:
                continue

            # Limit to capacity to avoid overflowing the expert.
            if matches.shape[0] > capacity:
                matches = matches[:capacity]

            token_indices = matches[:, 0]
            gate_indices = matches[:, 1]

            # Gather token representations assigned to the current expert.
            expert_in = flat_hidden.index_select(0, token_indices)
            expert_out = expert(expert_in)

            gating_scores = flat_probs[token_indices, gate_indices].unsqueeze(-1)
            weighted = expert_out * gating_scores

            updates.index_add_(0, token_indices, weighted)

        if self.training:
            updates = self.dropout(updates)

        return updates.view(B, T, D)