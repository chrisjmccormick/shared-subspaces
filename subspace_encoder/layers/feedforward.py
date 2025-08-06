"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `feedforward.py`

Regarding dropout:

- I don't see it applied to the MoE in DeepSeek-V3, [here](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py).

- I don't see it applied in [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L140)

Norms:

* nn.RMSNorm [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)

## FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shared_space_config import SharedSpaceEncoderConfig

# TODO - Find a shared place to put this.
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SubspaceFeedForward(nn.Module):
    """
    Feed-forward block for SharedSpaceEncoder.

    Implements SwiGLU:
        FFN(x) = W_out( Swish(W_in(x)) ⊙ W_gate(x) ) + residual

    Supports both dense and decomposed MLP variants.

    Dense:
        - W_in:   Linear(hidden_dim → intermediate_dim)
        - W_gate: Linear(hidden_dim → intermediate_dim)
        - W_out:  Linear(intermediate_dim → hidden_dim)

    Decomposed:
        - W_in_shared:   Linear(hidden_dim → rank, bias=False)
        - W_in_shared_norm: RMSNorm
        - W_in:          Linear(rank → intermediate_dim)
        - W_gate_shared: Linear(hidden_dim → rank, bias=False)
        - W_gate_shared_norm: RMSNorm
        - W_gate:        Linear(rank → intermediate_dim)
        - W_out:         Linear(intermediate_dim → rank, bias=False)
        - W_out_shared:  Linear(rank → hidden_dim)

    Residual, dropout, and post-norm are handled inside the block.
    """

    def __init__(self, config, layer_idx):
        super().__init__()


        #dropout_prob = config.hidden_dropout_prob # TODO - Style -- don't define variables if only used once.

        # Determine whether this is a dense or decomposed layer.
        # It's dense if either:
        #  - ffn_decompose is disabled (no dense layers at all)
        #  - ffn_decompose is enabled, but this is one of the early dense layers.
        self.is_dense = (not config.ffn_decompose) or (layer_idx < config.num_dense_layers)

        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size # TODO - Find something shorter, and use the same name.

        # If it's one of the dense layers,
        if self.is_dense:
            # === Dense FFN Projections ===
            self.W_in = nn.Linear(hidden_dim, intermediate_dim)
            self.W_gate = nn.Linear(hidden_dim, intermediate_dim)
            self.W_out = nn.Linear(intermediate_dim, hidden_dim)

        # Define weights for the decomposed version.
        else:
            rank = config.ffn_rank

            print("hidden_dim:", hidden_dim)
            print("rank:", rank)

            # === Input Projections ===
            self.W_in_shared = nn.Linear(hidden_dim, rank, bias=False)
            self.W_in_shared_norm = DeepseekV3RMSNorm(rank, eps=config.rms_norm_eps)
            self.W_in = nn.Linear(rank, intermediate_dim, bias=True)

            # === Gate Projections ===
            self.W_gate_shared = nn.Linear(hidden_dim, rank, bias=False)
            self.W_gate_shared_norm = DeepseekV3RMSNorm(rank, eps=config.rms_norm_eps)
            self.W_gate = nn.Linear(rank, intermediate_dim, bias=True)

            # === Output Projection ===
            self.W_out = nn.Linear(intermediate_dim, rank, bias=False)
            # TODO - Could experiment with this.
            #self.W_out_shared_layernorm = DeepseekV3RMSNorm(rank, eps=config.eps)
            self.W_out_shared = nn.Linear(rank, hidden_dim, bias=True)

        # See notes no dropout
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Tensor Dimension Symbols ===
        # B: batch_size     — number of samples in the batch
        # T: seq_len        — number of tokens per sample
        # D: hidden_dim     — model embedding size
        # R: ffn_rank       — latent shared subspace dimension
        # D_ff: intermediate_size — FFN hidden dimension

        # =========================
        #    Gated Feedforward
        # =========================

        if self.is_dense:
            # =============
            #     Dense
            # =============

            # Input:  x [B, T, D]
            # Output: x_proj [B, T, D_ff]
            x_proj = self.W_in(x)

            # Output: gate [B, T, D_ff]
            gate = self.W_gate(x)

            # SwiGLU nonlinearity
            x = F.silu(x_proj) * gate  # [B, T, D_ff]

            # See notes on dropout
            #x = self.dropout(x)

            # Output: x [B, T, D]
            x = self.W_out(x)

        else:
            # ==================
            #     Decomposed
            # ==================

            # Input:  x [B, T, D]
            # Output: x_proj [B, T, D_ff]
            x_proj = self.W_in(self.W_in_shared_norm(self.W_in_shared(x)))

            # Input:  x [B, T, D]
            # Output: gate [B, T, D_ff]
            gate = self.W_gate(self.W_gate_shared_norm(self.W_gate_shared(x)))

            # SwiGLU nonlinearity
            x = F.silu(x_proj) * gate  # [B, T, D_ff]

            # See notes on dropout
            #x = self.dropout(x)

            # Output: x [B, T, D]
            x = self.W_out_shared(self.W_out(x))


        return x




