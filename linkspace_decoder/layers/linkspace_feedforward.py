"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `linkspace_feedforward.py`

LinkedSpace Feed-Forward Network with flexible space-to-module mappings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linkspace_config import LinkedSpaceDecoderConfig


def create_norm_layer(hidden_size: int, config: LinkedSpaceDecoderConfig) -> nn.Module:
    """
    Create a normalization layer based on the config norm_type.
    
    Args:
        hidden_size: The dimension to normalize over
        config: Configuration containing norm_type and epsilon values
    
    Returns:
        Either a LayerNorm or RMSNorm layer
    """
    if config.norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
    elif config.norm_type == "rmsnorm":
        return DeepseekV3RMSNorm(hidden_size, eps=config.rms_norm_eps)
    else:
        # This should be caught by config validation, but being defensive
        raise ValueError(f"Unknown norm_type: {config.norm_type}")


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

class LinkedSpaceFeedForward(nn.Module):
    """
    Feed-forward block with flexible LinkedSpace configuration.

    Implements SwiGLU:
        FFN(x) = W_out( Swish(W_in(x)) ⊙ W_gate(x) )

    Supports flexible mappings of "in", "gate", and "out" modules to shared spaces.

    Dense (no linkspaces):
        - W_in:   Linear(hidden_dim → intermediate_dim)
        - W_gate: Linear(hidden_dim → intermediate_dim)
        - W_out:  Linear(intermediate_dim → hidden_dim)

    With LinkSpaces:
        Each module can optionally project through a shared space:
        - If "in" is in a space:
          W_in_shared → norm → W_in_private
        - If "gate" is in a space:
          W_gate_shared → norm → W_gate_private
        - If "out" is in a space:
          W_out_private → norm → W_out_shared
    """

    def __init__(self, config: LinkedSpaceDecoderConfig, layer_idx: int):
        super().__init__()

        # Determine whether this is a dense or linkspace layer.
        self.is_dense = layer_idx < config.num_dense_layers

        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size

        # If it's one of the dense layers,
        if self.is_dense:
            # === Dense FFN Projections ===
            self.W_in = nn.Linear(hidden_dim, intermediate_dim)
            self.W_gate = nn.Linear(hidden_dim, intermediate_dim)
            self.W_out = nn.Linear(intermediate_dim, hidden_dim)

        # Define weights for the linkspace version.
        else:
            # === LinkSpace FFN Projections ===
            self._setup_linkspace_projections(config, hidden_dim, intermediate_dim)

    def _setup_linkspace_projections(
        self, 
        config: LinkedSpaceDecoderConfig, 
        hidden_dim: int, 
        intermediate_dim: int
    ):
        """
        Setup projections for "in", "gate", "out" based on linkspace configuration.
        """

        # === Input Projection ("in") ===
        in_space_config = config.get_module_space_config("in")
        if in_space_config is not None:
            # "in" is in a linkspace
            self.in_uses_space = True
            in_space_dim = in_space_config['size']
            
            self.W_in_shared = nn.Linear(hidden_dim, in_space_dim, bias=False)
            
            if in_space_config['norm']:
                self.W_in_shared_norm = create_norm_layer(in_space_dim, config)
            else:
                self.W_in_shared_norm = nn.Identity()
            
            self.W_in = nn.Linear(in_space_dim, intermediate_dim, bias=True)
        else:
            # "in" uses direct projection
            self.in_uses_space = False
            self.W_in = nn.Linear(hidden_dim, intermediate_dim, bias=True)

        # === Gate Projection ("gate") ===
        gate_space_config = config.get_module_space_config("gate")
        if gate_space_config is not None:
            # "gate" is in a linkspace
            self.gate_uses_space = True
            gate_space_dim = gate_space_config['size']
            
            self.W_gate_shared = nn.Linear(hidden_dim, gate_space_dim, bias=False)
            
            if gate_space_config['norm']:
                self.W_gate_shared_norm = create_norm_layer(gate_space_dim, config)
            else:
                self.W_gate_shared_norm = nn.Identity()
            
            self.W_gate = nn.Linear(gate_space_dim, intermediate_dim, bias=True)
        else:
            # "gate" uses direct projection
            self.gate_uses_space = False
            self.W_gate = nn.Linear(hidden_dim, intermediate_dim, bias=True)

        # === Output Projection ("out") ===
        out_space_config = config.get_module_space_config("out")
        if out_space_config is not None:
            # "out" is in a linkspace
            self.out_uses_space = True
            out_space_dim = out_space_config['size']
            
            self.W_out = nn.Linear(intermediate_dim, out_space_dim, bias=False)
            
            if out_space_config['norm']:
                self.W_out_norm = create_norm_layer(out_space_dim, config)
            else:
                self.W_out_norm = nn.Identity()
            
            self.W_out_shared = nn.Linear(out_space_dim, hidden_dim, bias=True)
        else:
            # "out" uses direct projection
            self.out_uses_space = False
            self.W_out = nn.Linear(intermediate_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Tensor Dimension Symbols ===
        # B: batch_size     — number of samples in the batch
        # T: seq_len        — number of tokens per sample
        # D: hidden_dim     — model embedding size
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

            # Output: x [B, T, D]
            x = self.W_out(x)

        else:
            # ==================
            #     LinkSpace
            # ==================

            # === Input ===
            if self.in_uses_space:
                x_proj = self.W_in(self.W_in_shared_norm(self.W_in_shared(x)))
            else:
                x_proj = self.W_in(x)

            # === Gate ===
            if self.gate_uses_space:
                gate = self.W_gate(self.W_gate_shared_norm(self.W_gate_shared(x)))
            else:
                gate = self.W_gate(x)

            # SwiGLU nonlinearity
            x = F.silu(x_proj) * gate  # [B, T, D_ff]

            # === Output ===
            if self.out_uses_space:
                x = self.W_out_shared(self.W_out_norm(self.W_out(x)))
            else:
                x = self.W_out(x)

        return x

