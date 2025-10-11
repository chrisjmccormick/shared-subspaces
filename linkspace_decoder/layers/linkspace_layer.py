"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `linkspace_layer.py`

LinkedSpace Decoder Layer combining attention and feedforward with flexible space-to-module mappings.

This unified layer allows attention outputs and FFN inputs to share the same space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.linkspace_config import LinkedSpaceDecoderConfig


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class LinkedSpaceDecoderLayer(nn.Module):
    """
    A unified decoder layer combining attention and feed-forward networks.
    
    This layer merges the attention and FFN components to allow attention outputs
    and FFN inputs to share the same space, enabling more efficient parameter usage.
    
    Architecture:
        1. Pre-attention norm (RMSNorm on hidden_size)
        2. Multi-head latent attention with flexible space mappings for Q, K, V, O
        3. Residual connection after attention
        4. Pre-FFN norm (RMSNorm on hidden_size)
        5. SwiGLU feed-forward with flexible space mappings for in, gate, out
        6. Residual connection after FFN
    
    All normalization layers are RMSNorm and are always present.
    """

    def __init__(self, config: LinkedSpaceDecoderConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout_prob = config.attention_dropout_prob

        # Attention parameters
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.rope_dims = config.rope_dims
        self.nope_dims = config.nope_dims
        self.qk_private_dim = config.qk_private_dim
        self.vo_private_dim = config.vo_private_dim
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Determine if this is a dense layer
        self.is_dense = layer_idx < config.num_dense_layers

        # =========================
        #   Normalization Layers
        # =========================
        # All norms are RMSNorm and always present
        self.attn_input_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_input_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # =========================
        #   Setup Projections
        # =========================
        self._setup_unified_projections(config)

        # Softmax scaling factor
        self.softmax_scale = self.qk_private_dim ** (-0.5)

    def _setup_unified_projections(self, config: LinkedSpaceDecoderConfig):
        """
        Setup unified space-based projections for all modules (Q, K, V, O, in, gate, out).
        
        All modules are treated uniformly using the spaces architecture:
        1. Create a projection and norm for each unique shared space
        2. Build module -> space_id mapping for all modules
        3. Create private projections for each module
        4. Output modules (O, out) have special handling: use transposed space projections
        
        Dense layers use "identity" space; linkspace layers require all modules in spaces.
        """
        
        # =========================
        # Step 1: Create Space Projections
        # =========================
        
        # Store space projections and norms in ModuleDict
        self.space_projections = nn.ModuleDict()
        self.space_norms = nn.ModuleDict()
        
        if self.is_dense:
            # Dense layer: create identity space for all modules
            self.space_projections["identity"] = nn.Identity()
            self.space_norms["identity"] = nn.Identity()
            space_sizes = {"identity": config.hidden_size}
        else:
            # Linkspace layer: create projections for each defined space
            space_sizes = {}
            for space_id, space_config in config.spaces.items():
                space_key = str(space_id)
                space_size = space_config['size']
                space_sizes[space_key] = space_size
                
                # Create projection WITHOUT bias (allows transposed reuse for outputs)
                self.space_projections[space_key] = nn.Linear(
                    config.hidden_size,
                    space_size,
                    bias=False,
                )
                
                # Create RMSNorm (always present for linkspace)
                self.space_norms[space_key] = DeepseekV3RMSNorm(
                    space_size, 
                    eps=config.rms_norm_eps
                )
        
        # =========================
        # Step 2: Build Module -> Space Mapping
        # =========================
        
        self.module_to_space = {}
        all_modules = ["Q", "K", "V", "O", "in", "gate", "out"]
        
        if self.is_dense:
            # All modules use identity space
            for module in all_modules:
                self.module_to_space[module] = "identity"
        else:
            # All modules must be in a defined space
            for module in all_modules:
                space_id = config.get_space_for_module(module)
                if space_id is None:
                    raise ValueError(
                        f"Module '{module}' must be assigned to a space in linkspace layers. "
                        f"Layer {self.layer_idx} is a linkspace layer (>= num_dense_layers)."
                    )
                self.module_to_space[module] = str(space_id)
        
        # =========================
        # Step 3: Attention Private Projections
        # =========================
        
        # Query
        q_space_key = self.module_to_space["Q"]
        q_input_dim = space_sizes[q_space_key]
        self.q_private_proj = nn.Linear(
            q_input_dim,
            self.num_heads * self.qk_private_dim,
            bias=False
        )
        
        # Key
        k_space_key = self.module_to_space["K"]
        k_input_dim = space_sizes[k_space_key]
        self.k_private_proj = nn.Linear(
            k_input_dim,
            self.num_heads * self.qk_private_dim,
            bias=False
        )
        
        # Value
        v_space_key = self.module_to_space["V"]
        v_input_dim = space_sizes[v_space_key]
        self.v_private_proj = nn.Linear(
            v_input_dim,
            self.num_heads * self.vo_private_dim,
            bias=False
        )
        
        # Output (O) - uses transposed space projection
        o_space_key = self.module_to_space["O"]
        o_space_dim = space_sizes[o_space_key]
        
        self.o_private_proj = nn.Linear(
            self.num_heads * self.vo_private_dim,
            o_space_dim,
            bias=False
        )
        
        # O's normalization (applied before transposed projection)
        if self.is_dense:
            self.o_norm = nn.Identity()
        else:
            self.o_norm = DeepseekV3RMSNorm(o_space_dim, eps=config.rms_norm_eps)
        
        # =========================
        # Step 4: FFN Private Projections
        # =========================
        
        # Input (in)
        in_space_key = self.module_to_space["in"]
        in_space_dim = space_sizes[in_space_key]
        self.in_private_proj = nn.Linear(
            in_space_dim,
            self.intermediate_size,
            bias=True
        )
        
        # Gate
        gate_space_key = self.module_to_space["gate"]
        gate_space_dim = space_sizes[gate_space_key]
        self.gate_private_proj = nn.Linear(
            gate_space_dim,
            self.intermediate_size,
            bias=True
        )
        
        # Output (out) - uses transposed space projection
        out_space_key = self.module_to_space["out"]
        out_space_dim = space_sizes[out_space_key]
        
        self.out_private_proj = nn.Linear(
            self.intermediate_size,
            out_space_dim,
            bias=False
        )
        
        # Out's normalization (applied before transposed projection)
        if self.is_dense:
            self.out_norm = nn.Identity()
        else:
            self.out_norm = DeepseekV3RMSNorm(out_space_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the unified decoder layer.
        
        Args:
            hidden_states: Input tensor [B, T, D]
            position_embeddings: Tuple of (cos, sin) RoPE embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [B, T, D] after attention + FFN with residuals
        """
        
        # === Tensor Dimension Symbols ===
        #    B: batch_size     — number of samples in the batch
        #    T: seq_len        — number of tokens per sample
        #    H: n_heads        — number of attention heads
        #    D: hidden_dim     — model embedding size
        #   Dq: qk_private_dim - per-head query/key dimension
        #   Dv: vo_private_dim - per-head value/output dimension
        #   Dr: rope_dims      - dimensions receiving RoPE
        #  Dff: intermediate_size - FFN hidden dimension

        # ========================
        #     Self Attention
        # ========================
        residual = hidden_states

        # Normalize the hidden states to create the input to attention
        attn_input = self.attn_input_norm(hidden_states)

        # Run attention
        attn_output = self._forward_attention(attn_input, position_embeddings, attention_mask)

        # Add residual connection
        hidden_states = residual + attn_output

        # ===========================
        #     Feed-Forward Network
        # ===========================
        residual = hidden_states

        # Normalize the updated hidden states prior to the FFN
        ffn_input = self.ffn_input_norm(hidden_states)

        # Run FFN
        ffn_output = self._forward_feedforward(ffn_input)

        # Add residual connection
        hidden_states = residual + ffn_output

        return hidden_states

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute multi-head latent attention with flexible space mappings.
        
        Args:
            hidden_states: Normalized input [B, T, D]
            position_embeddings: Tuple of (cos, sin) RoPE embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Attention output [B, T, D]
        """
        B, T = hidden_states.shape[:2]
        H = self.num_heads
        Dq = self.qk_private_dim
        Dv = self.vo_private_dim

        # ==============================
        #   Unified Space-Based Forward
        # ==============================
        
        # Cache for evaluated spaces to avoid redundant computation
        space_cache = {}
        
        def get_space_representation(module: str) -> torch.Tensor:
            """
            Get the space representation for a module (Q, K, V, O, in, gate, out).
            Uses caching to avoid recomputing the same space projection.
            """
            space_key = self.module_to_space[module]
            
            # Check if already computed
            if space_key not in space_cache:
                # Apply space projection and normalization
                space_proj = self.space_projections[space_key]
                space_norm = self.space_norms[space_key]
                
                space_repr = space_proj(hidden_states)
                space_repr = space_norm(space_repr)
                
                # Cache for reuse
                space_cache[space_key] = space_repr
            
            return space_cache[space_key]
        
        # === Query ===
        q_space = get_space_representation("Q")
        queries = self.q_private_proj(q_space)
        
        # === Key ===
        k_space = get_space_representation("K")
        keys = self.k_private_proj(k_space)
        
        # === Value ===
        v_space = get_space_representation("V")
        values = self.v_private_proj(v_space)

        # Split up queries, keys, values for multi-head attention
        # Inputs:  Each [B, T, H*Dh]
        # Outputs: Each [B, H, T, Dh]
        queries = queries.view(B, T, H, Dq).transpose(1, 2)
        keys = keys.view(B, T, H, Dq).transpose(1, 2)
        values = values.view(B, T, H, Dv).transpose(1, 2)

        # ==================
        #        RoPE
        # ==================

        # 1. Unpack the precomputed cosine and sine embeddings
        cos, sin = position_embeddings

        # 2. Split the query and key heads into the part to rotate and the part to pass through
        q_rope, q_pass = queries[..., :self.rope_dims], queries[..., self.rope_dims:]
        k_rope, k_pass = keys[..., :self.rope_dims], keys[..., self.rope_dims:]

        # 3. Apply the rotary embedding to the designated slice
        # To broadcast cos and sin across the batch and head dimensions, we unsqueeze them.
        # Shape change: [T, Dr] -> [1, 1, T, Dr]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rotated = (q_rope * cos) + (rotate_half(q_rope) * sin)
        k_rotated = (k_rope * cos) + (rotate_half(k_rope) * sin)

        # 4. Concatenate the rotated and pass-through parts back together
        queries = torch.cat((q_rotated, q_pass), dim=-1)
        keys = torch.cat((k_rotated, k_pass), dim=-1)

        # ===================
        #       Attention
        # ===================

        # Only apply dropout during training
        dropout_p = self.attention_dropout_prob if self.training else 0.0

        # Call SDPA / Flash Attention
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=dropout_p,
            scale=self.softmax_scale,
            is_causal=True,
        )

        # Reshape output back to [B, T, H * Dv] from [B, H, T, Dv]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * Dv)

        # =========================
        #  Unified Output Projection
        # =========================
        
        # Step 1: Project to output space (private projection)
        attn_output = self.o_private_proj(attn_output)
        
        # Step 2: Apply O's normalization (RMSNorm)
        attn_output = self.o_norm(attn_output)
        
        # Step 3: Project back to hidden size using TRANSPOSED space projection
        o_space_key = self.module_to_space["O"]
        o_space_proj = self.space_projections[o_space_key]
        
        if isinstance(o_space_proj, nn.Identity):
            # Dense layer - identity projection
            attn_output = o_space_proj(attn_output)
        else:
            # Linkspace layer - use transposed weight to go from space back to hidden
            attn_output = F.linear(attn_output, o_space_proj.weight.t(), bias=None)

        return attn_output

    def _forward_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SwiGLU feed-forward network with unified space mappings.
        
        FFN(x) = W_out( Swish(W_in(x)) ⊙ W_gate(x) )
        
        Uses the same unified space architecture as attention:
        - Input modules (in, gate) go through: space_proj -> space_norm -> private_proj
        - Output module (out) goes through: private_proj -> out_norm -> transposed space_proj
        
        Args:
            x: Normalized input [B, T, D]
            
        Returns:
            FFN output [B, T, D]
        """
        
        # ==============================
        #   Unified Space-Based Forward
        # ==============================
        
        # Cache for evaluated spaces to avoid redundant computation
        # (can reuse if in and gate share a space)
        space_cache = {}
        
        def get_space_representation(module: str) -> torch.Tensor:
            """
            Get the space representation for a module.
            Uses caching to avoid recomputing the same space projection.
            """
            space_key = self.module_to_space[module]
            
            # Check if already computed
            if space_key not in space_cache:
                # Apply space projection and normalization
                space_proj = self.space_projections[space_key]
                space_norm = self.space_norms[space_key]
                
                space_repr = space_proj(x)
                space_repr = space_norm(space_repr)
                
                # Cache for reuse
                space_cache[space_key] = space_repr
            
            return space_cache[space_key]
        
        # === Input (in) ===
        in_space = get_space_representation("in")
        x_proj = self.in_private_proj(in_space)
        
        # === Gate ===
        gate_space = get_space_representation("gate")
        gate = self.gate_private_proj(gate_space)
        
        # SwiGLU nonlinearity
        ffn_hidden = F.silu(x_proj) * gate  # [B, T, intermediate_size]
        
        # =========================
        #  Output Projection (out)
        # =========================
        
        # Step 1: Project to output space (private projection)
        ffn_output = self.out_private_proj(ffn_hidden)
        
        # Step 2: Apply out's normalization (RMSNorm)
        ffn_output = self.out_norm(ffn_output)
        
        # Step 3: Project back to hidden size using TRANSPOSED space projection
        out_space_key = self.module_to_space["out"]
        out_space_proj = self.space_projections[out_space_key]
        
        if isinstance(out_space_proj, nn.Identity):
            # Dense layer - identity projection
            ffn_output = out_space_proj(ffn_output)
        else:
            # Linkspace layer - use transposed weight to go from space back to hidden
            ffn_output = F.linear(ffn_output, out_space_proj.weight.t(), bias=None)
        
        return ffn_output

