"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `linkspace_mla.py`

LinkedSpace Multi-head Latent Attention with flexible space-to-module mappings.

Based on: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py

## RotaryEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.linkspace_config import LinkedSpaceDecoderConfig


def create_norm_layer(hidden_size: int, config: LinkedSpaceDecoderConfig) -> nn.Module:
    """
    Create a normalization layer based on the config norm_type.

    If `hidden_size` is `None`, this returns an identity layer.

    Args:
        hidden_size: The dimension to normalize over
        config: Configuration containing norm_type and epsilon values

    Returns:
        Either a LayerNorm or RMSNorm layer
    """
    if hidden_size is None:
        return nn.Identity()
    elif config.norm_type == "layernorm":
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """Precompute RoPE embeddings and store them as buffers."""

    def __init__(self, config: LinkedSpaceDecoderConfig) -> None:
        super().__init__()

        dim = config.rope_dims
        seq_len = config.max_position_embeddings

        # ------------------------------
        # Compute inverse frequencies
        # ------------------------------
        # Shape: [dim // 2]
        #   inv_freq[i] = 1 / (theta^(i / dim))
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        # ------------------------------
        # Apply RoPE scaling if configured
        # ------------------------------
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling.get("type", "linear")
            scaling_factor = config.rope_scaling.get("factor", 1.0)

            if scaling_type == "linear":
                # Linear scaling: divide frequencies by scaling factor
                inv_freq = inv_freq / scaling_factor
            elif scaling_type == "dynamic":
                # Dynamic scaling: adjust based on sequence length
                # This is a simplified implementation
                inv_freq = inv_freq / scaling_factor
            else:
                print(f"Warning: Unknown RoPE scaling type '{scaling_type}', using linear scaling")
                inv_freq = inv_freq / scaling_factor

        # ------------------------------
        # Compute position indices
        # ------------------------------
        # Shape: [seq_len]
        t = torch.arange(seq_len, dtype=torch.float32)

        # ------------------------------
        # Outer product: [seq_len, dim // 2]
        # Each row i contains: t[i] * inv_freq
        # ------------------------------
        freqs = torch.outer(t, inv_freq)

        # ------------------------------
        # Duplicate for interleaved sin/cos: [seq_len, dim]
        # This matches the common format: [sin_0, cos_0, sin_1, cos_1, ...]
        # ------------------------------
        emb = torch.cat((freqs, freqs), dim=-1)

        # ------------------------------
        # Register cos/sin as buffers
        # - Stored in float32
        # - Will be moved to correct device/dtype via model.to(...)
        # - Not saved with state_dict (persistent=False)
        # ------------------------------
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        return None # This function is not necessary.

"""## LinkedSpaceMLA"""

class LinkedSpaceMLA(nn.Module):
    """
    Multi-head Latent Attention with flexible LinkedSpace configuration.
    
    Instead of fixed Q/KV/O subspaces, this layer supports arbitrary mappings
    of Q, K, V, O modules to configurable shared spaces.
    
    This unified architecture handles both dense and linkspace layers:
    - Dense layers: components map to "identity" space (no compression)
    - Linkspace layers: components map to configured shared spaces
    """

    def __init__(self, config: LinkedSpaceDecoderConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout_prob = config.attention_dropout_prob

        self.num_heads = config.num_attention_heads

        self.rope_theta = config.rope_theta
        self.rope_dims = config.rope_dims
        self.nope_dims = config.nope_dims

        self.qk_private_dim = config.qk_private_dim
        self.vo_private_dim = config.vo_private_dim

        self.hidden_size = config.hidden_size

        # =========================
        #   Unified Space Setup
        # =========================
        
        # Determine if this is a dense layer
        is_dense = self.layer_idx < config.num_dense_layers
        
        # Setup all projections using unified space-based architecture
        self._setup_unified_projections(config, is_dense)

        # Softmax scaling factor.
        self.softmax_scale = self.qk_private_dim ** (-0.5)

    def _setup_unified_projections(self, config: LinkedSpaceDecoderConfig, is_dense: bool):
        """
        Setup unified space-based projections for Q, K, V, O.
        
        Strategy:
        1. Create a projection and norm for each unique shared space
        2. Build reverse mapping: component -> space_id
        3. Create private projections for each component
        4. Special case O: has its own norm, uses space projection transposed
        5. Dense layers use "identity" space; linkspace requires all modules in spaces
        
        Args:
            config: Model configuration
            is_dense: If True, use identity projections (dense layer)
        """
        
        # =========================
        # Step 1: Create Space Projections
        # =========================
        
        # Store space projections and norms in ModuleDicts
        self.space_projections = nn.ModuleDict()
        self.space_norms = nn.ModuleDict()
        
        if is_dense:
            # Dense layer: create identity space
            # Use "identity" as the space key
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
                
                # Create projection WITHOUT bias (allows transposed reuse for output)
                self.space_projections[space_key] = nn.Linear(
                    config.hidden_size,
                    space_size,
                    bias=False,  # No bias to allow transposed sharing
                )
                
                # Create normalization for input components (Q, K, V)
                # O will have its own separate norm
                if space_config['norm']:
                    self.space_norms[space_key] = create_norm_layer(space_size, config)
                else:
                    self.space_norms[space_key] = nn.Identity()
        
        # =========================
        # Step 2: Build Component -> Space Mapping
        # =========================
        
        self.component_to_space = {}
        
        if is_dense:
            # All components use identity space
            for component in ["Q", "K", "V", "O"]:
                self.component_to_space[component] = "identity"
        else:
            # All components must be in a defined space
            for component in ["Q", "K", "V", "O"]:
                space_id = config.get_space_for_module(component)
                if space_id is None:
                    raise ValueError(
                        f"Component '{component}' must be assigned to a space in linkspace layers. "
                        f"Layer {self.layer_idx} is a linkspace layer (>= num_dense_layers)."
                    )
                self.component_to_space[component] = str(space_id)
        
        # =========================
        # Step 3: Create Private Projections
        # =========================
        
        # Query private projection
        q_space_key = self.component_to_space["Q"]
        q_input_dim = space_sizes[q_space_key]
        self.q_private_proj = nn.Linear(
            q_input_dim,
            self.num_heads * self.qk_private_dim,
            bias=False
        )
        
        # Key private projection
        k_space_key = self.component_to_space["K"]
        k_input_dim = space_sizes[k_space_key]
        self.k_private_proj = nn.Linear(
            k_input_dim,
            self.num_heads * self.qk_private_dim,
            bias=False
        )
        
        # Value private projection
        v_space_key = self.component_to_space["V"]
        v_input_dim = space_sizes[v_space_key]
        self.v_private_proj = nn.Linear(
            v_input_dim,
            self.num_heads * self.vo_private_dim,
            bias=False
        )
        
        # =========================
        # Step 4: Output Special Case
        # =========================
        
        # Output has special handling:
        # 1. Private proj from attention output to O's space
        # 2. Separate O-specific norm (not shared with input components)
        # 3. Uses space projection TRANSPOSED to go back to hidden size
        
        o_space_key = self.component_to_space["O"]
        o_space_dim = space_sizes[o_space_key]
        
        # Private: from attention output to output space
        self.o_private_proj = nn.Linear(
            self.num_heads * self.vo_private_dim,
            o_space_dim,
            bias=False
        )
        
        # O's own normalization (separate from shared space norm)
        if is_dense:
            # Dense layers don't need special norm handling
            self.o_norm = nn.Identity()
        else:
            # Get O's space config to determine if norm is needed
            o_space_id = int(o_space_key) if o_space_key != "identity" else None
            if o_space_id is not None:
                o_space_config = config.spaces[o_space_id]
                if o_space_config['norm']:
                    self.o_norm = create_norm_layer(o_space_dim, config)
                else:
                    self.o_norm = nn.Identity()
            else:
                self.o_norm = nn.Identity()
        
        # Note: We use space_projections[o_space_key] transposed in forward pass
        # No separate o_shared_proj needed!


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        # === Tensor Dimension Symbols ===
        #    B: batch_size     — number of samples in the batch
        #    T: seq_len        — number of tokens per sample
        #    H: n_heads        — number of attention heads
        #    D: hidden_dim     — model embedding size
        #   Dv: vo_private_dim - per-head value/output projection dimension
        #   Dr: rope_dims      - The first Dr dimensions receive rope.

        # Input token embeddings
        # hidden_states: [B, T, D]
        B, T = hidden_states.shape[:2]
        H = self.num_heads
        Dq = self.qk_private_dim     # per-head dim for Q and K
        Dv = self.vo_private_dim     # per-head dim for V/O

        # ==============================
        #   Unified Space-Based Forward
        # ==============================
        
        # Cache for evaluated spaces to avoid redundant computation
        space_cache = {}
        
        def get_space_representation(component: str) -> torch.Tensor:
            """
            Get the space representation for a component (Q, K, V, O).
            Uses caching to avoid recomputing the same space projection.
            """
            space_key = self.component_to_space[component]
            
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

        # Split up queries so that there's just one per row.
        # Same for keys and values.
        #
        # Inputs:
        #   Each  [B, T, H*Dh]
        # Output:
        #   Each  [B, H,  T,  Dh]
        queries = queries.view(B, T, H, Dq).transpose(1, 2)
        keys =       keys.view(B, T, H, Dq).transpose(1, 2)
        values =   values.view(B, T, H, Dv).transpose(1, 2)

        # ==================
        #        RoPE
        # ==================

        # 1. Unpack the precomputed cosine and sine embeddings
        cos, sin = position_embeddings

        # 2. Split the query and key heads into the part to rotate and the part
        #    to pass through
        q_rope, q_pass = queries[..., :self.rope_dims], queries[..., self.rope_dims:]
        k_rope, k_pass =    keys[..., :self.rope_dims],    keys[..., self.rope_dims:]

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

        # Only apply dropout during training.
        if self.training:
            dropout_p = self.attention_dropout_prob
        else:
            dropout_p = 0.0

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
        
        # Step 2: Apply O's normalization
        attn_output = self.o_norm(attn_output)
        
        # Step 3: Project back to hidden size using TRANSPOSED space projection
        # This reuses the same weights as the forward space projection but in reverse:
        #
        # Forward (Q/K/V): hidden [B,T,H] -> space [B,T,S]
        #   Implemented as: hidden @ W_stored.T  (nn.Linear does this)
        #   where W_stored is [S, H]
        #
        # Backward (O): space [B,T,S] -> hidden [B,T,H] 
        #   We want: space @ W_stored (the "inverse" direction)
        #   F.linear(x, weight) computes: x @ weight.T
        #   So: F.linear(space, W_stored.t()) = space @ W_stored.t().T = space @ W_stored ✓
        
        o_space_key = self.component_to_space["O"]
        o_space_proj = self.space_projections[o_space_key]
        
        if isinstance(o_space_proj, nn.Identity):
            # Dense layer - identity projection
            attn_output = o_space_proj(attn_output)
        else:
            # Linkspace layer - use transposed weight to go from space back to hidden
            attn_output = F.linear(attn_output, o_space_proj.weight.t(), bias=None)

        return attn_output

