"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `mla.py`

Based on: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py

## RotaryEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..models.shared_space_config import SharedSpaceDecoderConfig


def create_norm_layer(hidden_size: int, config: SharedSpaceDecoderConfig) -> nn.Module:
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


# Helper function needed because it's called twice during RoPE,
# but I dumped it in the comments there.
# TODO - Nah, screw it, just write it twice! At least then you get
# to use the word 'query' instead of 'x'.
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """Precompute RoPE embeddings and store them as buffers."""

    def __init__(self, config: SharedSpaceDecoderConfig) -> None:
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

"""## MLA"""

class MultiheadLatentAttention(nn.Module):
    """
    A variant of MLA with:
    - Simplified RoPE handling:
      - A portion of the head dimensions are used for position information.
      - Same number of queries as keys. (no MQA)
    - Optional output subspace
    """

    def __init__(self, config: SharedSpaceDecoderConfig, layer_idx: int):
        super().__init__()

        self.config = config

        # Used to determine if this layer is dense or uses latents.
        self.layer_idx = layer_idx
        self.attention_dropout_prob = config.attention_dropout_prob

        self.num_heads = config.num_attention_heads

        self.rope_theta = config.rope_theta
        self.rope_dims = config.rope_dims
        self.nope_dims = config.nope_dims

        self.q_shared_dim = config.q_shared_dim
        self.kv_shared_dim = config.kv_shared_dim
        self.o_shared_dim = config.o_shared_dim

        self.qk_private_dim = config.qk_private_dim
        self.vo_private_dim = config.vo_private_dim

        self.hidden_size = config.hidden_size

        # =========================
        #     Input Projections
        # =========================

        # If this is one of the dense layers,
        if self.layer_idx < config.num_dense_layers:

            # =========================
            #     Dense Attention
            # =========================

            # No latent projections.
            self.latent_spaces = False

            # Define the standard QKV projection
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                self.num_heads * (self.qk_private_dim * 2 + self.vo_private_dim),
                bias=config.attention_bias,
            )

            # Dense output projection
            self.o_proj = nn.Linear(
                self.num_heads * self.vo_private_dim,
                config.hidden_size,
                bias=config.attention_bias,
            )

        # If we're past the dense layers,
        else:

            # =========================
            #     Latent Attention
            # =========================

            # Use latent projections.
            self.latent_spaces = True

            # Input latent projections
            
            # If we're using a shared query subspace,
            if config.q_shared_dim is not None:
                # Set a flag that we'll check in `forward`.
                self.query_shared = True

                self.q_shared_proj = nn.Linear(
                    config.hidden_size,
                    self.q_shared_dim,
                    bias=config.attention_bias,
                )
                
                self.q_shared_norm = create_norm_layer(self.q_shared_dim, config)               
                
            else:
                # Set a flag that we'll check in `forward`.
                self.query_shared = False

                self.q_shared_dim = config.hidden_size

                #print("Updated self.q_shared_dim to", self.q_shared_dim) 
                
                # Use identity.
                self.q_shared_proj = nn.Identity()
                self.q_shared_norm = nn.Identity()

            # If we're using a shared key/value subspace,
            if config.kv_shared_dim is not None:
                # Set a flag that we'll check in `forward`.
                self.keyvalue_shared = True

                self.kv_shared_proj = nn.Linear(
                    config.hidden_size,
                    self.kv_shared_dim,
                    bias=config.attention_bias,
                )

                self.kv_shared_norm = create_norm_layer(self.kv_shared_dim, config)
                
            else:
                # Set a flag that we'll check in `forward`.
                self.keyvalue_shared = False

                self.kv_shared_dim = config.hidden_size
                
                # Use identity.
                self.kv_shared_proj = nn.Identity()
                self.kv_shared_norm = nn.Identity()

            #print("config.q_shared_dim", config.q_shared_dim)
            #print("self.qk_private_dim", self.qk_private_dim)
            
            # Query heads
            self.q_private_proj = nn.Linear(
                self.q_shared_dim,
                self.num_heads * self.qk_private_dim,
                bias=False # TODO
            )

            # Key and Value heads, concatenated
            self.kv_private_proj = nn.Linear(
                self.kv_shared_dim,
                self.num_heads * (self.qk_private_dim + self.vo_private_dim),
                bias=False,
            )

            # Use output subspace if o_shared_dim is specified
            self.output_subspace = config.o_shared_dim is not None

            # If we're using an output subspace,
            if self.output_subspace:

                # ==========================
                #     Output Subspace
                # ==========================

                self.o_shared_dim = config.o_shared_dim

                # Per-head output projections
                # (Similar to original W^O, but projects the scored value vectors
                #  into a latent space instead of back to the model)
                self.o_private_proj = nn.Linear(
                    self.num_heads * self.vo_private_dim,
                    self.o_shared_dim,
                    bias=False
                )

                # Norm layer between o_private_proj and o_shared_proj
                # Note: In previous ViT experiments, this norm step hurt performance, but was beneficial
                #       in the DeepSeekV3 experiments.
                # However, we're making it configurable so it can be tested in different contexts.
                self.o_private_norm = create_norm_layer(self.o_shared_dim, config)

                # Shared output projection
                # The head outputs from `o_private_proj` are first summed together (across
                # heads) in the latent space.
                # Then we project their combined outputs (a single vector per token)
                # back to model space via `o_shared_proj`.
                self.o_shared_proj = nn.Linear(
                    self.o_shared_dim,
                    self.hidden_size,
                    bias=config.attention_bias
                )
            else:
                # Dense output projection
                self.o_proj = nn.Linear(
                    self.num_heads * self.vo_private_dim,
                    config.hidden_size,
                    bias=config.attention_bias,
                )

        # Softmax scaling factor.
        self.softmax_scale = self.qk_private_dim ** (-0.5)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        #past_key_value: Optional[Cache] = None, # TODO - Can I remove this?
        #cache_position: Optional[torch.LongTensor] = None, # TODO - Can I remove this?
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # === Tensor Dimension Symbols ===
        #    B: batch_size     — number of samples in the batch
        #    T: seq_len        — number of tokens per sample
        #    H: n_heads        — number of attention heads
        #    D: hidden_dim     — model embedding size
        #   Dv: vo_private_dim - per-head value/output projection dimension
        #   Dr: rope_dims      - The first Dr dimensions receive rope.
        #   Cq: q_shared_dim   - query shared subspace size
        #  Ckv: kv_shared_dim  - key-value shared subspace size
        #   Co: o_shared_dim   - output shared subspace size

        # Input token embeddings
        # hidden_states: [B, T, D]
        B, T = hidden_states.shape[:2]
        H = self.num_heads
        Dq = self.qk_private_dim     # per-head dim for Q and K
        Dv = self.vo_private_dim     # per-head dim for V/O

        Dc_q, Dc_kv = self.q_shared_dim, self.kv_shared_dim

        # ==============================
        #      QKV Head Projections
        # ==============================
        # Project tokens into per-head query, key, and value vectors

        # If this layer uses latent projections,
        if self.latent_spaces:

            # ================================
            #     Shared Space Projections
            # ================================

            # Project token embeddings into shared latents
            # Input:
            #     hidden_states [B, T, D]
            #     q_shared_proj [D, Cq]
            #    kv_shared_proj [D, Ckv]
            # Output:
            #          q_shared  [B, T, Cq]
            #          kv_shared [B, T, Ckv]

            # If we're using a shared query subspace,
            if self.q_shared_dim is not None:
                q_shared = self.q_shared_proj(hidden_states)

                # Normalize latent vectors, shapes unchanged.
                q_shared = self.q_shared_norm(q_shared)
            # Otherwise,
            else:
                # Use the hidden states
                q_shared = hidden_states

            # If we're using a shared key/value subspace,
            if self.kv_shared_dim is not None:

                # Project token embeddings into shared subspace.
                kv_shared = self.kv_shared_proj(hidden_states)

                # Normalize latent vectors, shapes unchanged.
                kv_shared = self.kv_shared_norm(kv_shared)
            # Otherwise,
            else:
                # Use the hidden states
                kv_shared = hidden_states

            # ======================================
            #     Per-Head (Private) Projections
            # ======================================

            # Project query latents onto query heads.
            # Input:
            #     q_shared       [B, T, Cq]
            #     q_private_proj [Cq, H*Dh]
            # Output:
            #     queries   [B, T, H*Dh]
            queries = self.q_private_proj(q_shared)

            # Project key/value latents onto key and value heads.
            # The key and value heads are all concatenated, each head occupies
            # Dh columns of the kv_private_proj. This yields the key and value
            # vectors concatenated in the same way.
            #
            # Input:
            #          kv_shared [B, T, Ckv]
            #    kv_private_proj [Ckv, 2*H*Dh]
            # Output:
            #     keysvalues [B, T, 2*H*Dh]
            keysvalues = self.kv_private_proj(kv_shared)

            # Split into key and value tensors
            # Each: [B, T, H * Dh]
            keys, values = keysvalues.chunk(2, dim=-1)

        # If this is a dense attention layer (no latent projections),
        else:

            # ====================
            #     Standard MHA
            # ====================

            # Standard QKV projection
            # Input:
            #   hidden_states     [B, T, D]
            #         qkv_proj    [D, 3*H*Dh]
            # Output:
            #   querieskeysvalues [B, T, 3*H*Dh]
            querieskeysvalues = self.qkv_proj(hidden_states)

            # Separate query, key, and value vectors
            # Each: [B, T, H * Dh]
            queries, keys, values = querieskeysvalues.chunk(3, dim=-1)

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
        # Apply rotary position embeddings to the first `self.rope_dims` of
        # each head.
        # The slice operations are free, but the concatenation is
        # not, because the outputs of the rotation operation are new data
        # occupying different memory. Still considered the best option,
        # though.

        # 1. Unpack the precomputed cosine and sine embeddings
        # Position embeddings is a tuple of
        #    (cos [seq_len, rope_dims],
        #     sin [seq_len, rope_dims])
        cos, sin = position_embeddings

        # 2. Split the query and key heads into the part to rotate and the part
        #    to pass through (early columns get position info, later ones don't)
        #
        #  (Using queries as example)
        #  Inputs:
        #    queries  [B, H, T, Dh]  Dh = rope_dims + not_rope_dims
        #  Outputs:
        #    q_rope   [B, H, T,  Dr]
        #    q_pass   [B, H, T, Dh-Dr]
        q_rope, q_pass = queries[..., :self.rope_dims], queries[..., self.rope_dims:]
        k_rope, k_pass =    keys[..., :self.rope_dims],    keys[..., self.rope_dims:]

        # 3. Apply the rotary embedding to the designated slice
        #
        # To broadcast cos and sin across the batch and head dimensions, we unsqueeze them.
        # Shape change: [T, Dr] -> [1, 1, T, Dr]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        #print("q_rope.shape[-1] // 2:", (q_rope.shape[-1] // 2))
        #print("x1 = x[..., :x.shape[-1] // 2 ].shape:", q_rope[..., :q_rope.shape[-1] // 2 ].shape)
        #print("sin/cos.shape:", cos.shape)
        #print("q_rope.shape:", q_rope.shape)
        #print("(q_rope * cos).shape:", (q_rope * cos).shape)
        #print("rotate_half(q_rope).shape:", rotate_half(q_rope).shape)
        #print("(rotate_half(q_rope) * sin).shape:", (rotate_half(q_rope) * sin).shape)
        """
        In this example   batch_size = 2, hum_heads = 8, seq_len = 65, rope_dims = 16

                        q_rope.shape[-1] // 2: 8
        x1 = x[..., :x.shape[-1] // 2 ].shape: torch.Size([2, 8, 65, 8])

                    sin/cos.shape: torch.Size([1, 1, 65, 16])  # After double unsqueeze.
                    vq_rope.shape: torch.Size([2, 8, 65, 16])

             (q_rope * cos).shape: torch.Size([2, 8, 65, 16])

        rotate_half(q_rope).shape: torch.Size([2, 8, 65, 16])
        (rotate_half(q_rope) * sin).shape: torch.Size([2, 8, 65, 16])
        """


        # Let's walk through the queries as the example.
        # What does rotate half do?
        #    dim -1 is the row vectors, the queries
        #
        #  Step 1: Split the vector in half.
        #    "q_rope.shape[-1] // 2" <- How much to select. Half the length of the q_rope vector
        #    x1 = x[..., :x.shape[-1] // 2 ]  # Select the first half of the vector.
        #    x2 = x[...,  x.shape[-1] // 2:]  # Select the second half.
        #
        #  Step 2:
        #      - Apply negative to the values in the second half.
        #      - Reverse the order of the halves.
        #    return torch.cat((-x2, x1), dim=-1)
        #
        # ---- (q_rope * cos) ----
        # Element-wise multiply the values in each `cos` vector with the
        # corresponding (i.e., same sequence position) `q_rope` vector.
        #
        # Inputs:
        #    q_rope  [B, H, T, Dr]
        #       cos  [1, 1, T, Dr]
        #
        # Outputs:
        #        x   [B, H, T, Dr]
        #
        # ---- (rotate_half(q_rope)) ----
        #  TODO
        #
        # Inputs:
        #       q_rope    [B, T, Dr]
        #
        # Outputs:
        #   rot_q_rope    [B, T, Dr]
        #
        # ---- rotated * sin ----
        #  TODO
        q_rotated = (q_rope * cos) + (rotate_half(q_rope) * sin)
        k_rotated = (k_rope * cos) + (rotate_half(k_rope) * sin)

        # 4. Concatenate the rotated and pass-through parts back together
        # Input (each): [B, H, T, Dr] and [B, H, T, Dq-Dr]
        # Output (each): [B, H, T, Dq]
        queries = torch.cat((q_rotated, q_pass), dim=-1)
        keys = torch.cat((k_rotated, k_pass), dim=-1)

        # ===================
        #       Attention
        # ===================
        # The tensors (queries, keys, values) now have shape [B, H, T, Dq]
        # and are ready for the attention score calculation.

        # Only apply dropout during training.
        # self.training is a pytorch flag.
        if self.training:
            dropout_p = self.attention_dropout_prob
        else:
            dropout_p = 0.0

        # Call SDPA / Flash Attention
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None, # attention_mask,
            dropout_p=dropout_p,
            scale=self.softmax_scale,
            is_causal=True, # This is a decoder - apply causal masking
        )

        # Reshape output back to [B, T, H * Dv] from [B, H, T, Dv]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * Dv)

        # =========================
        #     Output Projection
        # =========================

        # If we are using an output latent projection,
        if self.latent_spaces and self.output_subspace:

            # Project the attention output into the output latent space.
            # This is analogous to the W^O matrix in standard attention but
            # projects to an intermediate latent dimension.
            attn_output = self.o_private_proj(attn_output)

            # Apply normalization to the output latents
            attn_output = self.o_private_norm(attn_output)

            # Re-project the output latent representation back to model space.
            attn_output = self.o_shared_proj(attn_output)

        # If this is a dense layer,
        else:
            # Project the values back into model space.
            attn_output = self.o_proj(attn_output)

        # -----------------------------------------

        return attn_output

