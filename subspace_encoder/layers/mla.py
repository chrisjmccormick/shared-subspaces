"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `mla.py`

Based on: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py

## RotaryEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

    def __init__(self, config: SharedSpaceEncoderConfig) -> None:
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

    def __init__(self, config: SharedSpaceEncoderConfig, layer_idx: int):
        super().__init__()

        self.config = config

        # Used to determine if this layer is dense or uses latents.
        self.layer_idx = layer_idx
        self.attention_dropout_prob = config.attention_dropout_prob

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim # New / Add

        self.rope_theta = config.rope_theta
        self.rope_dims = config.rope_dims # New / Add

        self.q_latent_dim = config.q_latent_dim
        self.kv_latent_dim = config.kv_latent_dim

        # Explicit dimensional attributes for clarity
        self.hidden_size = config.hidden_size
        self.v_head_dim = config.head_dim

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
                self.num_heads * (self.head_dim * 3),
                bias=config.attention_bias,
            )

            # Dense output projection (independent of output_subspace)
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
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

            # Input latent projections, concatenated.
            self.qkv_a_proj = nn.Linear(
                config.hidden_size,
                self.q_latent_dim + self.kv_latent_dim,
                bias=config.attention_bias,
            )

            # Normalize the latents.
            self.q_a_norm = DeepseekV3RMSNorm(
                self.q_latent_dim,
                eps=config.rms_norm_eps,
            )
            self.kv_a_norm = DeepseekV3RMSNorm(
                self.kv_latent_dim,
                eps=config.rms_norm_eps,
            )

            # Query heads
            self.q_b_proj = nn.Linear(
                config.q_latent_dim,
                self.num_heads * self.head_dim,
                bias=False # TODO
            )

            # Key and Value heads, concatenated
            self.kv_b_proj = nn.Linear(
                self.kv_latent_dim,
                self.num_heads * (self.head_dim * 2),
                bias=False,
            )

            self.output_subspace = config.output_subspace

            if self.output_subspace:

                # ==========================
                #     Output Subspace
                # ==========================

                self.o_latent_dim = config.o_latent_dim

                # Per-head output projections
                # (Similar to original W^O, but projects the scored value vectors
                #  into a latent space instead of back to the model)
                self.o_a_proj = nn.Linear(
                    self.num_heads * self.v_head_dim,
                    self.o_latent_dim,
                    bias=False
                )

                # Regarding bias terms:
                #   - The thought here is to mirror the behavior on the input
                #     latents, where only one of the two projections receives a
                #     bias term.
                #     - Haven't yet experimented with this (i.e., whether to place
                #       it on a or b or neither)
                #
                # Regarding Layernorm:
                #   - In the ViT experiments, the addition of a layernorm between
                #     the o_a and o_b projections (i.e., applying it to the output
                #     of o_a) hurt performance.
                #   - I have not tried applying it to the output of o_b.
                #
                #self.o_a_layernorm = DeepseekV3RMSNorm(
                #    self.o_latent_dim,
                #    eps=config.rms_norm_eps
                #)

                # Shared output projection
                # The head outputs from `o_a_proj` are first summed together (across
                # heads) in the latent space.
                # Then we project their combined outputs (a single vector per token)
                # back to model space via `o_b_proj`.
                self.o_b_proj = nn.Linear(
                    self.o_latent_dim,
                    self.hidden_size,
                    bias=config.attention_bias
                )
            else:
                # Dense output projection
                self.o_proj = nn.Linear(
                    self.num_heads * self.head_dim,
                    config.hidden_size,
                    bias=config.attention_bias,
                )



        # Softmax scaling factor.
        self.softmax_scale = self.head_dim ** (-0.5)

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
        #   Dh: head_dim       - per-head projection dimension
        #   Dr: rope_dims      - The first Dr dimensions receive rope.
        #   Cq: q_latent_dim   - query latent subspace size
        #  Ckv: kv_latent_dim  - key-value latent subspace size
        #   Co: o_latent_dim   - output latent subspace size

        # Input token embeddings
        # hidden_states: [B, T, D]
        B, T = hidden_states.shape[:2]
        H, Dh = self.num_heads, self.head_dim
        Dc_q, Dc_kv = self.q_latent_dim, self.kv_latent_dim

        # ==============================
        #      QKV Head Projections
        # ==============================
        # Project tokens into per-head query, key, and value vectors

        # If this layer uses latent projections,
        if self.latent_spaces:

            # Project token embeddings into shared latents
            # Input:
            #     hidden_states [B, T, D]
            #      qkv_a_proj [D, Cq + Ckv]
            # Output:
            #     input_latents [B, T, Cq + Ckv]
            input_latents = self.qkv_a_proj(hidden_states)

            # Split latents for queries and keys/values
            # Input:
            #     input_latents [B, T, Cq + Ckv]
            # Outputs:
            #          q_latents  [B, T, Cq]
            #          kv_latents [B, T, Ckv]
            q_latents, kv_latents = torch.split(
                input_latents, [self.q_latent_dim, self.kv_latent_dim], dim=-1
            )

            # Normalize latent vectors, shapes unchanged.
            q_latents = self.q_a_norm(q_latents)
            kv_latents = self.kv_a_norm(kv_latents)

            # Project query latents onto query heads.
            # Input:
            #     q_latents [B, T, Cq]
            #      q_b_proj [Cq, H*Dh]
            # Output:
            #     queries   [B, T, H*Dh]
            queries = self.q_b_proj(q_latents)

            # Project key/value latents onto key and value heads.
            # The key and value heads are all concatenated, each head occupies
            # Dh columns of the k_b_proj. This yields the key and value vectors
            # concatenated in the same way.
            #
            # Input:
            #     kv_latents [B, T, Ckv]
            #      kv_p_proj [Ckv, 2*H*Dh]
            # Output:
            #     keysvalues [B, T, 2*H*Dh]
            keysvalues = self.kv_b_proj(kv_latents)

            # Split into key and value tensors
            # Each: [B, T, H * Dh]
            keys, values = keysvalues.chunk(2, dim=-1)

        # If this is a dense attention layer (no latent projections),
        else:
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
        queries = queries.view(B, T, H, Dh).transpose(1, 2)
        keys =       keys.view(B, T, H, Dh).transpose(1, 2)
        values =   values.view(B, T, H, Dh).transpose(1, 2)

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
        # TODO - Add shapes
        queries = torch.cat((q_rotated, q_pass), dim=-1)
        keys = torch.cat((k_rotated, k_pass), dim=-1)

        # ===================
        #       Attention
        # ===================
        # The tensors (queries, keys, values) now have shape [B, H, T, Dh]
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
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=self.softmax_scale,
            is_causal=False, # Not a decoder
        )

        # Reshape output back to [B, T, H * Dh] from [B, H, T, Dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # =========================
        #     Output Projection
        # =========================

        # If we are using an output latent projection,
        if self.latent_spaces and self.output_subspace:

            # Project the attention output into the output latent space.
            # This is analogous to the W^O matrix in standard attention but
            # projects to an intermediate latent dimension.
            attn_output = self.o_a_proj(attn_output)

            # Re-project the output latent representation back to model space.
            attn_output = self.o_b_proj(attn_output)

        # If this is a dense layer,
        else:
            # Project the values back into model space.
            attn_output = self.o_proj(attn_output)

        # -----------------------------------------

        return attn_output

