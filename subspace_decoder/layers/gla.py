"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `gla.py`

Based on: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.shared_space_config import SharedSpaceDecoderConfig


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

"""## GLA"""

class GroupedLatentAttention(nn.Module):
    """
    This version of Multihead Latent Attention applies the re-ordering trick from DeepSeekV3.
    Instead of comparing the queries and keys in the query-key space, we compare them in the 
    kv-shared space. 
    
    For clarity, I've re-interpreted the naming of the heads, and am framing it as MQA.
    What were previously labeled the query and key heads are now treated as a low-rank decomposition
    of the query heads. 
    What we considered the "shared key/value space" is now a single key head that is also used as the
    value head.
    Finally, what we previously labeled the value and output heads are now treated as a low-rank 
    decomposition of the output heads.

    This interpretation / implementation is designed to leverage the performance benefits of GQA.
    The trade-off is that the query-key matching space is now larger--it will require a greater
    number of calculations to match the queries to the keys. The hope is that the memory bandwidth
    savings will outweigh the increased computational cost.

    The same applies to the value-output space.

    Note that, although the query-key and value-output spaces are now large, the low-rank 
    decomposition of the query heads and output heads ensures that the heads are still effectively
    low rank / not over-parameterized.

    Finally, note that this implementation also supports the optional use of shared spaces on 
    the query and output sides.
    
    I've named the class "GroupedLatentAttention" because I may expand it to support multiple
    key/value heads (i.e., multiple groups of query heads) in the future.

    ==== Adding RoPE to VO ====

    ### **Attempt**

    We're extending Rotary Position Embeddings (RoPE) beyond the query-key interaction to the **value-output path** in Multihead Latent Attention (MLA).

    * In DeepSeek-V3's MLA framing, the same **full-rank key/value head** provides both the keys (for patterns) and the values (for messages).
    * Queries and output heads are low-rank bottlenecks, effectively serving as vocabularies of **pattern directions** (Q) and **message directions** (O).
    * Standard RoPE only modulates the Q–K dot product. Our attempt is to also apply RoPE phases consistently in the V–O pathway, so that **positional dependence is preserved in both the matching (QK) and messaging (VO) sides**.

    --

    ### **Hypothesis**

    If we rotate value vectors by their **source position phase** and then apply the **inverse rotation at the destination** before output projection, the model gains a clean **relative-position equivariance** in the message path, mirroring the property RoPE provides for queries and keys.

    This should:

    1. Make the 1-to-1 correspondence between "pattern templates" (Q) and "message templates" (O) more consistent.
    2. Reduce the burden on output heads to learn ad-hoc positional compensation.
    3. Improve long-context generalization, since both attention matching *and* message passing would share the same relative-position geometry.


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
        # What was previously considered the key/value shared dimension is now the 
        # size of the MQA style single key/value head.
        self.kv_head_dim = config.kv_shared_dim
        self.o_shared_dim = config.o_shared_dim

        # What was previously the query/key head size is now the size of 
        # the query head decomposition.
        self.q_inner_dim = config.qk_private_dim
        
        # What was previously the value/output head size is now the size of 
        # the output head decomposition.
        self.o_inner_dim = config.vo_private_dim

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

            print("config.q_shared_dim", config.q_shared_dim)
            
            # ==========================
            #     Shared Query Space
            # ==========================

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
                print("Using identity for shared projection.")
                # Set a flag that we'll check in `forward`.
                self.query_shared = False

                self.q_shared_dim = config.hidden_size

                #print("Updated self.q_shared_dim to", self.q_shared_dim) 
                
                # Use identity.
                self.q_shared_proj = nn.Identity()
                self.q_shared_norm = nn.Identity()

            # ==========================
            #     Shared Output Space
            # ==========================

            # If we're using a shared output space,
            if config.o_shared_dim is not None:
                # Set a flag that we'll check in `forward`.
                self.output_shared = True

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

                self.o_shared_norm = create_norm_layer(self.o_shared_dim, config)

            else:
                # Set a flag that we'll check in `forward`.
                self.output_shared = False
                self.o_shared_dim = config.hidden_size
                
                # Use identity.
                self.o_shared_proj = nn.Identity()
                self.o_shared_norm = nn.Identity()

            # ================================
            #      Decomposed Query Heads
            # ================================

            # Query down projections.
            # The query head inner dimension makes the head low rank, as usual.
            self.q_priv_a_proj = nn.Linear(
                self.q_shared_dim,
                self.num_heads * self.q_inner_dim,
                bias=False
            )

            # Query up projections.
            # We project back to the larger key/value space.
            # Rather than create a linear and break it apart, we can create our 
            # desired shapes.
            #  per-head Dq_c -> Dkv     (store as [H, Dq_c, Dkv])
            self.q_priv_b_weight = nn.Parameter(
                torch.empty(self.num_heads, self.q_inner_dim, self.kv_head_dim)
            )
            nn.init.kaiming_uniform_(self.q_priv_b_weight, a=math.sqrt(5))

            # ====================================
            #      Single Joint Key/Value Head
            # ====================================

            # The single joint key/value head.
            self.kv_priv_proj = nn.Linear(
                self.hidden_size,
                self.kv_head_dim,
                bias=False,
            )

            self.kv_priv_norm = create_norm_layer(self.kv_head_dim, config)

            # ================================
            #      Decomposed Output Heads
            # ================================

            # Down: values [B,H,T,Dkv] -> per-head Do_c using weights [H, Dkv, Do_c]
            self.o_priv_a_weight = nn.Parameter(
                torch.empty(self.num_heads, self.kv_head_dim, self.o_inner_dim)
            )
            nn.init.kaiming_uniform_(self.o_priv_a_weight, a=math.sqrt(5))

            # Output up projections.
            
            # We project back to the larger output subspace (or the model space, 
            # if no subspace is used).
            self.o_priv_b_proj = nn.Linear(
                self.num_heads * self.o_inner_dim,
                self.o_shared_dim,
                bias=False
            )

        # Let SDPA choose 1/sqrt(E). If you want explicit: self.kv_head_dim ** -0.5
        self.softmax_scale = None


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
        #     B: batch_size     — number of samples in the batch
        #     T: seq_len        — number of tokens per sample
        #     H: n_heads        — number of attention heads
        #     D: hidden_dim     — model embedding size
        #  Dq_c: q_inner_dim    - per-head decomposition dim for Q
        Dq_c = self.q_inner_dim        # per-head inner dim for Q
        #  Do_c: o_inner_dim    - per-head decomposition dim for O
        Do_c = self.o_inner_dim        # per-head inner dim for O
        #   Dkv: kv_head_dim    - Head size of the joint key/value head
        Dkv  = self.kv_head_dim        # Head size of the joint key/value head
        #    Dr: rope_dims      - The first Dr dimensions receive rope.
        #  Dq_s: q_shared_dim   - query shared subspace size
        Dq_s = self.q_shared_dim 
        #  Do_s: o_shared_dim   - output shared subspace size
        Do_s = self.o_shared_dim

        # Input token embeddings
        # hidden_states: [B, T, D]
        B, T = hidden_states.shape[:2]
        H = self.num_heads
        
        

        # =============================
        #     Shared Query Space 
        # =============================
        # These are set to identity if no shared query space is used.

        # Project token embeddings into shared latents
        # Input:
        #     hidden_states [B, T, D]
        #     q_shared_proj [D, Dq_s]
        #    kv_shared_proj [D, Dkv]
        # Output:
        #          q_shared  [B, T, Dq_s]
        #          kv_shared [B, T, Dkv]
        q_shared = self.q_shared_proj(hidden_states)

        # Normalize latent vectors, shapes unchanged.
        q_shared = self.q_shared_norm(q_shared)

        # ================================
        #     Decomposed Query Heads
        # ================================


        # Project query latents onto decomposed query heads.
        # 
        # Down projection ('a')
        # Input:
        #     q_shared       [B, T, Dq_s]
        #     q_priv_a_proj [Dq_s, H*Dq_c]
        # Output:
        #     queries_c   [B, T, H*Dq_c]
        queries_c = self.q_priv_a_proj(q_shared)

        # Split the vectors by head
        # Input:
        #     queries_c        [B, T, H*Dq_c]
        # Output:
        #     queries_c   [B, T, H, Dq_c]
        queries_c = queries_c.view(B, T, H, Dq_c)
        
        # Up projection ('b')
        # Input:
        #     queries_c        [B, T, H, Dq_c]
        #     q_priv_b_weight        [H, Dq_c, Dkv]    
        # Output:
        #     queries     [B, H, T, Dkv]
        queries = torch.einsum("bthd,hdc->bhtc", queries_c, self.q_priv_b_weight)

        # ===================================
        #     Single Joint Key/Value Head
        # ===================================
    
        # Project token embeddings into single joint key/value head.
        # Input:
        #     hidden_states [B, T, D]
        #     kv_priv_proj [D, Dkv]
        # Output:
        #     keyvalue [B, T, Dkv]
        keyvalue = self.kv_priv_proj(hidden_states)

        # Apply QK normalization.
        keyvalue = self.kv_priv_norm(keyvalue)
        
        # Prepare the queries and keyvalue vectors for RoPE and flash attention.
        # We have multiple query heads, and the queries are in `queries`.
        # We have a single key head, and the keyvector is in `keyvalue`.

        # Move the head dimension to the front, so for each head, we have
        # a series of vectors for each token in the sequence.
        #
        # Inputs:
        #   keyvalue  [B, T, Dkv]
        # Output:
        #   keyvalue   [B, 1, T, Dkv]
        keyvalue = keyvalue.unsqueeze(1)

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
        #    queries  [B, H, T, Dkv]  Dkv = rope_dims + not_rope_dims
        #  Outputs:
        #    q_rope   [B, H, T, Dr]
        #    q_pass   [B, H, T, Dkv-Dr]
        q_rope, q_pass = queries[..., :self.rope_dims], queries[..., self.rope_dims:]
        k_rope, k_pass =   keyvalue[..., :self.rope_dims],   keyvalue[..., self.rope_dims:]

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
        # Input (each): [B, H, T, Dr] and [B, H, T, Dkv-Dr]
        # Output (each): [B, H, T, Dkv]
        # (Where h = 1 for the key head and h = num_heads for the query heads)
        queries = torch.cat((q_rotated, q_pass), dim=-1)
        keyvalue = torch.cat((k_rotated, k_pass), dim=-1)

        # ====================
        #      GQA / MQA
        # ====================
        # GPT says that flash attention will infer the broadcasting, so `expand` is not needed.
        #
        # We need to use the `expand` operation to broadcast the keyvalue vector
        # across the query heads.
        # Input:
        #     keyvalue [B, 1, T, Dkv]
        # Output:
        #     keyvalue [B, H, T, Dkv]
        #keyvalue = keyvalue.expand(-1, H, -1, -1)

        # ===================
        #       Attention
        # ===================
        # We're ready for the attention score calculation.

        # Only apply dropout during training.
        # self.training is a pytorch flag.
        if self.training:
            dropout_p = self.attention_dropout_prob
        else:
            dropout_p = 0.0

        # Call SDPA / Flash Attention
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # Apply MQA / GQA. In this case, we have a single key head, and multiple query heads.
        values = F.scaled_dot_product_attention(
            queries,
            keyvalue, # Single key vector (joint with value) for GQA / MQA.
            keyvalue, # Single value vector (joint with key) for GQA / MQA.
            attn_mask=None, # attention_mask,
            dropout_p=dropout_p,
            scale=self.softmax_scale,
            is_causal=True, # This is a decoder - apply causal masking
        )

        # Attention outputs:
        # values [B, H, T, Dkv]

        # The final Dr dims of the value vectors carry RoPE information.
        # We can either (1) add position dependence to the value-output process,
        # or (2) we can strip off the RoPE information and only use the non-RoPE parts.

        # Let's try option 1!

        # Split the values into the RoPE and non-RoPE parts.
        # Input:
        #     values [B, H, T, Dkv]
        # Output:
        #     values_rope [B, H, T, Dr]
        #     values_pass [B, H, T, Dkv-Dr]
        values_rope, values_pass = values[..., :self.rope_dims], values[..., self.rope_dims:]

        # Fold the query RoPE information into the value vectors.
        # Inverse rotation: R_{-θ} x  =  (x * cos)  - (rotate_half(x) * sin)
        # Input:
        #     values_rope [B, H, T, Dr]
        #            cos  [1, 1, T, Dr]
        #            sin  [1, 1, T, Dr]
        # Output:
        #     values_unrot [B, H, T, Dr]
        values_unrot = (values_rope * cos) - (rotate_half(values_rope) * sin)

        # Now the values have the offset information in their rope dimensions,
        # and the output heads can learn to use it.
        values = torch.cat((values_unrot, values_pass), dim=-1)  # [B,H,T,Dkv]

        # =========================
        #     Output Projection
        # =========================


        # Project the values onto the decomposed output heads.
        # Output down projection heads.
        # Input:
        #            values  [B, H, T, Dkv]
        #   o_priv_a_weight     [H, Dkv, Do_c]
        # Output:
        #         outputs_c  [B, H, T, Do_c]
        outputs_c = torch.einsum("bhtd,hdc->bhtc", values, self.o_priv_a_weight)

        # For the up projection, we can concatenate the 'outputs_c' vectors by head,
        # (in the same way we would usually concatenate the value vectors)
        # Input:
        #    outputs_c  [B, H, T, Do_c]
        # Output:
        #   outputs_c  [B, T, H*Do_c]
        
        outputs_c = outputs_c.permute(0, 2, 1, 3).contiguous().view(B, T, H * Do_c)

        # Project up to the shared output space and sum across the output heads.
        # Input:
        #    outputs_c  [B, T, H*Do_c]
        #    o_priv_b_proj [H*Do_c, Do_s]
        # Output:
        #    output_s  [B, T, Do_s]
        output_s = self.o_priv_b_proj(outputs_c)
        
        # Apply normalization to the output latents
        output_s = self.o_shared_norm(output_s)

        # Re-project the output latent representation back to model space.
        # Input:
        #    output_s      [B, T, Do_s]
        #    o_shared_proj [Do_s, D]
        # Output:
        #    attn_output   [B, T, D]
        attn_output = self.o_shared_proj(output_s)

        # TODO - Not currently supported.
        # If this is a dense layer,
        # Project the values back into model space.
        # attn_output = self.o_proj(attn_output)

        # -----------------------------------------

        return attn_output

