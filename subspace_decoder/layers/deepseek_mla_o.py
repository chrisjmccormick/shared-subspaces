
# Simplified DeepSeek V3 Attention with MLA-o
# Based on: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_output import CachedDecoderOutputs
from transformers.modeling_utils import Logger

from transformers import (
    DeepseekV3Config, 
    DeepseekV3RMSNorm, 
    DeepseekV3RotaryEmbedding, 
    DeepseekV3LinearScalingRotaryEmbedding, 
    DeepseekV3DynamicNTKScalingRotaryEmbedding, 
    DeepseekV3YarnRotaryEmbedding
)

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        # =========================
        #     Input Projections
        # =========================

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        # =========================
        #     Output Projection
        # =========================
        
        # RESEARCH MODIFICATION: Add optional output subspace projection
        # This mirrors the MLA implementation's output projection logic
        self.use_output_subspace = config.use_output_subspace
        self.o_latent_dim = config.o_latent_dim
        
        if self.use_output_subspace and self.o_latent_dim is not None:
            # Output latent projections
            self.o_a_proj = nn.Linear(
                self.num_heads * self.v_head_dim,
                self.o_latent_dim,
                bias=False
            )
            self.o_b_proj = nn.Linear(
                self.o_latent_dim,
                self.hidden_size,
                bias=config.attention_bias,
            )
        else:
            # Standard output projection
            self.o_proj = nn.Linear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=config.attention_bias,
            )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # =========================
        #     Input Projections
        # =========================

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # =========================
        #     Attention
        # =========================

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        # =========================
        #     Output Projection
        # =========================

        # RESEARCH MODIFICATION: Decomposed output projection
        if self.use_output_subspace and self.o_latent_dim is not None:
            # Project the attention output into the output latent space.
            # This is analogous to the W^O matrix in standard attention but
            # projects to an intermediate latent dimension.
            attn_output = self.o_a_proj(attn_output)

            # Re-project the output latent representation back to model space.
            attn_output = self.o_b_proj(attn_output)
        else:
            # Standard output projection
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0