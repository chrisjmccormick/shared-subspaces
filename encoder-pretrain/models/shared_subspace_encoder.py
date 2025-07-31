
"""SharedSubspaceEncoder skeleton.

This module contains very small stubs that mimic HuggingFace model classes.
The intent is to keep things easy to read while we experiment with shared
subspaces.

Style notes:
    - Keep assumptions explicit with ``assert`` and fail fast if a required
      field is missing.
    - Avoid inferring shapes or configuration defaults; raise an error instead
      of silently guessing.
    - Comment code that relies on HuggingFace conventions so newcomers can
      follow along.
    - Boilerplate should be minimal and clearly marked.

The model is encoder-only and will use RoPE for positional information rather
than a learnable embedding table.
"""

from typing import Optional

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class SharedSubspaceEncoderConfig(PretrainedConfig):
    """Configuration for :class:`SharedSubspaceEncoderModel`."""

    model_type = "shared_subspace_encoder"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        use_mla: bool = False,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        head_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        self.use_mla = use_mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.head_dim = head_dim

        # Explicitly mark this as an encoder-only architecture
        self.is_decoder = False


class SharedSubspaceEncoderPreTrainedModel(PreTrainedModel):
    """Base class with weight initialization."""

    config_class = SharedSubspaceEncoderConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module) -> None:
        """Weight initialization hook used by :class:`PreTrainedModel`.

        ``PreTrainedModel.post_init`` will recursively apply this function to
        every submodule right after construction.  HuggingFace models override
        it so that creating a model from scratch yields the same initialization
        as ``from_pretrained`` when no checkpoint is supplied.

        The modules themselves come with PyTorch defaults; this method simply
        enforces the initializer scheme used throughout the library.  It is not
        required, but leaving it out would lead to slightly different weight
        statistics.
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SharedSubspaceEncoderLayer(nn.Module):
    """Single transformer block using :class:`MultiheadLatentAttention`."""

    def __init__(self, config: SharedSubspaceEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = MultiheadLatentAttention(config, layer_idx)
        # TODO: add MLP and layer norms

    def forward(
        self,
        hidden_states: torch.Tensor,
        # ``position_embeddings`` carries the RoPE ``(cos, sin)`` tensors rather
        # than an index-based embedding lookup.
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class SharedSubspaceEncoderModel(SharedSubspaceEncoderPreTrainedModel):
    """Minimal encoder model with shared subspaces."""

    def __init__(self, config: SharedSubspaceEncoderConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # RoPE will supply position information; we intentionally omit a learned
        # position embedding table.
        self.layers = nn.ModuleList(
            [SharedSubspaceEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class MultiheadLatentAttention(nn.Module):
    """
    A variant of MLA with:
    - Simplified RoPE handling:
      - A portion of the head dimensions are used for position information.
      - Same number of queries as keys. (no MQA)
    - Optional output subspace
    """

    def __init__(self, config: SharedSubspaceEncoderConfig, layer_idx: int):
        super().__init__()
        
        self.config = config
        
        # Used to determine if this layer is dense or uses latents.
        self.layer_idx = layer_idx 
        self.attention_dropout_prob = config.attention_dropout_prob
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim # New / Add
        
        self.rope_theta = config.rope_theta
        self.rope_dims = config.rope_dims # New / Add

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        #self.qk_rope_head_dim = config.qk_rope_head_dim -- Remove
        #self.v_head_dim = config.v_head_dim # Remove
        #self.qk_nope_head_dim = config.qk_nope_head_dim -- Remove
        
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
        
        # If we're past the dense layers,
        else:

            # =========================
            #     Latent Attention
            # =========================

            # Use latent projections.
            self.latent_spaces = True

            # Input latent projections
            self.qkv_a_proj = nn.Linear(
                config.hidden_size,
                self.q_lora_rank + self.kv_lora_rank,
                bias=config.attention_bias,
            )

            # TODO - Decide whether to share or split.
            self.qkv_a_layernorm = DeepseekV3RMSNorm(
                self.q_lora_rank + self.kv_lora_rank,
                eps=config.rms_norm_eps,
            )

            # Query heads
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, 
                self.num_heads * self.head_dim, 
                bias=False # TODO
            )

            # Key and Value heads, concatenated
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_heads * (self.head_dim * 2),
                bias=False,
            )
            
        # ==========================
        #     Output Projections
        # ==========================

        self.output_subspace = config.output_subspace

        if self.output_subspace:
            
            # ==========================
            #     Output Subspace
            # ==========================

            self.o_lora_rank = config.o_lora_rank 

            # Per-head output projections
            # (Similar to original W^O, but projects the scored value vectors
            #  into a latent space instead of back to the model)
            self.o_a_proj = nn.Linear(
                self.num_heads * self.v_head_dim,
                self.o_lora_rank, 
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
            #    self.o_lora_rank, 
            #    eps=config.rms_norm_eps
            #)

            # Shared output projection
            # The head outputs from `o_a_proj` are first summed together (across
            # heads) in the latent space.
            # Then we project their combined outputs (a single vector per token)
            # back to model space via `o_b_proj`.
            self.o_b_proj = nn.Linear(
                self.o_lora_rank, 
                self.hidden_size, 
                bias=config.attention_bias
            )
   
        # Original output matrix
        else:
            # ========================
            #     Dense Output 
            # ========================

            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                config.hidden_size,
                bias=config.attention_bias,
            )

        
        #self.qk_head_dim = config.qk_nope_head_dim  -- Remove    
        #self.q_combined_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim -- Remove    
        # Define separate variables just for clarity in code on which one we're
        # actually dealing with, and to avoid the impression that it's the 
        # concatenation of the two.
        #self.q_head_dim = self.qk_head_dim - Remove
        #self.k_head_dim = self.qk_head_dim - Remove
        #self.v_head_dim -- Remove
        
        # This is not a decoder model
        self.is_causal = False # TODO - Is this needed by huggingface?
        
        # Softmax scaling factor.
        self.scaling = self.head_dim ** (-0.5)

        # TODO...
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None, # TODO - Can I remove this?
        cache_position: Optional[torch.LongTensor] = None, # TODO - Can I remove this?
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        batch_size, seq_length = hidden_states.shape[:-1]
        

        # ==========================
        #    QKV Head Projections
        # ==========================

        # If we're using the latent projections,
        if self.latent_spaces:

            # Project the tokens onto the latent spaces. This is two separate
            # projections, one for query latent and the other for the kv latent,
            # concatenated into one 'qkv_a' matrix. 
            input_latents = self.qkv_a_proj(hidden_states)
            
            # Apply normalization to the latents. 
            # (TODO - Ok to normalize these together?)
            input_latents = self.qkv_a_layernorm(input_latents)           
            
            # Break apart into query and key-value latents.
            q_latents, kv_latents = torch.split(input_latents, [self.q_lora_dim, self.kv_lora_dim], dim=-1)

            # Project the query latents onto the query heads. 
            queries = self.q_b_proj(q_latents)

            # Project the kv latents onto the key heads and value heads (kv)
            keysvalues = self.kv_b_proj(kv_latents)
            
            # Split them apart
            keys, values = torch.split(keysvalues, [self.head_dim, self.head_im], dim=-1)
            # TODO - Can einsum project and split?

        # If this is a dense attention layer (no latent projections--we'll do this in
        # early layers),        
        else:
            # Project the hidden states onto the query, key, and value projections.
            querieskeysvalues = self.qkv_proj(hidden_states)

            # Split them apart
            queries, keys, values = torch.split(querieskeysvalues, [self.head_dim, self.head_dim, self.head_dim], dim=-1)
        
        # ==================
        #        RoPE
        # ==================

        # Apply RoPE only to the last `rope_dims` dimensions of the querys and 
        # keys.
        # TODO...

        # ===================
        #      Attention
        # ===================

        # Reshape for SDPA / Flash Attention

        # TODO...

        # Invoke

        # TODO...

        # Reshape outputs if needed (TODO)

        # =========================
        #     Output Projection
        # =========================

        # If we are using an output latent projection,
        if self.output_subspace:

            # First, project the scored value vectors onto `o_a_proj`. This is
            # equivalent to projecting onto W^O in standard attention, except 
            # that here we are projecting into an intermediate latent space. 
            # This projection is unique per-head, preserving head diversity, and
            # then sums the results into a single vector per token.
            attn_output = self.o_a_proj(attn_output)

            # MLA uses RMSNorm on the query and key-value latents. It's not
            # clear yet whether this is helpful for the output.
            #attn_output = self.o_a_layernorm(attn_output)

            #print(f"attn_output after o_a_proj: {attn_output.shape}")

            # The input to `o_b_proj` is the summed output latents of the 
            # attention heads. This step re-projects this single per-token 
            # latent back to model space.
            attn_output = self.o_b_proj(attn_output)

        # If this is a dense layer,
        else:
            # Project the values back into model space.
            attn_output = self.o_proj(attn_output)

        # -----------------------------------------

        return attn_output  #, attn_weights - TODO - does transformers require these?
