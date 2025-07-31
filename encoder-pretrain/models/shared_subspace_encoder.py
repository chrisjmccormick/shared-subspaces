

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
        q_lora_rank: int | None = None,  # TODO - let's move away from 'lora'
        kv_lora_rank: int | None = None,
        o_lora_rank: int | None = None,
        head_dim: int | None = None,
        vocab_decompose: bool = False,
        vocab_rank: int = 128,
        ffn_decompose: bool = False,
        ffn_rank: bool = False
        
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
        self.vocab_decompose = vocab_decompose
        self.vocab_rank = vocab_rank

        # TODO - FFN decompose, rank, o_lora_rank, 
        
        # TODO - Is this needed by huggingface?
        # Explicitly mark this as an encoder-only architecture
        self.is_decoder = False


class SharedSubspaceEncoderPreTrainedModel(PreTrainedModel):
    """
    The **PreTrainedModel object:
      - Is instantiated when TODO
      - Initializes:
        - TODO
      - Provides access to TODO
      - Executes TODO    
    """

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
    """
    The **Layer object:
      - Is instantiated by TODO
      - Initializes:
        - TODO
      - Provides access to TODO
      - Executes TODO
    """

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
    """
    The **Model object:
      - Initializes:
        - The vocabulary embeddings (and optional decomposition)
        - All of the **Layer objects.
      - Provides interface to vocab embeddings.
      - Executes the whole model in `forward`.   
    """
    

    def __init__(self, config: SharedSubspaceEncoderConfig) -> None:
        super().__init__(config)

        # ============================
        #    Vocabulary Embeddings
        # ============================

        # Decide the length of the embedding vectors.
        
        # If we're decomposing them,
        if config.vocab_decompose:
            # Use the requested rank.
            embed_dim = config.vocab_rank
        # Otherwise,
        else:
            # They're the model size.
            embed_dim = config.hidden_size

        # Create the embedding table.
        self.vocab_embed = nn.Embedding(config.vocab_size, embed_dim)

        # Create a shared projection for the vocabulary.
        if config.vocab_decompose:
            self.embed_proj = nn.Linear(embed_dim, config.hidden_size, bias=False)
        #else:
            #self.embed_proj = None  # No. Keep things brittle to expose mistakes.

        # ===================
        #    Create Layers 
        # ===================
        
        layers = []
        
        # For each layer,
        for i in range(config.num_hidden_layers):
            # Create a **Layer, providing the config and indicating its number.
            layers.append(
                SharedSubspaceEncoderLayer(config, i)
            )
        
        self.layers = nn.ModuleList(layers)
                
        self.post_init() # TODO - Some examples of what's likely here?

    # TODO - Is this necessary? Or fluff?
    def get_input_embeddings(self) -> nn.Embedding:
        return self.vocab_embed

    # TODO - Why would we be setting the vocab? Is this how it gets
    #        loaded in from a saved checkpoint?
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.vocab_embed = value

    # TODO - Who calls this?
    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return token embeddings projected to model space."""
        
        # TODO - Provide return shape.
        x = self.vocab_embed(input_ids)
        
        if self.embed_proj is not None: # TODO - Use flag
            # TODO - Define shapes x, embed_proj, output
            x = self.embed_proj(x)
        
        return x

    # TODO - Should this be called the output matrix?
    #        Who calls this? Part of MLM training?
    def shared_vocab_matrix(self) -> torch.Tensor:
        """Return the tied input/output embedding matrix."""
        
        if self.embed_proj is None:  # TODO - Use the flag instead.
            
            return self.vocab_embed.weight.T  # TODO - Document shapes here and below.
        # TODO!!! - Excuse me??? No. Project the residual stream through the shared projection
        # into the lower dimensional space, then multiply with the largest single weight matrix
        # in the whole freaking model.
        #return self.embed_proj.weight @ self.vocab_embed.weight.T
        return "TODO"

    # Comment--evaluates the model on (describe expected input shape) and returns (describe)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

# Helper function needed because it's called twice during RoPE,
# but I dumped it in the comments there.
# TODO - Nah, screw it, just write it twice! At least then you get
# to use the word 'query' instead of 'x'.
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    

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

        # Explicit dimensional attributes for clarity
        self.hidden_size = config.hidden_size
        self.v_head_dim = config.v_head_dim
        self.q_lora_dim = config.q_lora_rank
        self.kv_lora_dim = config.kv_lora_rank

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
        # === Tensor Dimension Symbols ===
        #    B: batch_size     — number of samples in the batch
        #    T: seq_len        — number of tokens per sample
        #    H: n_heads        — number of attention heads
        #    D: hidden_dim     — model embedding size
        #   Dh: head_dim       — per-head projection dimension
        #   Cq: q_latent_dim   - query latent subspace size
        #  Ckv: kv_latent_dim  - key-value latent subspace size
        #   Co: o_latent_dim   - output latent subspace size


        # Input token embeddings
        # hidden_states: [B, T, D]
        B, T = hidden_states.shape[:2]
        H, Dh = self.num_heads, self.head_dim
        Dc_q, Dc_kv = self.q_lora_dim, self.kv_lora_dim
        

        # ==============================
        #     QKV Head Projections
        # ==============================
        # Project tokens into per-head query, key, and value vectors

        # If this layer uses latent projections,
        if self.latent_spaces:

            # Project token embeddings into shared latents
            # Input:  hidden_states [B, T, D]
            # Output: input_latents [B, T, Dc_q + Dc_kv] TODO
            input_latents = self.qkv_a_proj(hidden_states)

            # Normalize latent vectors
            # Input:  input_latents [B, T, Dc_q + Dc_kv] TODO
            # Output: input_latents [B, T, Dc_q + Dc_kv] TODO
            input_latents = self.qkv_a_layernorm(input_latents)

            # Split latents for queries and keys/values
            # Input:  
            #    input_latents [B, T, Dc_q + Dc_kv]  TODO
            # Outputs:
            #       q_latents  [B, T, Dc_q]  TODO
            #       kv_latents [B, T, Dc_kv]  TODO
            q_latents, kv_latents = torch.split(
                input_latents, [self.q_lora_dim, self.kv_lora_dim], dim=-1
            )

            # Linear projection of query latents
            # Input:  
            #    q_latents [B, T, Cq]
            #     q_b_proj  TODO
            # Output: 
            #    queries   [B, T, H*Dh]
            queries = self.q_b_proj(q_latents)

            # Linear projection of key/value latents
            # Input:  kv_latents [B, T, Dc_kv]
            # Output: keysvalues [B, T, H * 2 * Dh]
            keysvalues = self.kv_b_proj(kv_latents)

            # Split into key and value tensors
            # Each: [B, T, H * Dh]
            keys, values = keysvalues.chunk(2, dim=-1)
            # TODO - Can einsum project and split?

        # If this is a dense attention layer (no latent projections),
        else:
            # Standard QKV projection
            # Input:  
            #   hidden_states [B, T, D]
            #   qkv_proj     [ TODO ]
            # Output: 
            #   querieskeysvalues [B, T, H * 3 * Dh]
            querieskeysvalues = self.qkv_proj(hidden_states)

            # Separate query, key, and value vectors
            # Each: [B, T, H * Dh]
            queries, keys, values = querieskeysvalues.chunk(3, dim=-1)
        
        # TODO - Add to style conventions. When performing the same operation
        #        for multiple things, document the first one then say "repeat for..."
        
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
        
        # TODO - Style guide: Align dimensions where feasible.
        
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
        # Assuming position_embeddings is a tuple (cos, sin) <---TODO
        # and each tensor has shape [T, self.rope_dims]
        cos, sin = position_embeddings

        # 2. Split the query and key heads into the part to rotate and the part to pass through
        q_rope, q_pass = queries[..., :self.rope_dims], queries[..., self.rope_dims:]
        k_rope, k_pass =    keys[..., :self.rope_dims],    keys[..., self.rope_dims:]

        # 3. Apply the rotary embedding to the designated slice
        # To broadcast cos and sin across the batch and head dimensions, we unsqueeze them.
        # Shape change: [T, rope_dims] -> [1, 1, T, rope_dims]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # `rotate_half`:
        #    dim -1 is the row vectors, the queries.        
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
        # TODO - What are the shapes of `sin` and `cos`.
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

        # Call SDPA or Flash Attention
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout_prob if self.training else 0.0,
        )
        
        # Reshape output back to [B, T, H * Dh]
        # TODO - Add shapes
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # =========================
        #     Output Projection
        # =========================

        # If we are using an output latent projection,
        if self.output_subspace:

            # TODO - Move this comment out.
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


class SubspaceFeedForward(nn.Module):
    """
    Feed-forward block for SharedSubspaceEncoder.
    
    Implements SwiGLU:
        FFN(x) = W_out( Swish(W_in(x)) ⊙ W_gate(x) ) + residual

    Supports both dense and decomposed MLP variants.

    Dense:
        - W_in:   Linear(hidden_dim → intermediate_dim)
        - W_gate: Linear(hidden_dim → intermediate_dim)
        - W_out:  Linear(intermediate_dim → hidden_dim)

    Decomposed:
        - W_in_shared:   Linear(hidden_dim → rank, bias=False)
        - W_in:          Linear(rank → intermediate_dim)
        - W_gate_shared: Linear(hidden_dim → rank, bias=False)
        - W_gate:        Linear(rank → intermediate_dim)
        - W_out:         Linear(intermediate_dim → rank, bias=False)
        - W_out_shared:  Linear(rank → hidden_dim)

    Residual, dropout, and post-norm are handled inside the block.
    """

    def __init__(self, config):
        super().__init__()

        
        #dropout_prob = config.hidden_dropout_prob # TODO - Style -- don't define variables if only used once.     

        # TODO - Style guide + changes -- Let's use self.cfg for stuff instead of copying.
        #        The bot can maintain a config validator utility for us to sanity check it at the beginning of 
        #        every script.
        #        Exceptions are stuff like below where the dimensions are used a bunch of times briefly and then
        #        discarded.
        self.cfg = config

        # getattr(config, "ffn_decompose", False) <-- reasonable, but let's stay brittle.
        
        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size # TODO - Find something shorter, and use the same name.


        # Define weights for the decomposed version.
        if self.cfg.ffn_decompose:
            # Verify that the config has an `ffn_rank` value, and that it's
            # been set. 
            assert hasattr(config, "ffn_rank") and config.ffn_rank is not None, \
                "Must specify `ffn_rank` when `ffn_decompose=True`."
            
            rank = config.ffn_rank

            # === Input Projections ===
            self.W_in_shared = nn.Linear(hidden_dim, rank, bias=False)
            self.W_in = nn.Linear(rank, intermediate_dim)

            # === Gate Projections ===
            self.W_gate_shared = nn.Linear(hidden_dim, rank, bias=False)
            self.W_gate = nn.Linear(rank, intermediate_dim)

            # === Output Projection ===
            self.W_out = nn.Linear(intermediate_dim, rank, bias=False)
            self.W_out_shared = nn.Linear(rank, hidden_dim)

        else:
            # === Dense FFN Projections ===
            self.W_in = nn.Linear(hidden_dim, intermediate_dim)
            self.W_gate = nn.Linear(hidden_dim, intermediate_dim)
            self.W_out = nn.Linear(intermediate_dim, hidden_dim)

        self.dropout = nn.Dropout(self.cfg.hidden_dropout_prob)
        self.norm = nn.LayerNorm(hidden_dim, eps=self.cfg.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Tensor Dimension Symbols ===
        # B: batch_size     — number of samples in the batch
        # T: seq_len        — number of tokens per sample
        # D: hidden_dim     — model embedding size
        # R: ffn_rank       — latent shared subspace dimension
        # D_ff: intermediate_size — FFN hidden dimension

        residual = x  # [B, T, D]

        # =========================
        #    Gated Feedforward
        # =========================
        
        if self.use_decomposition:
            # ==================
            #     Decomposed
            # ==================

            # Input:  x [B, T, D]
            # Output: x_proj [B, T, D_ff]
            x_proj = self.W_in(self.W_in_shared(x))

            # Input:  x [B, T, D]
            # Output: gate [B, T, D_ff]
            gate = self.W_gate(self.W_gate_shared(x))

            # SwiGLU nonlinearity
            x = F.silu(x_proj) * gate  # [B, T, D_ff]

            # Output: x [B, T, D]
            x = self.W_out_shared(self.W_out(x))

        else:
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

        x = self.dropout(x)
        x = self.norm(x + residual)  # [B, T, D]
        return x
