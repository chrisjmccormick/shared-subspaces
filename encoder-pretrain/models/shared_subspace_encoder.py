

from typing import Optional

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class SharedSubspaceEncoderConfig(PretrainedConfig):
    r"""
    Configuration class for SharedSubspaceEncoderConfig.

    Extends the HuggingFace `PretrainedConfig` to support architectural 
    variations including:
    - Multi-Head Latent Attention (MLA)
    - Decomposed MLPs (low-rank FFNs)
    - Flexible attention backends (eager, flash, sdpa)
    - Explicit shared subspaces for Q, K, V, and O projections

    This config does not infer any defaults based on `hidden_size`. All 
    dimensions and ranks must be explicitly specified. If required values are 
    missing, a `ValueError` is raised during initialization.

    ----------------------
    Core BERT Parameters:
    ----------------------
    - vocab_size (`int`) — Vocabulary size.
    - hidden_size (`int`) — Model hidden dimension.
    - num_hidden_layers (`int`) — Number of transformer blocks.
    - num_attention_heads (`int`) — Number of attention heads.
    - intermediate_size (`int`) — Feed-forward hidden dimension.
    - hidden_act (`str`) — Activation function.
    - hidden_dropout_prob (`float`) — Dropout after projections and FFNs.
    - attention_dropout_prob (`float`) — Dropout applied to attention scores.
    - max_position_embeddings (`int`) — Max sequence length.
    - initializer_range (`float`) — Stddev of weight init.
    - layer_norm_eps (`float`) — Epsilon for LayerNorm.
    - classifier_dropout (`float` or None) — Dropout for final classifier.

    - vocab_subspace
    _ vocab_rank

    ----------------------
    Multi-Head Latent Attention (MLA):
    ----------------------
    - q_latent_dim (`int`) — Rank of the shared query subspace.
    - kv_latent_dim (`int`) — Rank of the shared key/value subspace.
    - rope_dims (`int`) — Number of head dimensions carrying RoPE.    
    - output_subspace (`bool`) — Whether to use a shared latent subspace for output projections.
    - o_latent_dim (`int`) — Rank of the shared output subspace (required if `output_subspace=True`).
    - rope_theta (`float`) — Base frequency used for RoPE.
    - rope_scaling (`dict` or None) — HF-style scaling dict for RoPE.
    - attention_bias (`bool`) — Whether to include bias terms in Q/K/V projections.
    - num_dense_layers (`int`) — Number of leading layers that do not use 
                                 subspaces for attention or FFNs. 
    - attention_backend (`str`) — Must be one of `"eager"`, `"flash"`, or `"sdpa"`.

    ----------------------
    Decomposed MLP (Low-Rank FFN):
    ----------------------
    - ffn_decompose (`bool`) — Whether to enable low-rank FFNs.
    - ffn_rank (`int`) — Rank of the shared FFN latent space (required if `ffn_decompose=True`).

    ----------------------
    Validation Behavior:
    ----------------------
    Raises `ValueError` at init time if:
    - FFN decomposition is enabled without specifying `ffn_rank`.
    - An unknown `attention_backend` is provided.
    """

    model_type = "shared_subspace_encoder"

    def __init__(
        self,
        
        # === Core Model ===
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        max_position_embeddings: int = 2048,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=None,

        vocab_subspace=False,
        vocab_rank=None,

        # === Multi-Head Latent Attention ===
        q_latent_dim=None,
        kv_latent_dim=None,
        rope_dims=None,
        head_dim=None,
        output_subspace=False,
        o_latent_dim=None,
        attention_backend="eager",
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,

        # === MLA Composition ===
        num_dense_layers=0,  # dense MHA layers before MLA starts

        # === Decomposed MLP ===
        ffn_decompose=False,
        ffn_rank=None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        # === Core Model ===
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout

        self.vocab_subspace = vocab_subspace
        self.vocab_rank = vocab_rank

        # === MLA ===
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.output_subspace = output_subspace
        self.o_latent_dim = o_latent_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.num_dense_layers = num_dense_layers

        # === Decomposed FFN ===
        self.ffn_decompose = ffn_decompose
        self.ffn_rank = ffn_rank

        # === Attention backend ===
        self.attention_backend = attention_backend

        # === Validation ===
        self._validate()

        print(
            f"  > SubspaceBertConfig.init - {self.num_hidden_layers}l - mla{self.use_mla} - ndense{self.num_dense_layers} - dcmp{self.ffn_decompose}\n"
        )
        print(f"    - attention backend: {self.attention_backend}\n")

    def _validate(self):
        # === MLA Validation ===
        if self.use_mla:
            if self.output_subspace and self.o_latent_dim is None:
                raise ValueError("`o_latent_dim` must be set when `output_subspace=True`")

        # === Decomposed FFN ===
        if self.ffn_decompose and self.ffn_rank is None:
            raise ValueError("`ffn_rank` must be set when `ffn_decompose=True`")

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
      - Is instantiated by :class:`SharedSubspaceEncoderModel` for each
        Transformer block in the encoder.
      - Initializes:
        - ``self_attn`` – multi-head latent attention implementing either
          dense or latent projections depending on the configuration.
        - ``mlp`` – a :class:`SubspaceFeedForward` block.
        - Dropout and LayerNorm used for the residual connection around the
          attention block.
      - Provides access to the attention and feed-forward submodules via the
        attributes ``self_attn`` and ``mlp``.
      - Executes a single encoder block in :meth:`forward`.
    """

    def __init__(self, config: SharedSubspaceEncoderConfig, layer_idx: int) -> None:
        
        super().__init__()

        # 
        self.self_attn = MultiheadLatentAttention(config, layer_idx)

        # Feed-forward network used after attention
        self.mlp = SubspaceFeedForward(config)

        # Residual components for the attention block
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, "layer_norm_eps", getattr(config, "eps", 1e-5)),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        # ``position_embeddings`` carries the RoPE ``(cos, sin)`` tensors rather
        # than an index-based embedding lookup.
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        
        # === Tensor Dimension Symbols ===
        #  B: batch_size    — number of samples in the batch
        #  T: seq_len       — number of tokens per sample
        #  D: hidden_dim    — model embedding size

        residual = hidden_states  # [B, T, D]

        # ==============================
        #     Self Attention
        # ==============================
        attn_out = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
        )

        # TODO - Make sure we're following modern best practices here, not just
        #        copying BERT. Pretty sure DS-V3 uses input norms rather than this.

        attn_out = self.attn_dropout(attn_out)
        hidden_states = self.attn_norm(attn_out + residual)

        # ==============================
        #     Feed Forward
        # ==============================
        hidden_states = self.mlp(hidden_states)

        return hidden_states


class SharedSubspaceEncoderModel(SharedSubspaceEncoderPreTrainedModel):
    """
    The **Model object:
      - Initializes:
        - The vocabulary embeddings (and optional decomposition)
        - All of the **Layer objects.
      - Provides interface to vocab embeddings.
      - Executes the whole model in `forward`.   
      
      TODO / Note - The projection needs to be retained for creating the
      residual stream, but otherwise can be fused into the attention input
      heads after training. Or maybe the layer 0 attention heads should have
      a different input size from the start.
      
    """

    def __init__(self, config: SharedSubspaceEncoderConfig) -> None:
        super().__init__(config)

        # ============================
        #    Vocabulary Embeddings
        # ============================
        # Decomposing the vocabulary (if enabled) defines a shared projection
        # which constrains the model to store semantic information (and 
        # whatever other static token knowledge) into a limited set of 
        # feature directions.
        
        # If we're decomposing the token embeddings,
        # TODO - Rename to vocab_subspace.
        if config.vocab_subspace:

            # Create the embedding table. Vocabulary embeddings are learned
            # in a lower dimensional latent space.
            self.vocab_embed = nn.Embedding(
                config.vocab_size, # Number of tokens
                config.vocab_rank  # Subspace dimension
            )

            # Create a 
            # Selected token latents will be projected up to model size.
            # vocab_proj has shape [vocab_rank x model_size]
            self.vocab_proj = nn.Linear(
                config.vocab_rank,  # Size of latents
                config.hidden_size, # Model size
                bias=False
            )            
            
        # Otherwise, for a dense vocabulary,
        else:
            # Create the dense embedding table in model space.
            self.vocab_embed = nn.Embedding(
                config.vocab_size,  # Number of tokens
                config.hidden_size  # Model size
            )

            self.vocab_proj = None

        # =====================
        #   RoPE Embeddings
        # =====================
        
        # Pre-computes the table of RoPE embeddings, leaving them in 
        # GPU memory.
        self.rope = RotaryEmbedding(config)                  

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
        
        # Wrap in torch ModuleList
        self.layers = nn.ModuleList(layers)
        
        # Whatever huggingface does behind the scenes...
        self.post_init() 

    # Agents: Do not define boilerplate helpers, e.g., get/set_input_embeddings

    
    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Return token embeddings for input ids.
        This will perform the up projection to model space if the vocabulary is
        decomposed.
        
        input_ids have shape [batch_size, seq_len]
        """
            
        # If the vocabulary is decomposed,
        if self.vocab_proj is not None: 
            
            # Retrieve the latents
            #  input_ids: [batch_size, seq_len]
            #          x: [batch_size, seq_len, latent_dim] 
            x = self.vocab_embed(input_ids)

            #  Project the latents back to model space and return.
            return(self.vocab_proj(x))
        
        # If the vocabulary is dense,
        else:
            # Just return the embeddings.
            return self.vocab_embed(input_ids)

    # Comment--evaluates the model on (describe expected input shape) and returns (describe)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        # TODO - the attention mask is... "additive mask (0 for keep, −inf for pad)"
        # TODO - For coding style--don't use symbols unless it's a big, dense function.

        Inputs:
            input_ids       [batch_size, seq_len] 
            attention_mask  [batch_size,    1,    1, seq_len] 

        Returns:
            Final encoder layer output   [batch_size, seq_len, model_size] 
        """
        
        # Retrieve the token embeddings for this sequence.
        # These are model_size, regardless of whether the vocab is decompd.
        hidden_states = self.embed(input_ids)  
        
        # Retrieve the rotary position embeddings for all of the positions in
        # our current input sequence. 
        
        seq_len = hidden_states.size(1)
        
        # Retrieves just the ones necessary for the sequence length of the
        # input. These are vectors, two per token. Their length is the 
        # number of head dimensions we're applying RoPE to.
        #  Input
        #     cos: [max_seq_len, rope_dims] 
        #     sin: [max_seq_len, rope_dims]
        #  Outputs:
        #     R_cos [seq_len, rope_dims] 
        #     R_sin [seq_len, rope_dims] 
        R_cos = self.rope.cos[:seq_len] 
        R_sin = self.rope.sin[:seq_len]
        
        # Run the model!
        
        # For each encoder layer,
        for layer_i, layer in enumerate(self.layers):
            
            # Evaluate the layer
            hidden_states = layer(
                hidden_states,       # Token embeddings
                (R_cos, R_sin),      # Rope embeddings, passed as a tuple.
                attention_mask,      # Attn mask
                layer_i              # Layer index, for any layer-specific behavior.
            )

        # Return the final output of the encoder stack.
        return hidden_states
        

class SharedSubspaceEncoderForMaskedLM(SharedSubspaceEncoderPreTrainedModel):
    """
    The `*MaskedLM` object: 
        - Initializes:
            - A `*Model` object from the given config.
            - (It doesn't create a new LM head--we just use the vocabulary)
    """

    def __init__(self, config: SharedSubspaceEncoderConfig) -> None:
        
        # Call the `*PreTrainedModel` init.
        super().__init__(config)

        # Create the `*Model`. Everything we need is already there.
        self.encoder_model = SharedSubspaceEncoderModel(config)
        
        # Call the `*PreTrainedModel` init
        self.post_init()

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        The `labels` are token ids with the `[MASK]` token id at the masked 
        positions, and -100 (we tell the loss function this) everywhere else.
        The `attention_mask`...  
        
        Inputs:
               input_ids: [batch_size, seq_len]
          attention_mask: [batch_size, 1, 1, seq_len]
                 labels : [batch_size, seq_len]
         
        The `logits` are the prediction scores over the vocabulary.
        The `loss` is a scalar value... per sample in the batch? It has had
        the mask applied to it? cross-entropy. Only when labels provided, otherwise it's...

        Outputs:
           logits: [batch_size, seq_len, vocab_size]  Predction scores over vocab
             loss: [batch_size]? Or average over batch?
        """

        # Run the input through the whole model.
        hidden_states = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            **kwargs, # TODO - What can be passed here?
        )                      

        # The hidden states are model size. If the vocabulary was decomposed,
        # We need to down project, and then multiply with the vocabulary latents.
        # Otherwise, multiply directly with the vocabulary embeddings.
        # --- Shared projection → logits  -------------------
        
        if self.encoder.vocab_proj is not None:
            #  B - batch_size
            #  T - sequence length
            #  D - model_size
            #  C - latent_size
            #  V - vocab_size
            
            # Linear stores the transpose of its projection, so vocab_proj
            # is functionally [C x D], but stored as [D x C]      
            # So the vocabulary latent space projection, W_E_proj, is [D x C]
            W_E_proj = self.encoder.vocab_proj.weight
            
            # Project the tokens output by the model into the vocabulary 
            # subspace.
            #
            # Inputs:
            #    hidden_states   [B, T, D]
            #         W_E_proj         [D, C]            
            # Outputs:
            #        h_latents   [B, T, C]
            #
            #  TODO - Fuse with the next op if beneficial.
            h_latents = einsum('btd,dc->btc', hidden_states, W_E_proj)
            
            # Multiply each token latent with every vocabulary latent to 
            # get the per-token logit scores over the vocabulary.
            # TODO
            #
            # Inputs:
            # 
            # Outputs: 
            #    logits  [B, T, V]
            
            #  TODO - next step will be to flatten--do it here if it makes sense.
        
        # If there's no vocabulary subspace,
        else:
            # TODO - Multiply the hidden states with the vocabulary.
            #
            # Inputs:
            #    hidden_states   [B, T, D]
            # Outputs: 
            #    logits  [B, T, V]
            pass
            
              

        loss = None
        if labels is not None:
            # Flatten everything for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss






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

    def __init__(self, config: SharedSubspaceEncoderConfig) -> None:
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
                self.q_latent_dim + self.kv_latent_dim,
                bias=config.attention_bias,
            )

            # TODO - Decide whether to share or split.
            self.qkv_a_layernorm = DeepseekV3RMSNorm(
                self.q_latent_dim + self.kv_latent_dim,
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
            
        # ==========================
        #     Output Projections
        # ==========================

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
        #past_key_value: Optional[Cache] = None, # TODO - Can I remove this?
        #cache_position: Optional[torch.LongTensor] = None, # TODO - Can I remove this?
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
        Dc_q, Dc_kv = self.q_latent_dim, self.kv_latent_dim
        

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
                input_latents, [self.q_latent_dim, self.kv_latent_dim], dim=-1
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
