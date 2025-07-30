from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

class SubspaceBertConfig(PretrainedConfig):
    r"""
    Configuration class for SubspaceBERT.

    Extends the HuggingFace `PretrainedConfig` to support architectural variations including:
    - Multi-Head Latent Attention (MLA)
    - Decomposed MLPs (low-rank FFNs)
    - Flexible attention backends (eager, flash, sdpa)
    - Explicit shared subspaces for Q, K, V, and O projections

    This config does not infer any defaults based on `hidden_size`. All dimensions and ranks must be explicitly specified. 
    If required values are missing, a `ValueError` is raised during initialization.

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
    - type_vocab_size (`int`) — Size of `token_type_ids` embedding.
    - initializer_range (`float`) — Stddev of weight init.
    - layer_norm_eps (`float`) — Epsilon for LayerNorm.
    - pad_token_id (`int`) — ID of the padding token.
    - position_embedding_type (`str`) — "absolute", "relative_key", or "relative_key_query".
    - use_cache (`bool`) — Whether to use KV cache (relevant for decoding).
    - classifier_dropout (`float` or None) — Dropout for final classifier.

    ----------------------
    Multi-Head Latent Attention (MLA):
    ----------------------
    - use_mla (`bool`) — Whether to enable MLA.
    - q_lora_rank (`int`) — Rank of the shared query subspace.
    - kv_lora_rank (`int`) — Rank of the shared key/value subspace.
    - qk_rope_head_dim (`int`) — Dimensionality of the rotary-position-encoded (RoPE) head subspace.
    - qk_nope_head_dim (`int`) — Dimensionality of the non-RoPE part of query/key heads.
    - v_head_dim (`int`) — Per-head value dimensionality.
    - output_subspace (`bool`) — Whether to use a shared latent subspace for output projections.
    - o_lora_rank (`int`) — Rank of the shared output subspace (required if `output_subspace=True`).
    - rope_theta (`float`) — Base frequency used for RoPE.
    - rope_scaling (`dict` or None) — HF-style scaling dict for RoPE.
    - rope_interleave (`bool`) — Whether to interleave RoPE dimensions.
    - attention_bias (`bool`) — Whether to include bias terms in Q/K/V projections.
    - num_dense_layers (`int`) — Number of leading layers that use dense MHA instead of MLA.
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
    - MLA is enabled and any required rank/dim fields are unset.
    - FFN decomposition is enabled without specifying `ffn_rank`.
    - An unknown `attention_backend` is provided.
    """
  
    
    model_type = "bert"

    def __init__(
        self,
        # === Core BERT ===
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,

        # === Multi-Head Latent Attention ===
        use_mla=False,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        qk_nope_head_dim=0,
        v_head_dim=None,
        output_subspace=False,
        o_lora_rank=None,
        attention_backend="eager",
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=False,
        attention_bias=False,

        # === MLA Composition ===
        num_dense_layers=0,  # dense MHA layers before MLA starts

        # === Decomposed MLP ===
        ffn_decompose=False,
        ffn_rank=None,

        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # === Core BERT ===
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        # === MLA ===
        self.use_mla = use_mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.output_subspace = output_subspace
        self.o_lora_rank = o_lora_rank
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias
        self.num_dense_layers = num_dense_layers

        # === Decomposed FFN ===
        self.ffn_decompose = ffn_decompose
        self.ffn_rank = ffn_rank

        # === Attention backend ===
        self.attention_backend = attention_backend
        if attention_backend == "flash":
            self._attn_implementation = "flash_attention_2"
        elif attention_backend == "sdpa":
            self._attn_implementation = "sdpa"
        elif attention_backend == "eager":
            self._attn_implementation = "eager"
        else:
            raise ValueError(f"Unknown attention backend: {attention_backend}")

        # === Validation ===
        self._validate()

        print(
            f"  > SubspaceBertConfig.init - {self.num_hidden_layers}l - mla{self.use_mla} - ndense{self.num_dense_layers} - dcmp{self.ffn_decompose}\n"
        )
        print(f"    - attention backend: {self.attention_backend}\n")

    def _validate(self):
        # === MLA Validation ===
        if self.use_mla:
            if self.q_lora_rank is None:
                raise ValueError("`q_lora_rank` must be set when `use_mla=True`")
            if self.kv_lora_rank is None:
                raise ValueError("`kv_lora_rank` must be set when `use_mla=True`")
            if self.qk_rope_head_dim is None:
                raise ValueError("`qk_rope_head_dim` must be set when `use_mla=True`")
            if self.v_head_dim is None:
                raise ValueError("`v_head_dim` must be set when `use_mla=True`")
            if self.output_subspace and self.o_lora_rank is None:
                raise ValueError("`o_lora_rank` must be set when `output_subspace=True`")

        # === Decomposed FFN ===
        if self.ffn_decompose and self.ffn_rank is None:
            raise ValueError("`ffn_rank` must be set when `ffn_decompose=True`")
