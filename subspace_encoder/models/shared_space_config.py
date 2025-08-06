"""# `shared_space_config.py`

#### `*Config`
"""

from typing import Optional

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

"""`def make_shorthand`"""

def make_shorthand(model_cfg):
    """
    Takes an instance subencoder `*Config` and constructs a shorthand
    name for the model based on settings.
    """

    dense_str = str(model_cfg.num_dense_layers) + "mha + "

    if model_cfg.output_subspace:
        o_str = "." + str(model_cfg.o_latent_dim)
    else:
        o_str = ""

    # If no output subspace is used, the dimension will show as -1.
    attn_str = (
        dense_str
        + "mla."
        + str(model_cfg.q_latent_dim)
        + "."
        + str(model_cfg.kv_latent_dim)
        + o_str
    )

    # MLP Configuration
    if model_cfg.ffn_decompose:
        dense_str = (
            str(model_cfg.num_dense_layers)
            + "mlp."
            + str(model_cfg.intermediate_size)
            + " + "
        )

        mlp_str = (
            dense_str
            + str(model_cfg.num_hidden_layers - model_cfg.num_dense_layers)
            + "dcmp."
            + "x"
            + str(model_cfg.intermediate_size)
            + "."
            + str(model_cfg.ffn_rank)
        )
    else:
        mlp_str = "mlp." + str(model_cfg.intermediate_size)

    # Assemble string
    shorthand = (
        f"{attn_str} - {mlp_str} - "
        f"h{model_cfg.hidden_size} - l{model_cfg.num_hidden_layers}"
    )

    """
    The run name includes training settings

    run_name = (
        f"{config['stats']['total_elements']} - "
        f"{attn_str} - {mlp_str} - "
        f"h{model_cfg.hidden_size} - l{model_cfg.num_hidden_layers} - "
        f"bs{ptrain_cfg['train_batch_size']} - lr{lr_str} - "
        f"seq{ptrain_cfg['max_seq_length']}"
    )
    """

    return shorthand


class SharedSpaceEncoderConfig(PretrainedConfig):
    r"""
    Configuration class for SharedSpaceEncoderConfig.

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
    Core Model Parameters:
    ----------------------
    - vocab_size (`int`) — Vocabulary size.
    - hidden_size (`int`) — Model hidden dimension.
    - num_hidden_layers (`int`) — Number of transformer blocks.
    - intermediate_size (`int`) — Feed-forward hidden dimension.
    - hidden_act (`str`) — Activation function.
    - hidden_dropout_prob (`float`) — Dropout after projections and FFNs.
    - attention_dropout_prob (`float`) — Dropout applied to attention scores.
    - max_position_embeddings (`int`) — Max sequence length.
    - initializer_range (`float`) — Stddev of weight init.

    TODO - Decide on norm
    - layer_norm_eps (`float`) — Epsilon for LayerNorm.
    - rms_norm_ps (`float`) — Epsilon for RMSNorm

    - classifier_dropout (`float` or None) — Dropout for final classifier.


    - vocab_subspace
    - vocab_rank

    ----------------------------------
    Multi-Head Latent Attention (MLA):
    ----------------------------------
    - num_attention_heads (`int`) — Number of attention heads.
    - head_dim (`int`) — Head dimension.
    - rope_dims (`int`) — Number of head dimensions carrying RoPE.

    - q_latent_dim (`int`) — Rank of the shared query subspace.
    - kv_latent_dim (`int`) — Rank of the shared key/value subspace.

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
        vocab_size:         int = 30522,
        hidden_size:        int = 512,
        num_hidden_layers:  int = 12,

        intermediate_size:  int = 3072,

        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        max_position_embeddings: int = 2048,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        rms_norm_eps=1e-6, # Their default, but confirm in config.
        classifier_dropout=None,

        vocab_subspace=False,
        vocab_rank=None,

        # === Multi-Head Latent Attention ===
        num_attention_heads: int = 16,
        head_dim:            int = 32,       # 16*32 = 512,
        rope_dims:           int = 16,

        q_latent_dim:        int = None,
        kv_latent_dim:       int = None,

        output_subspace=False, # Currently an experiment.
        o_latent_dim=None,

        attention_backend="eager",
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,

        # === MLA Composition ===
        num_dense_layers=12,  # dense MHA layers before MLA starts

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
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.classifier_dropout = classifier_dropout

        self.vocab_subspace = vocab_subspace
        self.vocab_rank = vocab_rank

        # === MLA ===
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.rope_dims = rope_dims

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
        # TODO - Somewhere during training these get instantiated with bad
        #        values...
        #self._validate()

        #print(f"  > SubEnc *Config.init: {make_shorthand(self)}\n")


    def _validate(self):
        # === Model ===
        if self.num_dense_layers > self.num_hidden_layers:
            raise ValueError("`num_dense_layers` must be <= `num_hidden_layers`")
        if self.vocab_subspace and self.vocab_rank is None:
            raise ValueError("`vocab_rank` must be set when `vocab_subspace=True`")

        # === MLA Validation ===
        if self.num_dense_layers < self.num_hidden_layers and self.q_latent_dim is None or self.kv_latent_dim is None:
            raise ValueError("Must set sizes for latents when there are subspace layers")
        if self.output_subspace and self.o_latent_dim is None:
            raise ValueError("`o_latent_dim` must be set when `output_subspace=True`")

        # === Decomposed FFN ===
        if self.ffn_decompose and self.ffn_rank is None:
            raise ValueError("`ffn_rank` must be set when `ffn_decompose=True`")
        if self.ffn_decompose and self.num_dense_layers >= self.num_hidden_layers:
            raise ValueError("`ffn_decompose` was set but `num_dense` is >= number of layers")

        # === Attention Backend ===
        valid_backends = ["eager", "flash_attention_2", "sdpa"]
        if self.attention_backend not in valid_backends:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}, options are {valid_backends}")

#### `get_config`

import json

def get_config(filename):

    # Load the config file.
    with open(filename) as f:
        full_cfg = json.load(f)

    # Strict key check on the model configuration.

    # Get the list of keys allowed / required by `*Config`
    valid_keys = SharedSpaceEncoderConfig.__init__.__code__.co_varnames
    # Remove `self` and `kwargs`
    valid_keys = set(valid_keys) - {"self", "kwargs"}

    # Compare the set of keys in the json file vs `*Config`
    extra_keys = set(full_cfg["model"]) - valid_keys
    missing_keys = valid_keys - set(full_cfg["model"])

    # If there any in the `json` that aren't in `*Config`,
    if extra_keys:
        # List them for the user.
        raise ValueError(f"Unknown keys in config: {sorted(extra_keys)}")

    #  If the json config is missing required keys,
    if missing_keys:
        # List them for the user.
        raise ValueError(f"config json is missing: {sorted(missing_keys)}")

    # Will raise TypeError, by design, if required args are missing
    # The asterisks unpack the dictionary into a list of keywords as though
    # all of the settings were writting out individually.
    model_cfg = SharedSpaceEncoderConfig(**full_cfg["model"])

    return full_cfg, model_cfg
