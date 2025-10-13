"""# `shared_space_config.py`

Configuration utilities for the Sparse MoE decoder. This file mirrors the
structure of `subspace_decoder/models/shared_space_config.py` so downstream
code can toggle between dense and sparse feed-forward stacks without touching
the higher-level training scripts.
"""

from __future__ import annotations

import json
from typing import Optional

from transformers.configuration_utils import PretrainedConfig


def make_shorthand(model_cfg: "SparseMoEDecoderConfig") -> str:
    """
    Construct a short descriptive string for runs.

    We extend the original helper to include MoE-related knobs so experiment
    names immediately communicate how many experts are active and the router
    configuration.
    """
    dense_str = f"{model_cfg.num_dense_layers}mha + "

    if model_cfg.o_shared_dim is not None:
        o_str = f".{model_cfg.o_shared_dim}"
    else:
        o_str = ""

    attn_str = (
        dense_str
        + "mla."
        + str(model_cfg.q_shared_dim)
        + "."
        + str(model_cfg.kv_shared_dim)
        + o_str
    )

    # MLP / MoE configuration summary.
    if model_cfg.use_sparse_moe and model_cfg.num_experts > 1:
        moe_piece = (
            f"moe.k{model_cfg.router_top_k}."
            f"e{model_cfg.num_experts}."
            f"c{model_cfg.capacity_factor}"
        )
    else:
        moe_piece = "mlp"

    if model_cfg.ffn_decompose:
        dense_mlp = (
            f"{model_cfg.num_dense_layers}mlp."
            f"{model_cfg.intermediate_size} + "
        )
        mlp_piece = (
            dense_mlp
            + f"{model_cfg.num_hidden_layers - model_cfg.num_dense_layers}"
            + "dcmp."
            + f"x{model_cfg.intermediate_size}."
            + str(model_cfg.ffn_rank)
        )
    else:
        mlp_piece = f"mlp.{model_cfg.intermediate_size}"

    shorthand = (
        f"{attn_str} - {mlp_piece} - {moe_piece} - "
        f"h{model_cfg.hidden_size} - l{model_cfg.num_hidden_layers}"
    )

    return shorthand


class SparseMoEDecoderConfig(PretrainedConfig):
    r"""
    Configuration class for the Sparse MoE decoder.

    The base parameters mirror `SharedSpaceDecoderConfig` so existing configs
    continue to load, while additional attributes describe the mixture-of-
    experts routing strategy.

    ----------------------
    Core Model Parameters:
    ----------------------
    Refer to the original documentation for details on the dense components.

    ----------------------
    Sparse MoE Parameters:
    ----------------------
    - use_sparse_moe (`bool`): Master switch for activating the MoE block.
    - num_experts (`int`): Number of experts in the mixture.
    - router_top_k (`int`): How many experts each token can route to.
    - capacity_factor (`float`): Scaling factor controlling expert capacity.
    - router_noise_std (`float`): Stddev of Gaussian noise added during routing.
    """

    model_type = "sparse_subspace_decoder"

    def __init__(
        self,

        # === Core Model ===
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        rms_norm_eps: float = 1e-6,
        norm_type: str = "layernorm",
        classifier_dropout: Optional[float] = None,

        vocab_subspace: bool = False,
        vocab_rank: Optional[int] = None,
        tie_word_embeddings: bool = True,

        # === MLA ===
        num_attention_heads: int = 16,
        rope_dims: int = 16,
        q_shared_dim: Optional[int] = None,
        kv_shared_dim: Optional[int] = None,
        o_shared_dim: Optional[int] = None,
        qk_private_dim: Optional[int] = None,
        vo_private_dim: Optional[int] = None,
        nope_dims: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        num_dense_layers: int = 12,
        attention_backend: str = "eager",

        # === Decomposed FFN ===
        ffn_decompose: bool = False,
        ffn_rank: Optional[int] = None,

        # === Sparse MoE ===
        use_sparse_moe: bool = True,
        num_experts: int = 4,
        router_top_k: int = 2,
        capacity_factor: float = 1.25,
        router_noise_std: float = 1.0,
        expert_dropout_prob: float = 0.0,
        **kwargs,
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
        self.norm_type = norm_type
        self.classifier_dropout = classifier_dropout

        self.vocab_subspace = vocab_subspace
        self.vocab_rank = vocab_rank
        self.tie_word_embeddings = tie_word_embeddings

        # === MLA ===
        self.num_attention_heads = num_attention_heads
        self.rope_dims = rope_dims
        self.q_shared_dim = q_shared_dim
        self.kv_shared_dim = kv_shared_dim
        self.o_shared_dim = o_shared_dim
        self.qk_private_dim = qk_private_dim
        self.vo_private_dim = vo_private_dim
        self.nope_dims = nope_dims
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.num_dense_layers = num_dense_layers
        self.attention_backend = attention_backend

        # === Decomposed FFN ===
        self.ffn_decompose = ffn_decompose
        self.ffn_rank = ffn_rank

        # === Sparse MoE ===
        self.use_sparse_moe = use_sparse_moe
        self.num_experts = num_experts
        self.router_top_k = router_top_k
        self.capacity_factor = capacity_factor
        self.router_noise_std = router_noise_std
        self.expert_dropout_prob = expert_dropout_prob

        # === Validation ===
        self._validate()

    # --------------------------------------------------------------------- #
    #                         Convenience Helpers                           #
    # --------------------------------------------------------------------- #
    @property
    def uses_sparse_experts(self) -> bool:
        """
        Quick predicate summarizing whether the sparse MoE stack should be
        active in downstream modules.
        """
        return self.use_sparse_moe and self.num_experts > 1

    def compute_capacity(self, batch_tokens: int) -> int:
        """
        Compute per-expert capacity given a batch size (batch_size * seq_len).

        Args:
            batch_tokens: Number of token positions in the batch.

        Returns:
            Integer capacity per expert.
        """
        total_slots = batch_tokens * self.router_top_k
        per_expert = total_slots / max(self.num_experts, 1)
        return max(1, int(per_expert * self.capacity_factor))

    # --------------------------------------------------------------------- #
    #                              Validation                               #
    # --------------------------------------------------------------------- #
    def _validate(self) -> None:
        if self.num_dense_layers > self.num_hidden_layers:
            raise ValueError("`num_dense_layers` must be <= `num_hidden_layers`")
        if self.vocab_subspace and self.vocab_rank is None:
            raise ValueError("`vocab_rank` must be set when `vocab_subspace=True`")

        # TODO - This check seems too strict, disable for now.
        # if (
        #     self.num_dense_layers < self.num_hidden_layers
        #     and self.q_shared_dim is None
        #     and self.kv_shared_dim is None
        # ):
        #     raise ValueError(
        #         "At least one of q_shared_dim or kv_shared_dim must be set when "
        #         "there are subspace layers"
        #     )

        if self.qk_private_dim is None or self.vo_private_dim is None:
            raise ValueError("Must set qk_private_dim and vo_private_dim")
        if self.nope_dims is None:
            raise ValueError("Must set nope_dims")

        if self.ffn_decompose and self.ffn_rank is None:
            raise ValueError("`ffn_rank` must be set when `ffn_decompose=True`")
        if self.ffn_decompose and self.num_dense_layers >= self.num_hidden_layers:
            raise ValueError(
                "`ffn_decompose` was set but `num_dense_layers` is >= number of layers"
            )

        valid_backends = ["eager", "flash_attention_2", "sdpa"]
        if self.attention_backend not in valid_backends:
            raise ValueError(
                f"Unknown attention backend: {self.attention_backend}, options are {valid_backends}"
            )

        valid_norm_types = ["layernorm", "rmsnorm"]
        if self.norm_type not in valid_norm_types:
            raise ValueError(
                f"Unknown norm type: {self.norm_type}, options are {valid_norm_types}"
            )

        if self.router_top_k < 1:
            raise ValueError("`router_top_k` must be >= 1")
        if self.num_experts < 1:
            raise ValueError("`num_experts` must be >= 1")
        if self.capacity_factor <= 0:
            raise ValueError("`capacity_factor` must be > 0")
        if self.router_noise_std < 0:
            raise ValueError("`router_noise_std` must be >= 0")
        if not 0.0 <= self.expert_dropout_prob <= 1.0:
            raise ValueError("`expert_dropout_prob` must be in [0, 1]")

    # how to pretty-print the SparseMoEDecoderConfig using json.dumps
    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

def get_config(filename: str):
    """
    Load a JSON config file and instantiate `SparseMoEDecoderConfig`.
    """
    with open(filename) as handle:
        full_cfg = json.load(handle)

    valid_keys = SparseMoEDecoderConfig.__init__.__code__.co_varnames
    valid_keys = set(valid_keys) - {"self", "kwargs"}

    extra_keys = set(full_cfg["model"]) - valid_keys
    missing_keys = valid_keys - set(full_cfg["model"])

    if extra_keys:
        raise ValueError(f"Unknown keys in config: {sorted(extra_keys)}")
    if missing_keys:
        raise ValueError(f"config json is missing: {sorted(missing_keys)}")

    model_cfg = SparseMoEDecoderConfig(**full_cfg["model"])

    return full_cfg, model_cfg
