"""# `linkspace_config.py`

Configuration for LinkedSpaceDecoder model with flexible space-to-module mappings.

This config allows arbitrary mappings of modules to shared subspaces, enabling
experimentation with different architectural configurations.
"""

from typing import Optional, Dict, List, Any

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


def make_shorthand(model_cfg):
    """
    Takes an instance of LinkedSpaceDecoderConfig and constructs a shorthand
    name for the model based on settings.
    """
    
    # Build a string representation of the spaces configuration
    space_strs = []
    for space_id, space_config in model_cfg.spaces.items():
        size = space_config['size']
        modules = ','.join(space_config['modules'])
        space_strs.append(f"sp{space_id}[{size}:{modules}]")
    
    spaces_str = " + ".join(space_strs)
    
    # Assemble string
    shorthand = (
        f"linkspace - {spaces_str} - "
        f"h{model_cfg.hidden_size} - l{model_cfg.num_hidden_layers}"
    )
    
    return shorthand


class LinkedSpaceDecoderConfig(PretrainedConfig):
    r"""
    Configuration class for LinkedSpaceDecoder.

    Extends the HuggingFace `PretrainedConfig` to support flexible space-to-module
    mappings. Instead of fixed projections for Q, K, V, O and FFN, this config allows
    arbitrary grouping of modules into shared subspaces.

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

    - rms_norm_eps (`float`) — Epsilon for RMSNorm (all norms are RMSNorm)

    - classifier_dropout (`float` or None) — Dropout for final classifier.

    - vocab_subspace (`bool`) — Whether to decompose vocabulary embeddings
    - vocab_rank (`int`) — Rank of vocabulary subspace

    ----------------------------------
    LinkedSpace Architecture:
    ----------------------------------
    - spaces (`dict`) — Dictionary mapping space IDs to space configurations.
      Each space config has:
        - size (`int`) — Dimension of the shared subspace
        - modules (`list`) — List of module names using this space
          Valid module names:
            Attention: "Q", "K", "V", "O"
            FFN: "in", "gate", "out"
      
      All spaces use RMSNorm normalization (always enabled).
      
      Example:
        spaces = {
            0: {"size": 768, "modules": ["K", "V", "Q", "in", "gate", "out"]},
            1: {"size": 256, "modules": ["O"]}
        }

    - num_attention_heads (`int`) — Number of attention heads.
    - qk_private_dim (`int`) — Query/key private dimension per head.
    - vo_private_dim (`int`) — Value/output private dimension per head.

    - rope_dims (`int`) — Number of head dimensions carrying RoPE.
    - nope_dims (`int`) — Non-positional encoding dimensions.
    - rope_theta (`float`) — Base frequency used for RoPE.
    - rope_scaling (`dict` or None) — HF-style scaling dict for RoPE.
    - attention_bias (`bool`) — Whether to include bias terms in projections.
    
    - num_dense_layers (`int`) — Number of leading layers that do not use
                                 subspaces for attention or FFNs.
    - attention_backend (`str`) — Must be one of `"eager"`, `"flash_attention_2"`, or `"sdpa"`.
    """

    model_type = "linkspace_decoder"

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
        rms_norm_eps=1e-6,
        classifier_dropout=None,

        vocab_subspace=False,
        vocab_rank=None,
        tie_word_embeddings=True,

        # === LinkedSpace Configuration ===
        spaces: Optional[Dict[int, Dict[str, Any]]] = None,

        # === Attention Parameters ===
        num_attention_heads: int = 16,
        rope_dims:           int = 16,

        # Private head dimensions
        qk_private_dim:      int = None,
        vo_private_dim:      int = None,
        nope_dims:           int = None,

        attention_backend="eager",
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,

        # === Layer Composition ===
        num_dense_layers=12,  # dense MHA layers before linkspace starts

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
        self.rms_norm_eps = rms_norm_eps
        self.classifier_dropout = classifier_dropout

        self.vocab_subspace = vocab_subspace
        self.vocab_rank = vocab_rank
        self.tie_word_embeddings = tie_word_embeddings

        # === LinkedSpace Configuration ===
        # If no spaces provided, default to a simple configuration
        if spaces is None:
            spaces = {
                0: {"size": hidden_size, "modules": ["Q", "K", "V", "O", "in", "gate", "out"]}
            }
        self.spaces = spaces

        # === Attention ===
        self.num_attention_heads = num_attention_heads
        self.rope_dims = rope_dims

        # Private head dimensions
        self.qk_private_dim = qk_private_dim
        self.vo_private_dim = vo_private_dim
        self.nope_dims = nope_dims
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.num_dense_layers = num_dense_layers

        # === Attention backend ===
        self.attention_backend = attention_backend

        # === Validation ===
        self._validate()

        #print(f"  > LinkedSpace *Config.init: {make_shorthand(self)}\n")

    def _validate(self):
        """Validate the configuration."""
        
        # === Model ===
        if self.num_dense_layers > self.num_hidden_layers:
            raise ValueError("`num_dense_layers` must be <= `num_hidden_layers`")
        if self.vocab_subspace and self.vocab_rank is None:
            raise ValueError("`vocab_rank` must be set when `vocab_subspace=True`")

        # === LinkedSpace Validation ===
        valid_modules = {"Q", "K", "V", "O", "in", "gate", "out"}
        
        # Check that each space has required keys
        for space_id, space_config in self.spaces.items():
            if not isinstance(space_config, dict):
                raise ValueError(f"Space {space_id} config must be a dictionary")
            
            if 'size' not in space_config:
                raise ValueError(f"Space {space_id} must have 'size' key")
            if 'modules' not in space_config:
                raise ValueError(f"Space {space_id} must have 'modules' key")
            
            # Validate module names
            for module in space_config['modules']:
                if module not in valid_modules:
                    raise ValueError(
                        f"Invalid module '{module}' in space {space_id}. "
                        f"Valid modules: {valid_modules}"
                    )
        
        # Check that each module is assigned to exactly one space
        module_assignments = {}
        for space_id, space_config in self.spaces.items():
            for module in space_config['modules']:
                if module in module_assignments:
                    raise ValueError(
                        f"Module '{module}' is assigned to multiple spaces: "
                        f"{module_assignments[module]} and {space_id}"
                    )
                module_assignments[module] = space_id
        
        # Validate that private dimensions are set
        if self.qk_private_dim is None or self.vo_private_dim is None:
            raise ValueError("Must set qk_private_dim and vo_private_dim")
        if self.nope_dims is None:
            raise ValueError("Must set nope_dims")

        # === Attention Backend ===
        valid_backends = ["eager", "flash_attention_2", "sdpa"]
        if self.attention_backend not in valid_backends:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}, options are {valid_backends}")

    def get_space_for_module(self, module: str) -> Optional[int]:
        """
        Get the space ID that a given module is assigned to.
        
        Args:
            module: Module name (e.g., "Q", "K", "V", "O", "in", "gate", "out")
            
        Returns:
            Space ID if module is in a space, None otherwise
        """
        for space_id, space_config in self.spaces.items():
            if module in space_config['modules']:
                return space_id
        return None


#### `get_config`

import json

def get_config(filename):
    """Load configuration from a JSON file."""

    # Load the config file.
    with open(filename) as f:
        full_cfg = json.load(f)

    # Strict key check on the model configuration.

    # Get the list of keys allowed / required by `*Config`
    valid_keys = LinkedSpaceDecoderConfig.__init__.__code__.co_varnames
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
    model_cfg = LinkedSpaceDecoderConfig(**full_cfg["model"])

    return full_cfg, model_cfg

