import torch
import pytest
import sys
import json
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.shared_subspace_encoder import (
    SharedSubspaceEncoderConfig,
    SharedSubspaceEncoderModel,
    MultiheadLatentAttention,
    SharedSubspaceEncoderLayer,
    DeepseekV3RMSNorm
)


# Load the baseline config file.    
with open('test_config.json') as f:
    config = json.load(f)

# Strict key check on the model configuration.
valid_keys = SharedSubspaceEncoderConfig.__init__.__code__.co_varnames
valid_keys = set(valid_keys) - {"self", "kwargs"}
extra_keys = set(config["model"]) - valid_keys
if extra_keys:
    raise ValueError(f"Unknown keys in config: {sorted(extra_keys)}")

# Will raise TypeError, by design, if required args are missing
model_cfg = SharedSubspaceEncoderConfig(**config["model"])

def make_config(**overrides):
    cfg = copy.deepcopy(model_cfg)
    cfg.attention_backend = "eager"
    cfg._attn_implementation = "eager"
    cfg.output_subspace = False
    cfg.vocab_subspace = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

def test_config_defaults():
    cfg = SharedSubspaceEncoderConfig()
    assert cfg.vocab_size == 30522
    assert cfg.hidden_size == 768
    assert cfg.is_decoder is False


def test_model_initialization():
    cfg = make_config()
    model = SharedSubspaceEncoderModel(cfg)
    assert model.vocab_embed.weight.shape == (cfg.vocab_size, cfg.hidden_size)
    assert len(model.layers) == cfg.num_hidden_layers


def test_forward_not_implemented():
    cfg = make_config()
    model = SharedSubspaceEncoderModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(NotImplementedError):
        model(input_ids)


def test_mla_init_with_output_latent():
    cfg = make_config(output_subspace=True)
    attn = MultiheadLatentAttention(cfg, layer_idx=0)
    assert attn.o_a_proj.weight.shape == (cfg.o_latent_dim, cfg.num_attention_heads * cfg.v_head_dim)
    assert attn.o_b_proj.weight.shape == (cfg.hidden_size, cfg.o_latent_dim)


def test_layer_initialization_dense():
    cfg = make_config()
    layer = SharedSubspaceEncoderLayer(cfg, layer_idx=0)

    # attention block
    assert isinstance(layer.self_attn, MultiheadLatentAttention)
    assert layer.attn_dropout.p == cfg.hidden_dropout_prob
    assert isinstance(layer.attn_norm, torch.nn.LayerNorm)

    # dense attention uses a single output projection
    assert hasattr(layer.self_attn, "o_proj")
    assert not hasattr(layer.self_attn, "o_a_proj")

    # dense FFN should not define shared weights
    assert not hasattr(layer.mlp, "W_in_shared")


def test_layer_with_subspaces():
    cfg = make_config(output_subspace=True, ffn_decompose=True, ffn_rank=4)
    layer = SharedSubspaceEncoderLayer(cfg, layer_idx=1)

    # output subspace creates the two projection matrices
    assert hasattr(layer.self_attn, "o_a_proj")
    assert hasattr(layer.self_attn, "o_b_proj")
    assert not hasattr(layer.self_attn, "o_proj")

    # decomposed FFN creates shared weights
    assert hasattr(layer.mlp, "W_in_shared")
    assert layer.mlp.W_in_shared.weight.shape == (cfg.ffn_rank, cfg.hidden_size)

