import torch
import pytest
import sys
import builtins
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The `MultiheadLatentAttention` stub references annotation types that aren't
# included in the skeleton.  Define simple placeholders so the module can be
# imported without raising errors during evaluation of those annotations.
builtins.Cache = type("Cache", (), {})
def _dummy_norm(*args, **kwargs):
    return torch.nn.Identity()

builtins.DeepseekV3RMSNorm = _dummy_norm

from models.shared_subspace_encoder import (
    SharedSubspaceEncoderConfig,
    SharedSubspaceEncoderModel,
    MultiheadLatentAttention,
    SharedSubspaceEncoderLayer,
)

# Clean up the temporary placeholders


def make_config(output_subspace=False, **overrides):
    cfg = SharedSubspaceEncoderConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        head_dim=8,
        q_latent_dim=8,
        kv_latent_dim=8,
        v_head_dim=8,
        attention_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        num_dense_layers=0,
        attention_bias=False,
        rope_theta=10000.0,
        rope_dims=4,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        eps=1e-5,
        initializer_range=0.02,
        output_subspace=output_subspace,
        o_latent_dim=16,
        **overrides,
    )

    # Manually apply overrides for attributes not handled by the config
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

