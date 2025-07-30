import torch
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.custom_bert import SubspaceBertForMaskedLM, SubspaceBertConfig
from models.layers.mla_attention import (
    DeepseekV3Attention,
    DeepseekV3Config,
    DeepseekV3RotaryEmbedding,
)


def test_custom_bert_forward():
    config = SubspaceBertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    # Use standard attention implementation
    config._attn_implementation = "eager"
    model = SubspaceBertForMaskedLM(config)
    # generate random input ids (batch_size=2, seq_len=8)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (2, 8, config.vocab_size)


def test_deepseek_attention_forward():
    ds_config = DeepseekV3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=16,
        qk_nope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=16,
        attention_dropout_prob=0.0,
        rms_norm_eps=1e-6,
    )
    ds_config._attn_implementation = "eager"
    attention = DeepseekV3Attention(ds_config, layer_idx=0)

    rotary = DeepseekV3RotaryEmbedding(ds_config)
    position_ids = torch.arange(0, 8).unsqueeze(0)
    dummy = torch.zeros(1, 1, ds_config.qk_rope_head_dim)
    cos, sin = rotary(dummy, position_ids)

    # random hidden states: (batch=2, seq_len=8, hidden_size)
    hidden_states = torch.randn(2, 8, ds_config.hidden_size)
    # full attention mask (batch=2, heads=1, seq=1, kv_seq=8)
    attn_mask = torch.ones(2, 1, 1, 8)
    out, _ = attention(hidden_states, (cos, sin), attn_mask)
    assert out.shape == (2, 8, ds_config.hidden_size)


def test_deepseek_attention_with_output_latent():
    ds_config = DeepseekV3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=16,
        qk_nope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=16,
        attention_dropout_prob=0.0,
        rms_norm_eps=1e-6,
        use_output_latent=True,
        o_lora_rank=32,
    )
    ds_config._attn_implementation = "eager"
    attention = DeepseekV3Attention(ds_config, layer_idx=0)

    assert hasattr(attention, "o_a_proj")
    assert hasattr(attention, "o_b_proj")

    rotary = DeepseekV3RotaryEmbedding(ds_config)
    position_ids = torch.arange(0, 8).unsqueeze(0)
    dummy = torch.zeros(1, 1, ds_config.qk_rope_head_dim)
    cos, sin = rotary(dummy, position_ids)

    hidden_states = torch.randn(2, 8, ds_config.hidden_size)
    attn_mask = torch.ones(2, 1, 1, 8)
    out, _ = attention(hidden_states, (cos, sin), attn_mask)
    assert out.shape == (2, 8, ds_config.hidden_size)

"""
We'll come back to this.
def test_deepseek_attention_flash():
    ds_config = DeepseekV3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=16,
        qk_nope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=16,
        attention_dropout_prob=0.0,
        rms_norm_eps=1e-6,
    )
    ds_config._attn_implementation = "flash_attention_2"
    attention = DeepseekV3Attention(ds_config, layer_idx=0)

    rotary = DeepseekV3RotaryEmbedding(ds_config)
    position_ids = torch.arange(0, 8).unsqueeze(0)
    dummy = torch.zeros(1, 1, ds_config.qk_rope_head_dim)
    cos, sin = rotary(dummy, position_ids)

    hidden_states = torch.randn(2, 8, ds_config.hidden_size)
    attn_mask = torch.ones(2, 1, 1, 8)
    out, _ = attention(hidden_states, (cos, sin), attn_mask)
    assert out.shape == (2, 8, ds_config.hidden_size)
"""

def test_custom_bert_with_mla():
    config = SubspaceBertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    config._attn_implementation = "eager" # Allows for manual implementation.
    config.use_mla = True # Use this to choose MLA instead.
    config.add_output_latent = False
    model = SubspaceBertForMaskedLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (2, 8, config.vocab_size)


def test_custom_bert_with_mla_output_latent():
    config = SubspaceBertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    config._attn_implementation = "eager" # Allows for manual implementation.
    config.use_mla = True # Use this to choose MLA instead.
    config.add_output_latent = True
    model = SubspaceBertForMaskedLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (2, 8, config.vocab_size)


def test_mla_with_dense_prefix_layers():
    """Ensure dense prefix layers fall back to standard MHA."""
    config = SubspaceBertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=64,
        use_mla=True,
        num_dense_layers=2,
    )
    config._attn_implementation = "eager"
    model = SubspaceBertForMaskedLM(config)

    # First layer should use standard attention
    assert not isinstance(
        model.bert.encoder.layer[0].attention.self, DeepseekV3Attention
    )
    # Subsequent layers should use MLA
    assert isinstance(
        model.bert.encoder.layer[2].attention.self, DeepseekV3Attention
    )


def test_decomposed_ffn():
    """Ensure decomposed FFN modules can replace dense ones."""
    config = SubspaceBertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=64,
        use_decomp_mlp=True,
        ffn_rank=16,
        num_dense_layers=2
    )
    config._attn_implementation = "eager"
    model = SubspaceBertForMaskedLM(config)

    # TODO - Confirm that the first two layers are still dense, e.g.
    #assert model.bert.encoder.layer[0].intermediate.__class__.__name__ == "?"
    #assert model.bert.encoder.layer[1].intermediate.__class__.__name__ == "?"
    
    assert hasattr(model.bert.encoder.layer[4], "intermediate")
    assert model.bert.encoder.layer[4].intermediate.__class__.__name__ == "BertIntermediateDecomp"


    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (2, 8, config.vocab_size)


if __name__ == "__main__":
    print("Testing standard BERT forward")
    test_custom_bert_forward()
    print("Testing MLA attention forward")
    test_deepseek_attention_forward()
    print("Testing MLA with output latent")
    test_deepseek_attention_with_output_latent()
    #print("Testing MLA flash attention")
    #test_deepseek_attention_flash()
    print("Testing BERT with MLA")
    test_custom_bert_with_mla()
    print("Testing BERT with MLA and output latent")
    test_custom_bert_with_mla_output_latent()
    print("Testing with mixed MHA and MLA layers")
    test_mla_with_dense_prefix_layers()
    print("Testing Decomposed FFN")
    test_decomposed_ffn()

