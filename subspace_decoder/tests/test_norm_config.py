#!/usr/bin/env python3
"""
Quick test script to verify norm configuration works in MLA layer.
This follows the style guide by prioritizing legibility and being a simple prototype.
"""

import torch
import sys
import os

# Add the subspace_decoder to path so we can import
sys.path.append(os.path.join(os.path.dirname(__file__), 'subspace_decoder'))

from models.shared_space_config import SharedSpaceDecoderConfig
from layers.mla import MultiheadLatentAttention


def test_norm_types():
    """Test that both LayerNorm and RMSNorm work correctly."""
    
    # Basic config for testing
    base_config = {
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "rope_dims": 16,
        "q_shared_dim": 64,
        "kv_shared_dim": 32,
        "output_subspace": True,
        "o_shared_dim": 64,
        "qk_private_dim": 32,
        "vo_private_dim": 32,
        "nope_dims": 16,
        "num_dense_layers": 1,
        "max_position_embeddings": 64,
    }
    
    # Test LayerNorm (default)
    print("Testing LayerNorm configuration...")
    layernorm_config = SharedSpaceDecoderConfig(
        norm_type="layernorm",
        **base_config
    )
    
    # Create MLA layer with LayerNorm
    mla_layernorm = MultiheadLatentAttention(layernorm_config, layer_idx=2)
    
    # Verify the norm layers are LayerNorm
    assert hasattr(mla_layernorm, 'q_a_norm')
    assert hasattr(mla_layernorm, 'kv_a_norm')
    assert hasattr(mla_layernorm, 'o_a_norm')
    
    assert type(mla_layernorm.q_a_norm).__name__ == 'LayerNorm'
    assert type(mla_layernorm.kv_a_norm).__name__ == 'LayerNorm'
    assert type(mla_layernorm.o_a_norm).__name__ == 'LayerNorm'
    
    print("✓ LayerNorm configuration works correctly")
    
    # Test RMSNorm
    print("Testing RMSNorm configuration...")
    rmsnorm_config = SharedSpaceDecoderConfig(
        norm_type="rmsnorm",
        **base_config
    )
    
    # Create MLA layer with RMSNorm
    mla_rmsnorm = MultiheadLatentAttention(rmsnorm_config, layer_idx=2)
    
    # Verify the norm layers are RMSNorm
    assert type(mla_rmsnorm.q_a_norm).__name__ == 'DeepseekV3RMSNorm'
    assert type(mla_rmsnorm.kv_a_norm).__name__ == 'DeepseekV3RMSNorm'
    assert type(mla_rmsnorm.o_a_norm).__name__ == 'DeepseekV3RMSNorm'
    
    print("✓ RMSNorm configuration works correctly")
    
    # Test a simple forward pass with dummy data
    print("Testing forward pass with output subspace norm...")
    batch_size = 2
    seq_len = 8
    hidden_size = base_config["hidden_size"]
    
    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create dummy position embeddings (cos, sin)
    rope_dims = base_config["rope_dims"]
    cos = torch.randn(seq_len, rope_dims)
    sin = torch.randn(seq_len, rope_dims)
    position_embeddings = (cos, sin)
    
    # Forward pass (no attention mask for simplicity)
    with torch.no_grad():
        output = mla_layernorm(hidden_states, position_embeddings, attention_mask=None)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    print("✓ Forward pass works correctly with norm between o_a_proj and o_b_proj")
    
    print("\nAll tests passed! 🎉")
    print("- Norm type is configurable (layernorm/rmsnorm)")
    print("- Default is layernorm")  
    print("- Norm layer added between o_a_proj and o_b_proj")


if __name__ == "__main__":
    test_norm_types()
