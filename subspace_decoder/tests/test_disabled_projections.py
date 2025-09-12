"""
Test script to verify that disabled projections work correctly.
Tests different combinations of None values for q_shared_dim, kv_shared_dim, o_shared_dim.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shared_space_config import SharedSpaceDecoderConfig
from layers.mla import MultiheadLatentAttention


def test_projection_combinations():
    """Test different combinations of disabled projections."""
    
    # Base config
    base_config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "rope_dims": 32,
        "qk_private_dim": 64,
        "vo_private_dim": 64,
        "nope_dims": 32,
        "attention_bias": False,
        "num_dense_layers": 1,  # Start MLA from layer 1
        "ffn_decompose": False,
        "ffn_rank": None,
        "vocab_subspace": False,
        "vocab_rank": None,
    }
    
    test_cases = [
        # (q_shared_dim, kv_shared_dim, o_shared_dim, description)
        (192, 96, 192, "All projections enabled"),
        (None, 96, 192, "Q projection disabled"),
        (192, None, 192, "KV projection disabled"),
        (192, 96, None, "O projection disabled"),
        (None, None, 192, "Q and KV projections disabled"),
        (None, 96, None, "Q and O projections disabled"),
        (192, None, None, "KV and O projections disabled"),
        (None, None, None, "All projections disabled"),
    ]
    
    batch_size, seq_len = 2, 64
    
    for q_dim, kv_dim, o_dim, desc in test_cases:
        print(f"\nTesting: {desc}")
        print(f"  q_shared_dim={q_dim}, kv_shared_dim={kv_dim}, o_shared_dim={o_dim}")
        
        # Create config
        config_dict = base_config.copy()
        config_dict.update({
            "q_shared_dim": q_dim,
            "kv_shared_dim": kv_dim,
            "o_shared_dim": o_dim,
        })
        
        try:
            config = SharedSpaceDecoderConfig(**config_dict)
            
            # Create MLA layer (layer 2, so it uses latent spaces)
            mla = MultiheadLatentAttention(config, layer_idx=2)
            
            # Create dummy input
            hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
            
            # Create dummy position embeddings (cos, sin)
            cos = torch.randn(seq_len, config.rope_dims)
            sin = torch.randn(seq_len, config.rope_dims)
            position_embeddings = (cos, sin)
            
            # Forward pass
            output = mla(hidden_states, position_embeddings, attention_mask=None)
            
            # Check output shape
            expected_shape = (batch_size, seq_len, config.hidden_size)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            print(f"  ✓ Success! Output shape: {output.shape}")
            
            # Print some architecture details
            print(f"    Effective q_shared_dim: {mla.q_shared_dim}")
            print(f"    Effective kv_shared_dim: {mla.kv_shared_dim}")
            print(f"    Uses output subspace: {mla.o_shared_dim is not None}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("Testing disabled projection combinations...")
    success = test_projection_combinations()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
