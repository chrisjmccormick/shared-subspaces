"""
Test script to verify LinkedSpace model forward pass.
Tests different space configurations with actual forward passes.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linkspace_config import LinkedSpaceDecoderConfig
from models.linkspace_decoder import LinkedSpaceDecoderModel
from layers.linkspace_mla import LinkedSpaceMLA
from layers.linkspace_feedforward import LinkedSpaceFeedForward


def test_basic_forward():
    """Test basic forward pass with simple configuration."""
    print("\n=== Testing basic forward pass ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V", "O"]},
        1: {"size": 128, "norm": True, "modules": ["in", "gate", "out"]}
    }
    
    config = LinkedSpaceDecoderConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        max_position_embeddings=128,
        spaces=spaces,
        num_dense_layers=1,
    )
    
    model = LinkedSpaceDecoderModel(config)
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✓ Basic forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    return True


def test_single_space_forward():
    """Test forward pass with all modules in one space."""
    print("\n=== Testing single space forward pass ===")
    
    spaces = {
        0: {
            "size": 384,
            "norm": True,
            "modules": ["Q", "K", "V", "O", "in", "gate", "out"]
        }
    }
    
    config = LinkedSpaceDecoderConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=2,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        max_position_embeddings=128,
        spaces=spaces,
        num_dense_layers=0,  # All layers use linkspaces
    )
    
    model = LinkedSpaceDecoderModel(config)
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape
    
    print(f"✓ Single space forward pass successful")
    print(f"  All modules share space 0 (size={spaces[0]['size']})")
    return True


def test_fine_grained_forward():
    """Test forward pass with fine-grained space configuration."""
    print("\n=== Testing fine-grained space forward pass ===")
    
    spaces = {
        0: {"size": 192, "norm": True, "modules": ["K", "V"]},
        1: {"size": 256, "norm": True, "modules": ["Q", "O"]},
        2: {"size": 128, "norm": False, "modules": ["in", "gate", "out"]}
    }
    
    config = LinkedSpaceDecoderConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=3,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        max_position_embeddings=128,
        spaces=spaces,
        num_dense_layers=1,
    )
    
    model = LinkedSpaceDecoderModel(config)
    
    batch_size = 2
    seq_len = 24
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape
    
    print(f"✓ Fine-grained space forward pass successful")
    print(f"  K,V in space 0; Q,O in space 1; in,gate,out in space 2")
    return True


def test_mla_layer_directly():
    """Test LinkedSpaceMLA layer directly."""
    print("\n=== Testing LinkedSpaceMLA layer ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V"]},
        1: {"size": 192, "norm": False, "modules": ["O"]}
    }
    
    config = LinkedSpaceDecoderConfig(
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        max_position_embeddings=128,
        spaces=spaces,
        num_dense_layers=1,
    )
    
    # Create MLA layer (layer 2 uses linkspaces)
    mla = LinkedSpaceMLA(config, layer_idx=2)
    
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Create position embeddings
    cos = torch.randn(seq_len, config.rope_dims)
    sin = torch.randn(seq_len, config.rope_dims)
    position_embeddings = (cos, sin)
    
    # Forward pass
    output = mla(hidden_states, position_embeddings, attention_mask=None)
    
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape
    
    print(f"✓ LinkedSpaceMLA layer forward pass successful")
    print(f"  Q,K,V in space 0; O in space 1")
    return True


def test_ffn_layer_directly():
    """Test LinkedSpaceFeedForward layer directly."""
    print("\n=== Testing LinkedSpaceFeedForward layer ===")
    
    spaces = {
        0: {"size": 384, "norm": True, "modules": ["in", "gate"]},
        1: {"size": 256, "norm": False, "modules": ["out"]}
    }
    
    config = LinkedSpaceDecoderConfig(
        hidden_size=512,
        num_hidden_layers=4,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        spaces=spaces,
        num_dense_layers=1,
    )
    
    # Create FFN layer (layer 2 uses linkspaces)
    ffn = LinkedSpaceFeedForward(config, layer_idx=2)
    
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = ffn(hidden_states)
    
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape
    
    print(f"✓ LinkedSpaceFeedForward layer forward pass successful")
    print(f"  in,gate in space 0; out in space 1")
    return True


def test_dense_vs_linkspace_layers():
    """Test that dense layers work correctly alongside linkspace layers."""
    print("\n=== Testing dense + linkspace layers ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V", "O"]},
        1: {"size": 128, "norm": True, "modules": ["in", "gate", "out"]}
    }
    
    config = LinkedSpaceDecoderConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        max_position_embeddings=128,
        spaces=spaces,
        num_dense_layers=3,  # First 3 layers are dense
    )
    
    model = LinkedSpaceDecoderModel(config)
    
    # Check that first 3 layers don't use linkspaces
    for i in range(3):
        assert not model.layers[i].self_attn.use_linkspaces
        assert model.layers[i].ffn.is_dense
    
    # Check that remaining layers use linkspaces
    for i in range(3, 6):
        assert model.layers[i].self_attn.use_linkspaces
        assert not model.layers[i].ffn.is_dense
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape
    
    print(f"✓ Dense + linkspace layers work correctly")
    print(f"  Layers 0-2: dense")
    print(f"  Layers 3-5: linkspace")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through linkspaces."""
    print("\n=== Testing gradient flow ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V", "O"]},
        1: {"size": 128, "norm": True, "modules": ["in", "gate", "out"]}
    }
    
    config = LinkedSpaceDecoderConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=2,
        intermediate_size=2048,
        num_attention_heads=8,
        rope_dims=16,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        max_position_embeddings=128,
        spaces=spaces,
        num_dense_layers=0,
    )
    
    model = LinkedSpaceDecoderModel(config)
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    
    # Compute a simple loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for key parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    print(f"✓ Gradients flow correctly through linkspaces")
    print(f"  All parameters have valid gradients")
    return True


if __name__ == "__main__":
    print("="*70)
    print("Testing LinkedSpace Forward Pass")
    print("="*70)
    
    tests = [
        test_basic_forward,
        test_single_space_forward,
        test_fine_grained_forward,
        test_mla_layer_directly,
        test_ffn_layer_directly,
        test_dense_vs_linkspace_layers,
        test_gradient_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 All forward pass tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)

