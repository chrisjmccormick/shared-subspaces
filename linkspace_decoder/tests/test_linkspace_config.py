"""
Test script to verify LinkedSpace configuration validation.
Tests different space configurations and module mappings.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linkspace_config import LinkedSpaceDecoderConfig


def test_basic_config():
    """Test basic valid configuration."""
    print("\n=== Testing basic configuration ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V", "O"]},
        1: {"size": 128, "norm": True, "modules": ["in", "gate", "out"]}
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
    
    print(f"✓ Basic config created successfully")
    print(f"  Space 0: {spaces[0]}")
    print(f"  Space 1: {spaces[1]}")
    return True


def test_single_space_all_modules():
    """Test configuration with all modules in one space."""
    print("\n=== Testing single space for all modules ===")
    
    spaces = {
        0: {
            "size": 768,
            "norm": True,
            "modules": ["Q", "K", "V", "O", "in", "gate", "out"]
        }
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
    
    print(f"✓ Single space config created successfully")
    print(f"  All modules in space 0 (size={spaces[0]['size']})")
    return True


def test_fine_grained_spaces():
    """Test fine-grained space configuration."""
    print("\n=== Testing fine-grained space configuration ===")
    
    spaces = {
        0: {"size": 192, "norm": True, "modules": ["K", "V"]},
        1: {"size": 256, "norm": True, "modules": ["Q"]},
        2: {"size": 384, "norm": False, "modules": ["O"]},
        3: {"size": 512, "norm": True, "modules": ["in", "gate"]},
        4: {"size": 256, "norm": False, "modules": ["out"]}
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
    
    print(f"✓ Fine-grained config created successfully")
    for space_id, space_cfg in spaces.items():
        print(f"  Space {space_id}: size={space_cfg['size']}, "
              f"norm={space_cfg['norm']}, modules={space_cfg['modules']}")
    return True


def test_module_assignment_query():
    """Test querying which space a module is assigned to."""
    print("\n=== Testing module assignment queries ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V"]},
        1: {"size": 128, "norm": True, "modules": ["O", "in"]},
        2: {"size": 192, "norm": False, "modules": ["gate", "out"]}
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
    
    # Test get_space_for_module
    assert config.get_space_for_module("Q") == 0
    assert config.get_space_for_module("K") == 0
    assert config.get_space_for_module("V") == 0
    assert config.get_space_for_module("O") == 1
    assert config.get_space_for_module("in") == 1
    assert config.get_space_for_module("gate") == 2
    assert config.get_space_for_module("out") == 2
    
    print(f"✓ Module assignment queries work correctly")
    print(f"  Q → space {config.get_space_for_module('Q')}")
    print(f"  O → space {config.get_space_for_module('O')}")
    print(f"  in → space {config.get_space_for_module('in')}")
    
    # Test get_module_space_config
    q_config = config.get_module_space_config("Q")
    assert q_config['size'] == 256
    assert q_config['norm'] == True
    
    out_config = config.get_module_space_config("out")
    assert out_config['size'] == 192
    assert out_config['norm'] == False
    
    print(f"✓ Module space config queries work correctly")
    return True


def test_invalid_configs():
    """Test that invalid configurations are rejected."""
    print("\n=== Testing invalid configuration rejection ===")
    
    # Test 1: Missing 'size' key
    try:
        spaces = {
            0: {"norm": True, "modules": ["Q", "K", "V", "O", "in", "gate", "out"]}
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
        print("✗ Should have failed: missing 'size' key")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected config missing 'size' key")
    
    # Test 2: Invalid module name
    try:
        spaces = {
            0: {"size": 256, "norm": True, "modules": ["Q", "K", "V", "X"]}  # X is invalid
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
        print("✗ Should have failed: invalid module name 'X'")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid module name")
    
    # Test 3: Module assigned to multiple spaces
    try:
        spaces = {
            0: {"size": 256, "norm": True, "modules": ["Q", "K"]},
            1: {"size": 128, "norm": True, "modules": ["Q", "V"]}  # Q in both!
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
        print("✗ Should have failed: Q assigned to multiple spaces")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected duplicate module assignment")
    
    # Test 4: Not all modules assigned
    try:
        spaces = {
            0: {"size": 256, "norm": True, "modules": ["Q", "K"]}  # Missing V, O, in, gate, out
        }
        # This should actually NOT fail during config creation, only when trying to use the layer
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
        print(f"✓ Config allows partial module assignment (modules not in spaces will use direct projection)")
    except ValueError as e:
        print(f"Note: Config rejected partial assignment: {e}")
    
    return True


def test_normalization_options():
    """Test that normalization can be disabled per space."""
    print("\n=== Testing normalization options ===")
    
    spaces = {
        0: {"size": 256, "norm": True, "modules": ["Q", "K", "V"]},
        1: {"size": 128, "norm": False, "modules": ["O"]},  # No norm
        2: {"size": 192, "norm": True, "modules": ["in", "gate", "out"]}
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
    
    # Verify norm settings
    assert config.get_module_space_config("Q")['norm'] == True
    assert config.get_module_space_config("O")['norm'] == False
    assert config.get_module_space_config("in")['norm'] == True
    
    print(f"✓ Normalization options work correctly")
    print(f"  Space 0 (Q,K,V): norm={True}")
    print(f"  Space 1 (O): norm={False}")
    print(f"  Space 2 (in,gate,out): norm={True}")
    return True


if __name__ == "__main__":
    print("="*70)
    print("Testing LinkedSpace Configuration")
    print("="*70)
    
    tests = [
        test_basic_config,
        test_single_space_all_modules,
        test_fine_grained_spaces,
        test_module_assignment_query,
        test_invalid_configs,
        test_normalization_options,
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
        print("\n🎉 All configuration tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)

