"""
Simple example script demonstrating LinkedSpace model usage.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linkspace_config import LinkedSpaceDecoderConfig
from models.linkspace_decoder import LinkedSpaceDecoderModel
from layers.task_heads import LinkedSpaceDecoderForCausalLM
from utils import summarize_parameters


def create_simple_model():
    """Create a simple LinkedSpace model."""
    
    print("="*70)
    print("Creating LinkedSpace Model - Simple Example")
    print("="*70)
    
    # Define space configuration
    # This example puts attention modules in one space and FFN in another
    spaces = {
        0: {
            "size": 256,
            "norm": True,
            "modules": ["Q", "K", "V", "O"]
        },
        1: {
            "size": 128,
            "norm": True,
            "modules": ["in", "gate", "out"]
        }
    }
    
    # Create configuration
    config = LinkedSpaceDecoderConfig(
        vocab_size=10000,  # Small vocab for quick testing
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        qk_private_dim=64,
        vo_private_dim=64,
        nope_dims=48,
        rope_dims=16,
        max_position_embeddings=256,
        spaces=spaces,
        num_dense_layers=1,  # First layer uses standard attention
        norm_type="rmsnorm",
    )
    
    print("\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Dense layers: {config.num_dense_layers}")
    print(f"\nSpace Configuration:")
    for space_id, space_config in config.spaces.items():
        print(f"  Space {space_id}:")
        print(f"    Size: {space_config['size']}")
        print(f"    Norm: {space_config['norm']}")
        print(f"    Modules: {', '.join(space_config['modules'])}")
    
    # Create model
    print("\nCreating model...")
    model = LinkedSpaceDecoderForCausalLM(config)
    
    print("\nModel created successfully!")
    print("\nParameter Summary:")
    summarize_parameters(model, display_bias=False)
    
    return model, config


def test_forward_pass(model, config):
    """Test a forward pass through the model."""
    
    print("\n" + "="*70)
    print("Testing Forward Pass")
    print("="*70)
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    print("\n✓ Forward pass successful!")
    
    # Show some example predictions
    print("\nExample predictions for first token:")
    first_token_logits = logits[0, 0, :10]  # First 10 vocab items
    print(f"  Logits: {first_token_logits.tolist()}")
    
    probs = torch.softmax(logits[0, 0], dim=-1)
    top_5_probs, top_5_indices = torch.topk(probs, k=5)
    print(f"\nTop 5 predicted tokens:")
    for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices)):
        print(f"  {i+1}. Token {idx.item()}: {prob.item():.4f}")


def compare_space_configurations():
    """Compare different space configurations."""
    
    print("\n" + "="*70)
    print("Comparing Different Space Configurations")
    print("="*70)
    
    configs = [
        {
            "name": "Separate Attn/FFN",
            "spaces": {
                0: {"size": 256, "norm": True, "modules": ["Q", "K", "V", "O"]},
                1: {"size": 128, "norm": True, "modules": ["in", "gate", "out"]}
            }
        },
        {
            "name": "Unified Space",
            "spaces": {
                0: {"size": 384, "norm": True, "modules": ["Q", "K", "V", "O", "in", "gate", "out"]}
            }
        },
        {
            "name": "Fine-grained",
            "spaces": {
                0: {"size": 192, "norm": True, "modules": ["K", "V"]},
                1: {"size": 256, "norm": True, "modules": ["Q", "O"]},
                2: {"size": 128, "norm": True, "modules": ["in", "gate", "out"]}
            }
        }
    ]
    
    print("\n")
    for cfg in configs:
        config = LinkedSpaceDecoderConfig(
            vocab_size=10000,
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            qk_private_dim=64,
            vo_private_dim=64,
            nope_dims=48,
            rope_dims=16,
            max_position_embeddings=256,
            spaces=cfg["spaces"],
            num_dense_layers=1,
        )
        
        model = LinkedSpaceDecoderModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"{cfg['name']:20} - {total_params:,} parameters")
    
    print("\nNote: Parameter counts vary based on space configuration!")


if __name__ == "__main__":
    # Create and test a simple model
    model, config = create_simple_model()
    
    # Test forward pass
    test_forward_pass(model, config)
    
    # Compare configurations
    compare_space_configurations()
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)

