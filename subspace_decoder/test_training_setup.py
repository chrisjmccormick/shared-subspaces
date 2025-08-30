#!/usr/bin/env python3
"""
Test script to verify the training setup works with MLA-o attention.
"""

import torch
import json
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import create_model_with_mla_attention

def test_model_creation():
    """Test that we can create a model with MLA-o attention."""
    
    # Load the config
    config_path = PROJECT_ROOT / "configs" / "initial_mla.json"
    with open(config_path, 'r') as f:
        full_cfg = json.load(f)
    
    model_cfg = full_cfg['model']
    
    print("Testing model creation with MLA-o attention...")
    print(f"Config: {json.dumps(model_cfg, indent=2)}")
    
    # Create the model
    model = create_model_with_mla_attention(model_cfg)
    
    print(f"\nModel created successfully!")
    print(f"Model type: {type(model)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test a forward pass
    print("\nTesting forward pass...")
    
    # Create dummy input
    batch_size = 2
    seq_length = 64
    vocab_size = model_cfg["vocab_size"]
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # For causal LM, labels are the same as input_ids
        )
    
    print(f"Forward pass successful!")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Check if MLA-o attention is actually being used
    print("\nChecking MLA-o attention usage...")
    
    mla_layers = 0
    total_layers = 0
    
    for i, layer in enumerate(model.model.layers):
        total_layers += 1
        if hasattr(layer.self_attn, 'o_a_proj') and hasattr(layer.self_attn, 'o_b_proj'):
            mla_layers += 1
            print(f"  Layer {i}: MLA-o attention (o_a_proj: {layer.self_attn.o_a_proj.out_features}, o_b_proj: {layer.self_attn.o_b_proj.in_features})")
        else:
            print(f"  Layer {i}: Standard attention")
    
    print(f"\nSummary:")
    print(f"  Total layers: {total_layers}")
    print(f"  MLA-o layers: {mla_layers}")
    print(f"  Standard layers: {total_layers - mla_layers}")
    
    if mla_layers > 0:
        print("✅ MLA-o attention is being used!")
    else:
        print("❌ MLA-o attention is not being used!")
    
    return model

if __name__ == "__main__":
    test_model_creation()
