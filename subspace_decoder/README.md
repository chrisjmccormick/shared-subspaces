# DeepSeek V3 with MLA-o, Research Prototype

This repository contains a minimal modification to the DeepSeek V3 attention mechanism, implementing the MLA-o output projection for research and benchmarking purposes.

## Overview

The goal is to evaluate the impact of adding a latent subspace projection to the attention output in the Multihead Latent Attention (MLA) architecture. This is implemented as a minimal modification to the established DeepSeek V3 implementation to ensure credibility and reproducibility.

## Key Changes

The modification decomposes the output heads into private and shared subspaces. 

### Architecture Details

```python
# Standard output projection
attn_output = self.o_proj(attn_output)

# Decomposed output projection (when enabled)
if self.use_output_subspace:
    attn_output = self.o_a_proj(attn_output)  # Project to latent space
    attn_output = self.o_b_proj(attn_output)  # Project back to model space
```

## Files

- `layers/deepseek_mla_o.py`: Modified DeepSeek V3 attention class with output subspace projection
- `models/configuration_deepseek.py`: Extended DeepSeek V3 config with output subspace parameters
- `example_monkey_patch.py`: Example script showing how to apply the modification
- `configs/initial_mla.json`: Example training configuration
- `requirements.txt`: Minimal dependencies for the research prototype

## Usage

### Basic Monkey Patching

```python
from transformers import AutoModelForCausalLM
from layers.deepseek_mla_o import DeepseekV3Attention
from example_monkey_patch import monkey_patch_attention_layers

# Load the base model
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5")

# Apply the modification
modified_model = monkey_patch_attention_layers(
    model,
    use_output_subspace=True,
    o_latent_dim=512  # Dimension of the output latent space
)
```

### Configuration Options

The modification is controlled by two config parameters:

- `use_output_subspace`: Boolean flag to enable/disable the change
- `o_latent_dim`: Dimension of the intermediate latent space

These parameters are properly integrated into the extended `DeepseekV3Config` class with validation.

### Running the Example

```bash
pip install -r requirements.txt
python example_monkey_patch.py
```

## Research Approach

### Why This Approach?

1. **Minimal Modification**: Only changes the output projection, keeping everything else identical
2. **Credibility**: Based on established DeepSeek V3 implementation
3. **Isolation**: Changes are contained and don't affect other model components
4. **Reproducibility**: Easy for others to verify and reproduce

### Dependencies Simplified

The implementation removes unnecessary dependencies:
- ❌ Flash attention (adds complexity)
- ❌ Unused transformers utilities
- ❌ Heavy padding/unpadding logic
- ✅ Core PyTorch and transformers functionality
- ✅ Essential attention mechanisms

### Weight Initialization

When enabling the output subspace, the new projection weights are initialized using Xavier uniform initialization. For research purposes, you might want to experiment with different initialization strategies.

## Benchmarking

This prototype is designed for:
- Small-medium scale pre-training runs
- Fine-tuning on standard benchmarks
- Ablation studies comparing MLA vs MLA-o
- Performance and efficiency analysis

## Limitations

1. **No Flash Attention**: Removed for simplicity, may impact performance on long sequences
2. **Limited Testing**: This is a research prototype, not production-ready
3. **Weight Transfer**: The weight copying logic assumes compatible architectures

## Future Work

- Add flash attention support back if needed
- Implement more sophisticated weight initialization strategies
- Add comprehensive testing and validation
- Explore different latent space dimensions and architectures

## Citation

If you use this code in your research, please cite both the original DeepSeek V3 paper and the MLA paper:

```bibtex
@article{deepseek2024,
  title={DeepSeek V3: Scaling Up DeepSeek Coder with a Comprehensive Code Dataset},
  author={...},
  journal={...},
  year={2024}
}

@article{mla2024,
  title={Multihead Latent Attention: A Novel Attention Mechanism for Transformer Models},
  author={...},
  journal={...},
  year={2024}
}
```
