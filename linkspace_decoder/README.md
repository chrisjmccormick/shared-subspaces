# LinkedSpace Decoder: Flexible Module-to-Space Mappings

## Overview

The **LinkedSpace Decoder** extends the concept of shared attention subspaces (as seen in Multi-head Latent Attention architectures like DeepSeek-V3) by introducing **flexible, configurable mappings** between model modules and shared spaces.

Instead of fixed architectural patterns (e.g., "Q/KV share one space, O shares another"), LinkedSpace allows you to experiment with arbitrary groupings of modules across both attention and feed-forward layers.

## Key Concept: Flexible Space Configuration

In traditional MLA architectures, you have fixed roles for each subspace:
- Query subspace
- Key/Value subspace  
- Output subspace (optional)
- FFN subspace (optional)

**LinkedSpace** replaces this with a single, unified `spaces` configuration where you explicitly define:
1. The **size** of each shared space
2. Whether to apply **normalization** in that space
3. Which **modules** use that space

### Module Names

Valid module names that can be mapped to spaces:

**Attention modules:**
- `"Q"` - Query projections
- `"K"` - Key projections
- `"V"` - Value projections
- `"O"` - Output projections

**Feed-Forward modules:**
- `"in"` - Input projections (SwiGLU input branch)
- `"gate"` - Gate projections (SwiGLU gate branch)
- `"out"` - Output projections

## Configuration Examples

### Example 1: All modules in one space

```python
spaces = {
    0: {
        "size": 768,
        "norm": True,
        "modules": ["Q", "K", "V", "O", "in", "gate", "out"]
    }
}
```

All attention and FFN modules share a single 768-dimensional space with normalization.

### Example 2: Separate attention and FFN spaces

```python
spaces = {
    0: {
        "size": 512,
        "norm": True,
        "modules": ["Q", "K", "V", "O"]
    },
    1: {
        "size": 256,
        "norm": True,
        "modules": ["in", "gate", "out"]
    }
}
```

Attention modules share one space, FFN modules share another.

### Example 3: Fine-grained control

```python
spaces = {
    0: {
        "size": 384,
        "norm": True,
        "modules": ["K", "V"]  # Keys and values together
    },
    1: {
        "size": 256,
        "norm": True,
        "modules": ["Q", "O"]  # Queries and outputs together
    },
    2: {
        "size": 512,
        "norm": False,
        "modules": ["in", "gate", "out"]  # FFN without norm
    }
}
```

Each group of modules gets its own space with custom configuration.

### Example 4: Cross-layer sharing (Advanced)

```python
spaces = {
    0: {
        "size": 768,
        "norm": True,
        "modules": ["in", "gate", "out"]  # FFN modules
    },
    1: {
        "size": 192,
        "norm": True,
        "modules": ["K", "V"]  # Keys and values
    },
    2: {
        "size": 384,
        "norm": True,
        "modules": ["Q", "O"]  # Queries and outputs
    }
}
```

This creates interesting cross-module dependencies, allowing the model to share representations between attention and feed-forward in novel ways.

## Architecture Details

### Projection Flow

For each module assigned to a space:

1. **Shared Projection**: `hidden_size → space_size`
2. **Optional Normalization**: Applied in the space (if `norm=True`)
3. **Private Projection**: `space_size → module_specific_size`

For example, if Query is in a 256-dim space:
```
Input [hidden_size] 
  → Q_shared_proj [space_size=256]
  → Q_shared_norm (optional)
  → Q_private_proj [num_heads × head_dim]
```

### Benefits of Flexible Spaces

1. **Gradient Flow**: Modules in the same space receive shared gradients, potentially enabling faster learning
2. **Parameter Efficiency**: Shared spaces reduce parameters while maintaining expressiveness
3. **Architectural Search**: Easy experimentation with different module groupings
4. **Controlled Coupling**: Explicitly define which modules should coordinate

## Usage

### Creating a Model

```python
from models.linkspace_config import LinkedSpaceDecoderConfig
from models.linkspace_decoder import LinkedSpaceDecoderModel

# Define your space configuration
spaces = {
    0: {"size": 512, "norm": True, "modules": ["Q", "K", "V", "O"]},
    1: {"size": 256, "norm": True, "modules": ["in", "gate", "out"]}
}

# Create config
config = LinkedSpaceDecoderConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    qk_private_dim=64,
    vo_private_dim=64,
    nope_dims=48,
    rope_dims=16,
    spaces=spaces,  # Your flexible space configuration
    num_dense_layers=1,  # First layer uses standard attention
)

# Create model
model = LinkedSpaceDecoderModel(config)
```

### For Language Modeling

```python
from layers.task_heads import LinkedSpaceDecoderForCausalLM

# Create model with LM head
model = LinkedSpaceDecoderForCausalLM(config)

# Use with HuggingFace Trainer or custom training loop
```

## Configuration Reference

### Core Parameters

- `vocab_size` (int): Vocabulary size
- `hidden_size` (int): Model hidden dimension
- `num_hidden_layers` (int): Number of transformer blocks
- `num_attention_heads` (int): Number of attention heads
- `intermediate_size` (int): Feed-forward hidden dimension

### LinkedSpace Parameters

- `spaces` (dict): Dictionary defining space-to-module mappings
  - Each space has: `size`, `norm`, `modules`
- `qk_private_dim` (int): Per-head dimension for queries and keys
- `vo_private_dim` (int): Per-head dimension for values and outputs

### RoPE Parameters

- `rope_dims` (int): Number of dimensions with rotary position encoding
- `nope_dims` (int): Number of non-positional dimensions
- `rope_theta` (float): RoPE base frequency
- `max_position_embeddings` (int): Maximum sequence length

### Other Parameters

- `num_dense_layers` (int): Number of initial layers without linkspaces
- `rms_norm_eps` (float): Epsilon for RMSNorm (all norms use RMSNorm)
- `attention_bias` (bool): Whether to use bias in attention projections
- `vocab_subspace` (bool): Whether to decompose vocabulary embeddings
- `tie_word_embeddings` (bool): Whether to tie input/output embeddings

## Validation

The config automatically validates:
- All modules must be assigned to exactly one space
- Module names must be valid (Q, K, V, O, in, gate, out)
- Each space must have `size` and `modules` keys
- Private dimensions must be specified
- All spaces use RMSNorm normalization (always enabled)

## Comparison with SubspaceDecoder

| Feature | SubspaceDecoder | LinkedSpaceDecoder |
|---------|-----------------|-------------------|
| Q/K/V spaces | Fixed, separate parameters | Flexible, any grouping |
| O space | Optional, separate parameter | Flexible, any grouping |
| FFN spaces | Fixed pattern (in/gate/out) | Flexible, any grouping |
| Cross-module sharing | Not possible | Fully configurable |
| Config complexity | Lower (separate params) | Higher (unified spaces) |
| Flexibility | Limited to predefined patterns | Arbitrary module groupings |

## Experiments to Try

1. **Shared Q/O space**: Do queries and outputs benefit from a shared representation?
2. **Cross-layer spaces**: Can attention and FFN share beneficial features?
3. **Asymmetric spaces**: Different sizes for different module groups
4. **Normalization ablation**: Which spaces benefit from normalization?
5. **Minimal spaces**: Can one space serve all modules efficiently?

## Files

```
linkspace_decoder/
├── models/
│   ├── linkspace_config.py       # Configuration class
│   └── linkspace_decoder.py      # Model implementation
├── layers/
│   ├── linkspace_mla.py          # Attention layer
│   ├── linkspace_feedforward.py  # FFN layer
│   └── task_heads.py             # LM head and task-specific heads
├── configs/                       # Example configurations
├── tests/                         # Unit tests
├── scripts/                       # Training and evaluation scripts
├── utils.py                       # Utility functions
└── README.md                      # This file
```

## Citation

If you use LinkedSpace in your research, please cite:

```bibtex
@misc{linkedspace2024,
  title={LinkedSpace Decoder: Flexible Module-to-Space Mappings for Efficient Transformers},
  author={Your Name},
  year={2024},
}
```

## Related Work

- [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) - Multi-head Latent Attention
- [SubspaceDecoder](../subspace_decoder/) - Fixed subspace patterns
- [Blog Post on Output Latent Spaces](https://mccormickml.com/2025/07/28/output-latent-spaces-in-multihead-attention/)

## License

[Same as parent repository]

