# LinkSpace Decoder - Quick Start Guide

## What You've Got

The `linkspace_decoder` is a complete implementation of a flexible transformer decoder with **configurable space-to-module mappings**. Unlike traditional architectures with fixed subspaces, LinkSpace lets you experiment with arbitrary groupings of modules.

## Key Innovation

Instead of separate parameters like:
- `q_shared_dim`
- `kv_shared_dim`
- `o_shared_dim`
- `ffn_rank`

You get a unified `spaces` configuration:

```python
spaces = {
    0: {"size": 768, "modules": ["K", "V", "Q", "in", "gate", "out"]},
    1: {"size": 256, "modules": ["O"]}
}
```

This allows you to:
- Group any modules together (attention & FFN can share spaces!)
- Use uniform RMSNorm across all spaces
- Experiment with cross-module dependencies

## Quick Test

Run the simple example:

```bash
cd shared-subspaces/linkspace_decoder
python scripts/simple_example.py
```

This will:
1. Create a small LinkSpace model
2. Run a forward pass
3. Compare different space configurations

## Run Tests

Test the configuration system:
```bash
python tests/test_linkspace_config.py
```

Test forward passes:
```bash
python tests/test_linkspace_forward.py
```

## Example Configurations

We've provided 4 example configs in `configs/`:

1. **tiny_linkspace_example1.json** - Separate attention and FFN spaces
2. **tiny_linkspace_example2.json** - Single unified space for all modules
3. **tiny_linkspace_example3.json** - Fine-grained with Q/O sharing
4. **gpt2_linkspace.json** - GPT-2 sized model with LinkSpace

## Creating Your Own Config

Use the config creation utility:

```bash
python configs/create_new_config.py my_experiment \
    --base configs/tiny_linkspace_example1.json \
    --description "Experiment with larger spaces" \
    --set model.spaces.0.size=512 \
    --set model.num_hidden_layers=8
```

## Using in Code

```python
from models.linkspace_config import LinkedSpaceDecoderConfig
from models.linkspace_decoder import LinkedSpaceDecoderModel

# Define your space configuration
spaces = {
    0: {"size": 512, "modules": ["Q", "K", "V", "O"]},
    1: {"size": 256, "modules": ["in", "gate", "out"]}
}

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
    spaces=spaces,
    num_dense_layers=1,
)

model = LinkedSpaceDecoderModel(config)
```

## Valid Module Names

- **Attention**: `"Q"`, `"K"`, `"V"`, `"O"`
- **Feed-Forward**: `"in"`, `"gate"`, `"out"`

Each module must be assigned to exactly one space.

## Architecture Flow

For each module in a space:

```
Input [hidden_size]
  ↓
shared_proj [hidden_size → space_size]
  ↓
shared_norm (if enabled)
  ↓
private_proj [space_size → module_dim]
  ↓
Output [module_dim]
```

## File Structure

```
linkspace_decoder/
├── models/
│   ├── linkspace_config.py       # Configuration with flexible spaces
│   └── linkspace_decoder.py      # Main model implementation
├── layers/
│   ├── linkspace_mla.py          # Attention with linkspace support
│   ├── linkspace_feedforward.py  # FFN with linkspace support
│   └── task_heads.py             # Language modeling head
├── configs/                       # Example configurations
├── tests/                         # Comprehensive test suite
├── scripts/                       # Example scripts
├── utils.py                       # Utility functions
├── README.md                      # Full documentation
└── QUICKSTART.md                  # This file
```

## Next Steps

1. **Run the tests** to verify everything works
2. **Try the example script** to see it in action
3. **Experiment with different space configurations**
4. **Compare parameter counts** with different groupings
5. **Train on your task** and compare to baseline

## Experiments to Try

1. **Q/O sharing**: Do queries and outputs benefit from shared representations?
2. **Cross-layer spaces**: Can attention and FFN share features effectively?
3. **Minimal configuration**: How small can you make the spaces?
4. **Asymmetric spaces**: Different sizes for different modules
5. **Normalization ablation**: Which spaces need normalization?

## Key Differences from SubspaceDecoder

| Feature | SubspaceDecoder | LinkedSpaceDecoder |
|---------|-----------------|-------------------|
| Configuration | Fixed parameters | Flexible `spaces` dict |
| Module grouping | Pre-defined | Arbitrary |
| Cross-module sharing | Not possible | Fully supported |
| Config complexity | Simpler | More flexible |

## Support

See `README.md` for complete documentation, architecture details, and configuration reference.

Happy experimenting! 🚀

