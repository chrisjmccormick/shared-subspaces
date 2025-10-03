# Subspace Decoder

This project implements and evaluates shared output latent spaces in Transformer decoder architectures using a patched version of HuggingFace's DeepSeek-V3 implementation. Building on the encoder experiments, this decoder-based approach provides a more direct path to evaluating the output subspace decomposition in production-ready model architectures.

## Overview

Rather than implementing a custom decoder from scratch, this project takes a surgical approach by patching the existing `DeepseekV3ForCausalLM` model to add the shared output latent space decomposition. This allows us to leverage the robust, optimized implementation while focusing specifically on evaluating the impact of the output subspace.

Weights are available on huggingface:
*  [deepseek-tiny-v0.1](https://huggingface.co/ChrisMcCormick/deepseek-tiny-v0.1)
*  [deepseek-tiny-mla-o-v0.1](https://huggingface.co/ChrisMcCormick/deepseek-tiny-mla-o-v0.1)

### Key Features

- **Patched DeepSeek-V3**: Modifies the existing HuggingFace implementation rather than building from scratch
- **Multiple Normalization Strategies**: Experiments with different approaches to normalizing the decomposed output
- **Scalable Architecture**: Tests at both short (128 tokens) and longer (1,024 tokens) sequence lengths
- **Performance Optimizations**: Includes `torch.compile`, bf16 support, and example packing for efficient training

## Architecture Variants

The experiments compare two attention mechanisms:

1. **MLA**: Multihead Latent Attention (DeepSeek-V3's approach)
2. **MLA-o**: MLA with shared output latent space

(Note: It doesn't currently include results for standard Multihead Attention, but you can find those in the Encoder experiments.)

### Output Subspace Decomposition

The shared output subspace decomposes the standard attention output projection $W^O$ into two components:

```
W^O = W^{OB} \cdot W^{OA}
```

Where:
- $W^{OA}_i$: Per-head projection from attention dimension to output latent dimension
- $W^{OB}$: Shared projection from output latent dimension to model dimension

This is implemented by replacing the standard `o_proj` linear layer with a `nn.Sequential` containing the two decomposed projections.

## Experimental Results

### Model Configuration

- **Architecture**: 6-layer decoder with Mixture of Experts
- **Hidden Dimension**: 256
- **Attention Heads**: 8
- **Head Dimension**: 32
- **Parameters**: ~16M (MLA: 16.26M, MLA-o: 16.17M)

### Performance (Sequence Length 1,024)

| Attention | SST-2 Accuracy | WikiText-103 Perplexity | Query Latent | Key-Value Latent | Output Latent |
|:---------:|:--------------:|:----------------------:|:------------:|:----------------:|:-------------:|
| MLA       | 87.96%         | 28.89                  | 96           | 64               | n/a           |
| MLA-o     | 86.24%         | 29.33                  | 96           | 64               | 96            |

### Key Findings

1. **Performance Trade-off**: MLA-o achieves comparable performance to MLA with a ~1.7% drop in SST-2 accuracy
2. **Parameter Efficiency**: ~0.9% reduction in parameters (90K fewer parameters)
3. **Throughput**: No significant training throughput improvements observed at current scale
4. **Consistency**: Results are consistent across different sequence lengths (128 and 1,024 tokens)

## Implementation Details

### Patching Strategy

The core implementation uses a patching approach that replaces the `o_proj` module:

```python
# Replace single linear layer with sequential decomposition
attn.o_proj = nn.Sequential(
    nn.Linear(in_features, o_latent_dim, bias=False),  # W^OA
    nn.Linear(o_latent_dim, out_features, bias=bias),  # W^OB
)
```

Key implementation files:
- `layers/patch_o_proj.py`: Core patching functionality with multiple normalization variants
- `layers/deepseek_mla_o.py`: Alternative full attention class implementation (currently unused)
- `models/configuration_deepseek.py`: Extended DeepSeek config with output subspace parameters (currently unused)

### Normalization Strategies

The project experiments with several normalization approaches:

1. **Sequential**: Standard decomposition with optional post-sum RMSNorm
2. **Per-head RMSNorm**: Independent normalization for each attention head, optionally with shared gain parameter across heads.

The post-sum RMSNorm strategy provides the best performance-speed trade-off.

### Training Optimizations

- **torch.compile**: 15-25% training speedup using PyTorch compilation
- **bf16**: Mixed precision training for improved memory efficiency
- **Example Packing**: Concatenates samples to reduce padding tokens at longer sequence lengths

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

Model and training configurations are defined in JSON files in the `configs/` directory:

- `initial_mla.json`: Baseline MLA configuration
- `initial_mla_o_norm.json`: MLA-o with normalization
- `seqlen1024_mla-on_q96_k64_o96.json`: Optimized configuration for longer sequences

### Training

Pre-train a model on WikiText-103:

```bash
python scripts/train.py --config configs/seqlen1024_mla-on_q96_k64_o96.json
```

### Fine-tuning

Fine-tune on SST-2:

```bash
python scripts/finetune_sst2.py --config configs/seqlen1024_mla-on_q96_k64_o96.json
```

### Interactive Experiments

Use the Jupyter notebook for interactive experimentation:

```bash
jupyter notebook scripts/run_experiments.ipynb
```

## Configuration Parameters

### Output Subspace Settings

```json
{
  "use_output_subspace": true,
  "o_proj_variant": "sequential_norm",
  "o_latent_dim": 96
}
```

- `use_output_subspace`: Enable/disable the output subspace decomposition
- `o_proj_variant`: Normalization strategy ("sequential", "sequential_norm", "per_head", etc.)
- `o_latent_dim`: Dimension of the shared output latent space

### Performance Settings

```json
{
  "bf16": true,
  "torch_compile": true,
  "torch_compile_backend": "inductor",
  "torch_compile_mode": "default"
}
```

## Known Issues and Limitations

1. **RoPE Configuration**: Currently uses 32 RoPE dimensions and 0 NoPE dimensions due to HuggingFace utility constraints. Intended configuration of 16/16 split requires further implementation work.

2. **MoE Layers**: Models inadvertently trained with Mixture of Experts layers instead of intended dense layers. This appears to work well but requires router health evaluation.

3. **Scale Dependency**: Throughput benefits likely require larger models with more attention heads to become apparent.

## Research Context

This decoder implementation addresses the core research question: **If input bottlenecks in MLA are effective, what is the impact of adding a similar bottleneck to the output?**

The decoder experiments provide validation using a production-ready architecture, complementing the custom encoder implementation in the `subspace_encoder/` project. The consistent results across both architectures strengthen the findings about the trade-offs involved in shared output latent spaces.

## Future Work

1. **Scaling Studies**: Test larger models with more attention heads to identify computational crossover points
2. **RoPE Implementation**: Fix the RoPE/NoPE dimension split for more accurate MLA reproduction
3. **Router Analysis**: Evaluate Mixture of Experts router health in trained models
4. **Throughput Benchmarking**: Systematic performance analysis at different model scales
5. **Advanced Normalization**: Explore additional normalization strategies for the output subspace

## Contributing

This project is part of ongoing research into efficient attention mechanisms. See the main repository README for contribution guidelines and discussion channels.

## Acknowledgments

- DeepSeek team for the original MLA implementation and research insights
- HuggingFace for the robust Transformers library implementation
- The broader research community working on efficient attention mechanisms