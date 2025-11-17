# shared-subspaces

This repository documents a completed research project exploring the use of shared subspaces in Transformer attention and feed-forward networks. The core investigation examines the impact of adding a shared *output* latent space to Multihead Latent Attention (MLA), a parameter-efficient attention mechanism used in models like DeepSeek and Kimi-K2.

## Project Overview

State-of-the-art Multihead Latent Attention (MLA) models like DeepSeek-V3 aggressively bottleneck the inputs to the attention layer, projecting the model's hidden dimension (e.g., 7,168) down to much smaller latent spaces for the query (e.g., 1,536) and key/value pairs (e.g., 512).

This project investigated a key question: **If these input bottlenecks are effective, what is the impact of adding a similar bottleneck to the *output* of the attention layer?**

Using the language of mechanistic interpretability, we can think of a token's vector representation as a "residual stream"—a communication bus that all model components read from and write to. In this framing, MLA's input projections constrain how much information each head can *read* from the stream. This project explored constraining where they can *write* to.

<img src='https://lh3.googleusercontent.com/d/1Hh95gVcbyyWpn-vNo6Mx4equVBdjjLcZ' alt='Simple block diagram of the attention heads with shared spaces illustrated as trapezoids' width='800' />

_The trapezoids in this illustration represent projections shared by all heads in a layer. Multihead Latent Attention defines a shared Key-Value latent projection (bottom) and a larger, shared Query latent projection (top). We explored adding a shared Output latent projection (right)._

## Key Findings

### Main Results

**The most interesting finding was that constraining the output space has a similar impact to constraining the query space.** It is indeed possible to add an output latent space to MLA—it's not inherently or catastrophically harmful to the model, and we generally saw similar outcomes when adding a subspace to either the query OR output. 

**At small model scales, the addition of shared subspaces to the attention block did not result in a more accurate model.** In fact, across all of our various configurations, no decomposed attention layer outperformed the baseline dense Multi-Head Attention (MHA).

### Context and Expectations

Part of the original premise for this project was based on DeepSeek's claim that MLA is inherently better than standard multi-head attention. In comparison to Group Query Attention (GQA) and Multi-Query Attention (MQA), which sacrifice some accuracy in exchange for cache efficiency, DeepSeek claimed that MLA actually performs better. Based on this, we hypothesized that:

1. Substituting MLA for MHA should result in a more performant model (both speed and accuracy)
2. Adding an output subspace might provide further improvement over standard MHA

### What We Actually Found

**At small model scales:**
- Any speed improvement from the reduction in FLOPs was not noticeable
- Any kind of subspace constraint on the attention layer was harmful to accuracy
- No decomposed attention configuration outperformed the baseline dense MHA

**At larger scales (124M parameters, GPT-2 scale):**
- We did begin to see small speed improvements from the decompositions

### Interpretation

Our conclusion is that DeepSeek's claims about MLA superiority must be more true at particular model scales and with the right constraints. Several factors may explain our different results:

- **Scale dependency**: The benefits of latent attention may only emerge at larger model sizes
- **Non-isoparametric experiments**: Our experiments were not isoparametric, meaning the MLA variants had fewer parameters than the MHA baseline, potentially putting them at a disadvantage in terms of capacity
- **Optimization and training**: The specific training dynamics and hyperparameters may need to be optimized differently for latent attention mechanisms at smaller scales

## What We Built

One of the most valuable outcomes of this project was the creation of **a comprehensive framework for testing small-scale Transformer architectures** using the HuggingFace ecosystem, with particular focus on testing shared subspace configurations.

### Core Model Implementations

We built both **decoder and encoder models** with modern architectural improvements. In contrast to the original BERT and GPT-2 architectures, we modernized our implementations with:

- **RoPE (Rotary Position Embeddings)** with configurable dimensions
- **SwiGLU activation functions** for MLP layers
- **RMSNorm** for layer normalization
- **Flash Attention** for efficient attention computation

### Flexible Subspace Configuration

We implemented the ability to configure subspaces not only on the attention layer but also:
- **MLP layers** 
- **Vocabulary embeddings**

The implementations are in HuggingFace-compatible format with configuration classes that allow detailed specification of subspace dimensions across all model components.

### Experimentation Infrastructure

The project includes a complete training and evaluation pipeline:

- **Pre-training scripts** for language modeling
- **Fine-tuning scripts** for evaluation on SST-2 sentiment classification
- **Dataset pre-processing and sharding code** for 1% and 2% samples of the C4 dataset
- **Google Colab notebooks** for running experiments at low cost on cloud GPUs (by cloning the repo and running scripts from within the notebook)

### Integration with ML Tools

We integrated the framework with modern ML tooling:

- **Weights & Biases** for thorough experiment tracking and metrics visualization
- **lm-eval** for standardized language model evaluation
- **Flash Attention** for efficient training

### Documentation

The project journey, including many insights and struggles along the way, is captured in the `journals/` folder, providing a detailed record of the research process.

## Project Components

This repository is organized into three main components that follow the research narrative:

### 1. `fused_attn_svd/`

Before building new models, we first analyzed existing ones. This component performs Singular Value Decomposition (SVD) analysis on the attention weight matrices of large, pre-trained MLA models (DeepSeek-V3, Kimi-K2) to measure their "effective rank." The goal was to determine if the output heads already exhibit a low-rank structure that would suggest a shared subspace is feasible.

The analysis revealed that while there is some potential for rank reduction, especially in the early layers, simply decomposing the weights of a pre-trained model might not be the most effective approach. This motivated pre-training models with the output subspace constraint from the beginning.

### 2. `subspace_encoder/`

This component implements a custom Transformer encoder from scratch to experimentally validate the impact of a shared output latent space. We trained small (6-layer, 13M parameter) models on WikiText-103 and evaluated them on the SST-2 GLUE task.

The experiments compared three architectures:

1.  **MHA**: A standard Multihead Attention baseline
2.  **MLA**: Our implementation of Multihead Latent Attention
3.  **MLA-o**: Our proposed variant, MLA with a shared output latent space

### 3. `subspace_decoder/`

Building on the encoder experiments, this component implements and evaluates the shared output latent space using a decoder architecture. We built custom decoder implementations and also patched HuggingFace's DeepSeek-V3 implementation to add the output subspace decomposition.

Models were pre-trained on WikiText-103 and fine-tuned on SST-2, with experiments conducted at different model scales (from 13M to 124M parameters) and at various sequence lengths (128 to 1,024 tokens).

## Experimental Results

We conducted extensive experiments with both encoder and decoder architectures across multiple model scales.

### Small-Scale Results (13-16M Parameters)

**Encoder Results (SubspaceEncoder)**

The table below shows the best-performing encoder configurations evaluated on SST-2 test accuracy:


| # | Attention | Test Accuracy | Parameters | Query Latent | Key-Value Latent | Output Latent | Position Encoding | # of RoPE Dims |
|:-:|:---------:|:-------------:|:----------:|:------------:|------------------|---------------|:-----------------:|:--------------:|
| 1 | MHA       | 85.67         | 13.50M     | n/a          | n/a              | n/a           | RoPE              | 32             |
| 2 | MLA       | 84.75         | 12.67M     | 64           | 32               | n/a           | RoPE              | 16             |
| 3 | MLA-o     | 84.63         | 12.48M     | 64           | 32               | 64            | RoPE              | 32             |


**Decoder Results (Sequence Length 1,024)**

|| Attention | Test Accuracy | Parameters | Query Latent | Key-Value Latent | Output Latent | Perplexity (WikiText-103) |
|:---------:|:-------------:|:----------:|:------------:|------------------|---------------|:-------------------------:|
|| MLA       | 87.96         | 16.26M     | 96           | 64               | n/a           | 28.89                     |
|| MLA-o     | 86.24         | 16.17M     | 96           | 64               | 96            | 29.33                     |

**Key Observations:**

  * **Consistency Across Architectures**: Both encoder and decoder experiments show that any form of decomposed attention (MLA or MLA-o) underperforms dense MHA by 1-2 percentage points
  * **No Speed Benefits**: At these model scales, the reduction in FLOPs from subspace decomposition did not translate to noticeable throughput improvements
  * **Accuracy Trade-off**: The latent space constraints consistently harmed model accuracy without providing compensating benefits

### Larger-Scale Results (124M Parameters - GPT-2 Scale)

At the 124M parameter scale, we began to see the first signs of speed improvements from the decompositions, though accuracy remained a challenge. This suggests that the computational benefits of latent attention may only become apparent at larger model sizes.

## Potential Future Work

While this project has concluded, there remain interesting open questions for future research:

  * **Larger-Scale Experiments**: Testing at much larger model scales (billions of parameters) where the computational benefits of latent attention may fully materialize
  * **Isoparametric Comparisons**: Conducting experiments where all variants have exactly the same parameter count to eliminate capacity as a confounding variable
  * **Training Optimizations**: Exploring whether specialized training procedures, learning rate schedules, or initialization schemes could make latent attention more effective at smaller scales
  * **Architectural Variations**: Investigating different decomposition patterns or hybrid approaches that selectively apply subspace constraints

The framework and codebase developed in this project provides a solid foundation for anyone interested in pursuing these directions.

## Repository Structure

```
.
├── fused_attn_svd/      # SVD analysis of pre-trained models.
│   ├── Calculating Singular Values in Large MLA Models.ipynb
│   └── Plotting Effective Rank of Attention Matrices.ipynb
│
├── subspace_encoder/    # Experimental encoder model implementation.
│   ├── configs/         # Model and training hyperparameters.
│   ├── scripts/         # Scripts for pre-training and fine-tuning.
│   ├── models/          # The SharedSubspaceEncoder model definition.
│   └── run_experiments.ipynb  # Notebook for running experiments and analyzing results.
│
├── subspace_decoder/    # Decoder experiments using patched DeepSeek-V3.
│   ├── configs/         # Model and training configurations.
│   ├── layers/          # Output subspace patching and implementations.
│   ├── scripts/         # Training and fine-tuning scripts.
│   └── run_experiments.ipynb  # Notebook for decoder experiments.
│
├── journals/            # Research notes and experiment documentation.
│   └── 2025-09-02 - Initial Decoder Experiments.ipynb
│
├── .gitignore
└── README.md            # You are here!
```

## Using This Repository

This repository can serve as a learning resource and starting point for your own transformer architecture experiments.

### Exploring the Code

The codebase provides a working example of:
- Modern transformer implementations (encoder and decoder) with RoPE, SwiGLU, and RMSNorm
- Flexible subspace decomposition patterns applied to attention, MLP, and embeddings
- Integration with HuggingFace, Weights & Biases, and lm-eval
- Training scripts for both pre-training and fine-tuning

### Running Experiments

If you'd like to run experiments yourself:

1. Check out the [run_experiments.ipynb](https://github.com/chrisjmccormick/shared-subspaces/blob/main/subspace_encoder/scripts/run_experiments.ipynb) notebook in `subspace_encoder/` - it can be run in Google Colab and will clone the repository and kick off a pre-training run
2. Explore the configuration files in `configs/` to understand the hyperparameters
3. Review the `journals/` folder for insights into the experimental process and findings

### Learning from the Journey

The `journals/` folder contains detailed notes about the research process, including challenges encountered, hypotheses tested, and lessons learned. These may be particularly valuable for understanding what works, what doesn't, and why.

---

This project represents a learning journey into the mechanics of latent attention and the challenges of architectural innovation at small scales. While the specific hypothesis about output subspaces didn't pan out as hoped, the infrastructure built and lessons learned may prove valuable for future research in this space.
