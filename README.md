# shared-subspaces

This repository contains research code exploring the use of shared subspaces in Transformer attention and feed-forward networks. The core of this work investigates the impact of adding a shared *output* latent space to Multihead Latent Attention (MLA), a parameter-efficient attention mechanism used in models like DeepSeek and Kimi.

The projects here include Singular Value Decomposition (SVD) analysis of pre-trained models to motivate the research, as well as experiments with custom, latent-space-efficient Transformer variants.

## The Research Question: Constraining the Residual Stream

State-of-the-art Multihead Latent Attention (MLA) models like DeepSeek-V3 aggressively bottleneck the inputs to the attention layer. For instance, they project the model's hidden dimension (e.g., 7,168) down to much smaller latent spaces for the query (e.g., 1,536) and key/value pairs (e.g., 512).

This raises a key question: **If these input bottlenecks are effective, what is the impact of adding a similar bottleneck to the *output* of the attention layer?**

Using the language of mechanistic interpretability, we can think of a token's vector representation as a "residual stream"—a communication bus that all model components read from and write to. In this framing, MLA's input projections constrain how much information each head can *read* from the stream. This project explores constraining where they can *write* to.


<img src='https://lh3.googleusercontent.com/d/1Hh95gVcbyyWpn-vNo6Mx4equVBdjjLcZ' alt='Simple block diagram of the attention heads with shared spaces illustrated as trapezoids' width='800' />

_The trapezoids in this illustration represent projections shared by all heads in a layer. Multihead Latent Attention defines a shared Key-Value latent projection (bottom) and a larger, shared Query latent projection (top). We're proposing a shared Output latent projection (right)._
<font size="-.5"><center></center></font>


A shared output subspace, where the output matrix $W^O$ is decomposed into a per-head projection $W^{OA}\_i$ and a shared projection $W^{OB}$, could have competing effects:

  * **Potential Benefits:** It could encourage shared learning and feature reuse, as the shared projection receives gradient updates from every token.
  * **Potential Risks:** It could reduce head diversity or lead to destructive interference as heads compete for representation capacity in a smaller space.

This repository documents the investigation into whether, under the right conditions, an output latent space can be beneficial.

## Projects

This repository is organized into two main projects that follow the research narrative.

### 1. `fused_attn_svd/`

Before building new models, we first analyzed existing ones. This project performs a Singular Value Decomposition (SVD) analysis on the attention weight matrices of large, pre-trained MLA models (DeepSeek-V3, Kimi-K2) to measure their "effective rank." The primary goal was to see if the output heads already exhibit a low-rank structure that would suggest a shared subspace is feasible.

The analysis reveals that while there is some potential for rank reduction, especially in the early layers, simply decomposing the weights of a pre-trained model might not be the most effective approach. This motivated pre-training a model with the output subspace constraint from the beginning.

Dive into the analysis in the `fused_attn_svd/README.md`.

### 2. `subspace_encoder/`

This project implements a custom Transformer encoder from scratch to experimentally validate the impact of a shared output latent space. We train small (6-layer, 13M parameter) models on WikiText-103 and evaluate them on the SST-2 GLUE task.

The core experiments compare three architectures:

1.  **MHA**: A standard Multihead Attention baseline.
2.  **MLA**: Our implementation of Multihead Latent Attention.
3.  **MLA-o**: Our proposed variant, MLA with a shared output latent space.

Find the implementation, usage, and full experimental details in the `subspace_encoder/README.md`.

## Current Status & Preliminary Results

Experiments were run using 6-layer encoders with a hidden dimension of 256 and 8 attention heads. The table below shows the best-performing configurations for each variant found so far, evaluated on SST-2 test accuracy.


| # | Attention | Test Accuracy | Parameters | Query Latent | Key-Value Latent | Output Latent | Position Encoding | # of RoPE Dims |
|:-:|:---------:|:-------------:|:----------:|:------------:|------------------|---------------|:-----------------:|:--------------:|
| 1 | MHA       | 85.67         | 13.50M     | n/a          | n/a              | n/a           | RoPE              | 32             |
| 2 | MLA       | 84.75         | 12.67M     | 64           | 32               | n/a           | RoPE              | 16             |
| 3 | MLA-o     | 84.63         | 12.48M     | 64           | 32               | 64            | RoPE              | 32             |


**Key Observations:**

  * The standard MHA baseline currently achieves the highest accuracy.
  * Adding the shared output space (MLA-o) results in a slight drop in accuracy compared to standard MLA, while reducing parameter count by ~1.5%.
  * At this small scale, MLA-o is slower in training throughput, likely due to the overhead of an additional matrix multiplication.

These results are preliminary. Further exploration is needed to understand the trade-offs and identify scenarios where an output latent space could be advantageous.

## Future Directions & Collaboration

This is an active research project, and I welcome feedback, discussion, and collaboration! Some potential next steps include:

  * **Decoder Experiments:** The most direct test would be to add the output subspace to a well-known implementation like HuggingFace's `DeepSeekV3Attention` and run pre-training experiments.
  * **Throughput Analysis:** Systematically benchmark the performance (samples/sec) of an isolated attention layer to find the model/latent sizes where MLA-o becomes more computationally efficient.
  * **Hyperparameter Sweeps:** Thoroughly explore the impact of different latent space sizes for the query, key-value, and output projections.
  * **Subspace Alignment:** An interpretability tangent to investigate if the output heads align with other subspaces in the model.

If you are interested in these ideas, please feel free to open an issue or a pull request to discuss them further, or join the discussion in the Community Projects channel of the EleutherAI Discord server, [here](https://discord.com/channels/729741769192767510/1403903562706059334).

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
├── .gitignore
└── README.md            # You are here!
```
