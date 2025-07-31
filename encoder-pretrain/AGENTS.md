# Overview

We are evaluating the viability of modifications to the Transformer architecture relating to shared subspaces. 

The current topics to explore, in priority order:

1. The addition of a shared output latent space to Multihead Latent Attention (MLA).

2. The use of MLA in tandem with decomposed FFNs, to determine whether they work better together than with standard MHA.

3. Factorized vocabulary embeddings via a shared projection layer.

# Goals

The plan is to eventually partner with a company or institution to fully evaluate these techniques and publish on them.

The purspose of this current project, then, is to do smaller scale experiments aimed at demonstrating the potential (provided it's there) in order to attract outside interest. 

# Pivoting

We've been working on the "custom_bert.py" model, where we tried to build this project using stripped down versions of classes from huggingface transformers. 
We're pivoting toward building a new, lighter weight implementation from scratch, in /encoder-pretrain/models/shared_subspace_encoder.py
It's a research prototype, meant to focus on communicating a new idea. The code is explanatory.
We'll keep around the original classes for now as reference.
The intended features are the same, except now we are focusing on strictly an encoder model. 
We still want it to be compatible with huggingface transformers.

The prior implementation, which we're no longer using, but can be referenced:
encoder-pretrain/models/custom_bert.py
encoder-pretrain/models/layers/mla_attention.py

# Style Guide

## Configuration Values

- Keep assumptions explicit with ``assert`` and fail fast if a required
  field is missing.
- Avoid inferring shapes or configuration defaults; raise an error instead
  of silently guessing.
- Comment code that relies on HuggingFace conventions so newcomers can
  follow along.
- Boilerplate should be minimal and clearly marked.

- This repository uses a deliberately brittle `.json` config approach which
  strictly requires all possible config values to be specified. For example,
  the pre-training script performs the following at the top:

```python

    # Load the config file.    
    with open(args.config) as f:
        config = json.load(f)

    # Initialize the optional stats dictionary so later assignments don't fail.
    # Configuration files in this repo don't define a "stats" key yet.
    if "stats" not in config:
        config["stats"] = {}

    # Strict key check on the model configuration.
    valid_keys = SharedSubspaceEncoderConfig.__init__.__code__.co_varnames
    valid_keys = set(valid_keys) - {"self", "kwargs"}
    extra_keys = set(config["model"]) - valid_keys
    if extra_keys:
        raise ValueError(f"Unknown keys in config: {sorted(extra_keys)}")

    # Will raise TypeError, by design, if required args are missing
    model_config = SharedSubspaceEncoderConfig(**config["model"])
```  

- At this stage in development, it's more important to expose oversights 
  and ommisions than to worry about user convenience.

## Symbolic Tensor Dimensions

At the beginning of each function, map symbolic dimension names to their meanings:

```
# === Tensor Dimension Symbols ===
#  B: batch_size     — number of samples in the batch
#  T: seq_len        — number of tokens per sample
#  H: n_heads        — number of attention heads
#  D: hidden_dim     — model embedding size
# Dh: head_dim       — per-head projection dimension
#  C: latent_dim     — latent / subspace size
#  R: rope_dims      — number of head dims rotated 
```

(These are just examples--update the table to match what's used in the function.


## Block Commenting

Use large section headers for major steps, preceded by a short description of the purpose of that block.

```
# ==============================
#     Query Compression
# ==============================
```

## Shape Comments

Place comments above an operation detailing input and output shapes.

```python
# Linear projection of queries
# Input:  
#            x [B,  T,  D]
#       q_proj         [D, H*Dh]
# Output: 
#            q [B, T, H * Dh]
q = self.q_proj(x)
```

For reshape or view operations, indicate the transformation with arrows:

```python
# Reshape: [B, T, H * Dh] → [B, T, H, Dh]
q = q.view(B, T, H, Dh)
```

## Slice Assignments

Prefer multi-line slice assignments so that each side can have shape comments.

## Einsum Template

When using `einsum`, provide the symbolic string, shapes of inputs, and the meaning of the operation.

TODO - Update with an example from the code.

```
# === Project query into key space ===
# Operation:
#   einsum("bshd,hdc->bshc", q_nope, W^K)
# Inputs:
#   q_nope: [B, S, H, Dq]
#      W^K:       [H, Dq, C]
# Output:
#   q_proj: [B, S, H, C]
# Dropped dim: Dq
# Broadcasted: H
```

# Tasks

## Standardize on use of `latent`

Update `/encoder-pretrain/models/shared_subspace_encoder.py` to change from, e.g., `q_lora_rank` to `q_latent_dim`:

```python
        #   Cq: q_latent_dim   - query latent subspace size
        #  Ckv: kv_latent_dim  - key-value latent subspace size
        #   Co: o_latent_dim   - output latent subspace size
```

Leave other files alone for now. 
You can update `/encoder-pretrain/tests/test_shared_encoder.py` if this change causes errors.

## Rotary Embeddings

Define the RotaryEmbedding class.
To keep everything on the GPU and avoid repeated dtype/device transfers, 
precompute cos and sin once and cache them as buffers.

## Update shape docs

Many lines are missing / misapplying the style guide's shape definitions. I've marked many of these with "TODO".

## Full config

Review `SharedSubspaceEncoderConfig` for missing properties.