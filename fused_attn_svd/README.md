# SVD Analysis of Fused Attention Head Matrices

This project focuses on calculating and analyzing the singular values (SVD) of the weight matrices in Multihead Latent Attention (MLA) models such as DeepSeek-V3, DeepSeek-R1, and Kimi-K2.

Some key concepts are:

* Fused attention matrices: Fusing Value-Output and Query-Key matrices can reveal lower effective rank than either matrix independently.
* Stacked / Concatenated matrices: If you concatenate all of, e.g., the fused Value-Output matrices in a layer and compute their singular values, this lets you analyze the shared structure across the heads.

These two concepts can be used for analysis, or as part of parameter reduction techniques:

* Truncation: Keep only the top singular vectors.
* Shared subspaces: Break head matrices into smaller per-head matrices followed by a shared one. 

## Notebooks

### 1. Fuse and Rank Reduce - Part 1 - Truncation.ipynb

A preliminary version of a blog post, this explains the overall technique, goes into the theory behind it, and looks at its application to SVD truncation, specifically. (Part 2 will explore shared subspaces).

### 2. Calculating Singular Values in Large MLA Models.ipynb

Computes the singular values for every attention head in an MLA model. Calculates the values for the individual heads, as well as stacked together per-layer (for analyzing shared subspaces). 

It handles:

* Extraction and fusion of the weight matrices
* Disk space management (the models are too large to fit on the 250 GB disk) 
 
Note that I've already calculated the singular values for several models and shared them via huggingface datasets, so you don't necessarily need to run this Notebook. The actual analysis and plots are in the next notebook. 

**Data Structure**

The singular values are stored in dictionaries with the following structure:

```python
# Singular values dictionaries
S_subspaces = {}     # per-layer subspace matrices (e.g., KVA, QA)
S_heads = {}         # per-head singular values for each matrix type
S_stacked_heads = {} # singular values for stacked/concatenated heads
```

**Available Models**

Currently, pre-computed singular values are available for the following models:

* DeepSeek-V3-base
* DeepSeek-R1

You can choose a model by setting the `model_name` variable in the notebooks.

### 3. Plotting Effective Rank of Attention Matrices.ipynb

Provides visualizations of the effective rank of the attention matrices and their fused forms.

The effective rank is computed at 1% and 0.1% error thresholds using the "cumulative energy" metric.

There are two main types of plots:

* Layer-wise: These show the effective rank of the stacked matrices, illustrating shared structure across heads in a layer.

![Example Plot](https://lh3.googleusercontent.com/d/1jM0tRYOOVTyOWiNRAXfl-umO3feFLMRK)

* Head-wise: These are per-layer plots which show the effective rank of the individual heads, typically comparing this to their fused forms.

<img src='https://lh3.googleusercontent.com/d/1A4whsqz0PRgxR0fa3n73XTpSR0XfwUo9' alt='Plot of the effective ranks of the Value, Output, and fused VO matrix in DeepSeek-R1 showing the combined effective rank being dragged down by the value matrix' width='900' />




