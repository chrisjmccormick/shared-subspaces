# SVD Analysis of Fused Attention Head Matrices

This project focuses on calculating and analyzing the singular values (SVD) of the weight matrices in Multihead Latent Attention (MLA) models such as DeepSeek-V3, DeepSeek-R1, and Kimi-K2.

The initial focus was on evaluating the output space of the attention heads in order to determine the feasibility of creating a shared output subspace for the attention heads. The notebooks cover all of the attention matrices, however.

## Project Overview

The analysis is divided into two notebooks:

### 1. Calculating Singular Values in Large MLA Models.ipynb

This notebook implements the computation of the singular values, and discusses the theory behind some of the analysis, e.g.:

* Fused attention matrices: Fusing Value-Output and Query-Key matrices can reveal lower effective rank than either matrix independently.
* Stacked / Concatenated matrices: If you concatenate all of, e.g., the fused Value-Output matrices in a layer and compute their singular values, this lets you analyze the shared structure across the heads.
 

The notebook addresses some of the engineering challenges in calculating these values--namely, that the models are enormous!
 
I've already calculated the singular values for several models and shared them via huggingface datasets, so you don't necessarily need to run this Notebook. 
 
The actual analysis and plotting of the results is done in a separate notebook. 


### 2. Plotting Effective Rank of Attention Matrices.ipynb

This notebook provides visualizations of the effective rank of the attention matrices and their fused forms.

The effective rank is computed at 1% and 0.1% error thresholds using the "cumulative energy" metric.

There are two main types of plots:

* Layer-wise: These show the effective rank of the stacked matrices, illustrating shared structure across heads in a layer.

![Example Plot](https://lh3.googleusercontent.com/d/1jM0tRYOOVTyOWiNRAXfl-umO3feFLMRK)

* Head-wise: These are per-layer plots which show the effective rank of the individual heads, typically comparing this to their fused forms.


---

## Data Structure

The singular values are stored in dictionaries for easy access:

```python
# Singular values dictionaries
S_subspaces = {}     # per-layer subspace matrices (e.g., KVA, QA)
S_heads = {}         # per-head singular values for each matrix type
S_stacked_heads = {} # singular values for stacked/concatenated heads
```

### Available Models

Currently, pre-computed singular values are available for the following models:

* DeepSeek-V3-base
* DeepSeek-R1

You can choose a model by setting the `model_name` variable in the notebooks.

