# 1. Overview
-------------

We are evaluating the viability of modifications to the Transformer architecture relating to shared subspaces. 

The current topics to explore, in priority order:

1. The addition of a shared output latent space to Multihead Latent Attention (MLA).

2. The use of MLA in tandem with decomposed FFNs, to determine whether they work better together than with standard MHA.

3. Factorized vocabulary embeddings via a shared projection layer.

# 1.1. Goals
------------

The plan is to eventually partner with a company or institution to fully evaluate these techniques and publish on them.

The purspose of this current project, then, is to do smaller scale experiments aimed at demonstrating the potential (provided it's there) in order to attract outside interest. 

# 1.2. Pivoting
---------------

We've been working on the "custom_bert.py" model, where we tried to build this project using stripped down versions of classes from huggingface transformers. 
We're pivoting toward building a new, lighter weight implementation from scratch, in /encoder-pretrain/models/shared_subspace_encoder.py
It's a research prototype, meant to focus on communicating a new idea. The code is explanatory.
We'll keep around the original classes for now as reference.
The intended features are the same, except now we are focusing on strictly a decoder model. 
We still want it to be compatible with huggingface transformers.


# 2. Background
---------------

## 2.1. Multihead Latent Attention

Multihead Latent Attention (MLA) was introduced in 2024 by a company named DeepSeek. They released a frontier-scale model called DeepSeek-V3 in early 2025 which made waves for being both performant and more efficient to train.

MLA introduces:
- A projection to a per-layer KV-latent space. It is shared by all heads in a layer.
- A projection to a query latent space, also shared by all heads in a layer.
- Separate of position information into a separate Multi-Query attention mechanism with one key head.

Because it was first introduced by DeepSeek, the corresponding model code in huggingface transformers is called, e.g., "DeepSeekV3Attention". This is synonymous with MLA.

## 2.2. Output Latent Space

The proposal in this project is to complete the symmetry and introduce an additional latent space for the output heads. 

"Shared output projection" can be a confusing term, because the $W^O$ matrix is often misunderstood as already being a "shared projection", when in fact it is per-head.

Just as the QKV matrices contain concatenated input heads, $W^O$ contains concatenated output heads. We don't notice them because they don't need to be split apart like the input heads. Instead, we concatenate the (scored) value vectors $v_i$ from each head and perform:

$$
vW^O = [v_1 \; v_2 \; \dots \; v_h]
\begin{bmatrix}
W^O_1 \\
W^O_2 \\
\vdots \\
W^O_h
\end{bmatrix}
= \sum_{i=1}^{h} (v_i W^O_i)
$$

This combines two steps into one--the independent, per-head output projection, and the summation across the heads.


## 2.3. Equations

Here are the updated equations for a single token vector $x \in \mathbb{R}^{d_\text{model}}$ passing through a Multihead Latent Attention (MLA) layer with the proposed Output latent space. I'm excluding the RoPE heads for now for brevity. 

(Note that there is a summary table of the variables with example dimensions at the end).


**1. Shared Latent Projections**

First, the input token vector $x$ is projected down into two smaller, shared latent spaces: one for the Queries and one for the Keys and Values.

$$
c_q = x W^{QA} \quad \in \mathbb{R}^{d_\text{latent_q}} \\
c_{kv} = x W^{KVA} \quad \in \mathbb{R}^{d_\text{latent_kv}}
$$

Where

$$
W^{QA} \in \mathbb{R}^{d_\text{model} \times d_\text{latent_q}} \\
W^{KVA} \in \mathbb{R}^{d_\text{model} \times d_\text{latent_kv}}
$$



**2. Per-Head Projections**

Next, the latent vectors $c_q$ and $c_{kv}$ are used as input to the individual attention heads. For each head $i$ out of $h$ total heads:


$$
q_i = c_q W^{QB}_i \\
k_i = c_{kv} W^{KB}_i \\
v_i = c_{kv} W^{VB}_i
$$

Where:

$$
W^{QB}_i \in \mathbb{R}^{d_\text{latent_q} \times d_\text{query}} \\
W^{KB}_i \in \mathbb{R}^{d_\text{latent_kv} \times d_\text{key}} \\
W^{VB}_i \in \mathbb{R}^{d_\text{latent_kv} \times d_\text{value}}
$$

**3. Scaled Dot-Product Attention**

Attention scores are calculated for each head $i$ using the standard scaled dot-product attention mechanism. The output of this step for head $i$ is a scored value vector $z_i \in \mathbb{R}^{d_\text{value}}$:

$$
z_i = \text{softmax}\left(\frac{q_i k_i^\top}{\sqrt{d_\text{key}}} \right) v_i
$$

**4. Output Projections**

The attended value vectors $z_i$ from all heads are concatenated and then projected into the output latent space using the projection matrix $W^{OA} \in \mathbb{R}^{h \cdot d_\text{value} \times d_\text{latent_o}}$:

$$
c_o = \text{Concat}(z_1, z_2, \dots, z_h) W^{OA} \quad \in \mathbb{R}^{d_\text{latent_o}}
$$

This operation combines the per-head up projection and the summation across heads. It is equivalent to:

$$
c_o = \sum_{i=1}^{h} \left(z_i W^{OA}_i \right)
$$

Where $W^{OA}_i \in \mathbb{R}^{d_{value} \times d_\text{latent_o}}$ is the per-head projection into output latent space.


Finally, the ouput latent is re-projected back into model space via a shared projection, giving the final output of the attention layer:

$$
o = c_oW^{OB} \quad \in \mathbb{R}^{d_\text{model}}
$$

where $W^{OB} \in \mathbb{R}^{d_\text{latent_o} \times d_\text{model}}$ is the shared output projection into model space.



**Summary of Variables**

Example dimensions are taken from DeepSeek-V3 (except for the output latent dimension).

| Variable                                            | Description                      | Example Dimensions    |
| --------------------------------------------------- | -------------------------------- | --------------------- |
| $x$                                               | Input token vector               | $1 \times 7168$     |
| $d_\text{model}$                                 | Model (token vector) dimension   | $7168$                  |
| $d_\text{latent_q}$                             | Query latent space dimension     | $1536$                  |
| $d_\text{latent_kv}$                            | Key/Value latent space dimension | $512$                   |
| $d_\text{latent_o}$                            | Output latent space dimension | $1280$                   |
| $d_\text{query}, d_\text{key}, d_\text{value}$ | Per-head dimensions              | $128$                   |
| $h$                                               | Number of heads                  | $128$                   |
| $W^{QA}$                                          | Shared Query latent projection     | $7168 \times 1536$  |
| $W^{KVA}$                                         | Shared Key/Value latent projection | $7168 \times 512$   |
| $W^{QB}_i$                                       | Per-head Query projection        | $1536 \times 128$   |
| $W^{KB}_i$                                       | Per-head Key projection          | $512 \times 128$    |
| $W^{VB}_i$                                       | Per-head Value projection        | $512 \times 128$    |
| $W^{OA}_i$                                             | Per-head Output latent projection          | $128 \times 1280$ |
| $W^{OB}$                                             | Shared Output projection          | $1280 \times 7168$ |

## 2.4. In Code
---------------

```python

# ... preparation of queries, keys, and values ...

# Run attention
attn_output, attn_weights = attention_interface(
	self,
	query_states,
	key_states,
	value_states,
	attention_mask,
	dropout=0.0 if not self.training else self.attention_dropout_prob,
	scaling=self.scaling,
	**kwargs,
)

if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
	attn_output = attn_output[:, :, :, : self.v_head_dim]

attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()


# ------------------------------------------------------
# Modified version: Adding an intermediate latent space.

if self.output_subspace:
	# First, project the scored value vectors onto `o_a_proj`. This is
	# equivalent to projecting onto W^O in standard attention, except 
	# that here we are projecting into an intermediate latent space. 
	# This projection is unique per-head, preserving head diversity, and
	# then sums the results into a single vector per token.
	attn_output = self.o_a_proj(attn_output)

	# MLA uses RMSNorm on the query and key-value latents. It's not
	# clear yet whether this is helpful for the output.
	#attn_output = self.o_a_layernorm(attn_output)

	#print(f"attn_output after o_a_proj: {attn_output.shape}")

	# The input to `o_b_proj` is the summed output latents of the 
	# attention heads. This step re-projects this single per-token 
	# latent back to model space.
	attn_output = self.o_b_proj(attn_output)

# ----------------------------------------- 
# Original: Standard W^O output projection.

else:
	attn_output = self.o_proj(attn_output)

# -----------------------------------------

return attn_output, attn_weights

```

## 2.5. Shared Vocabulary Subspace
When the model hidden size is small, the vocabulary embedding matrix dominates
the parameter count. Tokens first map to a low-rank latent vector which is then
projected into model space through a shared linear layer. Reusing this projection for
the output logits ties the input and output embeddings together. Set ``vocab_decompose=True``
and adjust ``vocab_rank`` to enable this factorization.

# 3. Experiments
----------------

# 3.1. Observations from Vision Transformer Experiment

Initial from-scratch pre-training runs on a small Vision Transformer model on CIFAR-10 were promising, suggesting that the addition of the output latent improves both the accuracy and efficiency of MLA.

- An output latent size of similar scale to the existing query and key-value latents appeared to be sufficient (i.e., the output doesn't appear to need a significantly larger space).
- The best performing runs of the proposed architecture in the ViT experiment used a decomposed FFN--these outperformed dense MLPs for the MLA variants.
    - Standard MHA still appeared to work best with dense MLPs, though this wasn't explored as thoroughly.
- Adding one or more dense layers to the beginning of the architecture, similar to the practice with MoE models, improved performance.
    - In particular, using dense FFNs _and standard MHA_ together in these early layers was beneficial. 
	    - This is distinct from the DeepSeek-V3 architecture, which uses 3 initial layers of dense FFNs but MLA attention in all layers.
	- This is an interesting finding on its own, I think.

# 3.2. Encoder Architecture

- The current experiment which we are working on right now is to train a text encoder model from scratch to evaluate the performance impact of these different changes.

# 3.3. Hardware

- We are building the experiment as python scripts, and these are being run from within a Google Colab Notebook.
- The Colab instance is connected to a 40GB A100.

