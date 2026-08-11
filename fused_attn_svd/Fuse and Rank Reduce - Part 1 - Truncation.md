<!-- md -->
<a href="https://colab.research.google.com/github/chrisjmccormick/shared-subspaces/blob/main/fused_attn_svd/Fuse%20and%20Rank%20Reduce%20-%20Part%201%20-%20Truncation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- code -->
```python

```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Fuse and Rank Reduce (FaRR)

_Finding Hidden Low-Rank Structure in Attention_

<!-- md -->
Most SVD-based compression methods start by taking **one matrix** and breaking it into **two**.
That makes each piece smaller — but you’ve now added an extra multiply.
In fact, the parameter count *goes up* until you can remove enough singular components to offset the decomposition.

Fuse and Rank Reduce (FaRR) works differently.
Instead of introducing an extra step, it starts from **two matrices that are already multiplied together in the model** and recomposes them into a single fused matrix.
When you truncate this fused matrix and split it back apart, you keep the *same* number of multiplies as before — but with fewer parameters.

There are **two important caveats**, in addition to the standard quality–vs.–compression trade-off:

1. **It’s not always there** — the kind of low-rank structure you can exploit this way is the exception, not the norm.
2. **It can be hard to leverage without breaking parallelism** — attention heads are shaped for batch-friendly execution, and mismatched head sizes can erase your gains.

The method applies anywhere in a model where you have a **linear map** *(i.e., two matrices multiplied together with no nonlinearity in between)*.
In transformers, the prime examples are:

* **Value–Output** in the attention write path.
* **Query–Key** in the attention score path.

There are two opportunities to leverage FaRR that we'll explore:

1. **Rank reduction** — Fuse two matrices, measure the effective rank, truncate, and split them back into smaller matrices.
   In Part 1, we’ll show exactly how to do this and explore where the opportunity exists in the Kimi-K2 model.
2. **Shared subspace identification** — Fuse matrices to reveal where multiple heads use overlapping latent spaces.
   In Part 2, we’ll dig into *why* fused matrices sometimes have much lower rank, and use that insight to locate shared subspaces for even more parameter savings.

<!-- md -->
## S1. SVD Review & Notation

<!-- md -->
Before we start fusing matrices, let’s review what Singular Value Decomposition (SVD) actually *does* — in an intuitive, vector-level way.

<!-- md -->
### 1.1. Vector Formulation

<!-- md -->


When applied to a weight matrix \$W\$, SVD finds a set of vector pairs \$(u_i, v_i)\$, which, along with a scalar value \$\sigma_i\$, fully capture the behavior of the original matrix:

$$
W = \sum_{i} \sigma_i \, u_i \, v_i^\top
$$

* \$u_i\$ and \$v_i\$ are called **singular vectors**.
* \$\sigma_i\$ is called their **singular value**.

The vectors are **unit length**, meaning they purely capture **direction**.
They are sorted by their \$\sigma\$ value, which tells us how relevant they are to the matrix’s behavior and how important they will be in preserving its full functionality.

For example:

* If we take the direction \$v_0\$ and project it through \$W\$, we get:

$$
v_0 W = \sigma_0 u_0
$$

Here \$v_0\$ is the **input direction** that produces the strongest possible response from \$W\$ (given a normalized input).
SVD separates that response into its **magnitude** (\$\sigma_0\$) and its **direction** (\$u_0\$).

In matrices with low effective rank, the last singular vector, $v_d$ will have $\sigma_d$ near zero, meaning that direction produces almost no response from $W$.

Those trailing directions — with very small $\sigma$ — are the ones we can potentially drop with minimal impact to the matrix’s behavior.

_Unit Length and Orthogonal_

The vectors $v_i$ all have unit length and each one is orthogonal to every other. Their matrix is "orthonormal".

The same is true within the collection of $u_i$ vectors.

<!-- md -->
### 1.2. Matrix Formulation

<!-- md -->
The terminology used in the formal definition of SVD is problematic in the context of Transformers because it explicitly places the input on the right hand side of the equation (e.g., $Ax$) whereas the definition of Self-Attention places it on the left (e.g., $xW^Q$).

To bring the definition inline with out context, we will:

1. Define a "variant" of SVD which accepts and returns standard Attention variable names and shapes.
2. Adopt the LoRA-style convention of "$A$" for the input matrix and "$B$" for the output matrix.
3. Define the singular values as a vector $\sigma$ instead of a diagonal matrix $\Sigma$. This aligns with the implementation in deep learning libraries such as PyTorch.

<!-- md -->

**SVD Applied to Attention Heads**

Let $\text{SVD}_\text{QKV}(\cdot)$ define a function which decomposes an individual query, key, or value head. Using a query head to illustrate,

$$
\text{SVD}_\text{QKV}(W^Q) = W^{QA} \, \sigma^Q \, W^{QB}
$$

Where:

* $W^{QA} \in \mathbb{R}^{d_\text{model} \times r} \quad $ contains orthonormal column vectors,
* $\sigma^Q \in \mathbb{R}^{r} \quad \quad \quad $ contains their singular values, and
* $W^{QB} \in \mathbb{R}^{r \times d_\text{head}} \quad $ contains orthonormal row vectors.

<!-- md -->

**Element-wise Multiplication**

In our notation, the multiplication operation with $\sigma$ is inferred by its placement. For example,
* When multiplied with the matrix to its left, $(W^{QA} \sigma^Q)$, the singular values are applied element-wise to the columns.
* When multiplied with the matrix to its right, $(\sigma^Q W^{QB})$, the singular values are applied element-wise to the rows.

<!-- md -->

**In PyTorch**

All of this aligns well with how we'll express the operation in our PyTorch examples:

```python
# ======== Variable Shapes ========
#       W_q  (d_model, d_head)
#      W_qa  (d_model, r     )  
#      W_qb  (      r, d_head)
#   sigma_q  (r,)
#         x  (d_model,)
# Where, initially, r = d_head.
W_qa, sigma_q, W_qb = torch.linalg.svd(W_q)

# Examples of how to project an input vector `x` into query space:

# 1. Original matrix
q1 = x @ W_q

# 2. Apply sigma directly to x
q2 = (x * sigma_q) @ W_qa @ W_qb

# 3. Fold σ into the columns of W_qa
q3 = x @ (W_qa * sigma_q.unsqueeze(0)) @ W_qb

# 4. Fold σ into the rows of W_qb
q4 = x @ W_qa @ (W_qb * sigma_q.unsqueeze(1))
```

<!-- md -->
### 1.3. SVD Truncation

<!-- md -->
As an example, below is the decomposition of Query head number 50 from layer 13 of Moonshot's Kimi-K2--many of the Query heads in this layer have low effective rank.

Kimi-K2 uses a query latent space of length 1,536, and a head size of 128.

The below heatmaps show all of the floating point values in the respective matrices. After decomposing $W^Q_{50}$, we have $W^{QA}$, $\sigma^Q$, and $W^{QB}$.

Below are the contents of ${W^{QA}}^\top$.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1kTHmXdBKyPfOMsj0-9LUjtcgaGzDiSga' alt='Heatmap showing the full contents of the Wqa portion of the decomposition of layer 13 query head 50' width='900' />

<!-- md -->
> Aside: The striation pattern was surprising to see; it's not an effect of the decomposition--the original head seems to have the same pattern. Decomposition only normalizes the head dimensions of this matrix (the 128 rows in this plot are normalized, but not the columns). It's something we can come back to in part 2 when we look at the concept of shared subspaces.

Below is the other half of the decomposition, $W^{QB}$, with no perceivable patterns. The row vectors in this matrix are sorted by the magnitude of sigma, meaning we'll remove rows from the bottom if we do any truncating.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1i6LRvgyEEykmko3-rkg7kIaSgr-VKFO4' alt='Heatmap showing the contents of the 128 x 128 W_qb matrix of Kimi-K2 Layer 13 query head 50. Looks like noise, no pattern.' width='600' />

<!-- md -->

Next, we can visualize the singular values for this query head.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1jmG0N4gCPL93XRIRlkxP_1orELFPIKt8' alt='Line plot showing the singular values of Kimi-K2 Layer 13 query head 50. This is a very low-rank head with the elbow around 64.' width='600' />

<!-- md -->
Each sigma value corresponds to a column in $W^{QA}$ and a row in $W^{QB}$.

The 128 columns of $W^{QA}$ can be viewed as directions in the residual stream from which this query head will read.

The 128 rows of $W^{QB}$ are directions in a 128-dimensional space along which this head will write those values.

In between those two, however, are these sigma values, which capture how much the head will amplify or suppress/ignore the respective read/write pair.

This plot is showing us that this query head is largely ignoring roughly half of its directions.

We can visualize the effect of these sigma values by multiplying them with their corresponding row of $W^{QB}$.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/1zr1vjTQXjVOrnZXMC3dpabxRrAhoJRyQ' alt='Heatmap showing the contents of the 128 x 128 Wqb matrix after scaling with sigma, from Kimi-K2 Layer 13 query head 50. This low rank head becomes largely white.' width='600' />

<!-- md -->
The contrast has increased for the top rows, meaning they've been amplified, and many of the lower vectors are nearly 0.

With this folding, $W^{QA}$ may read information along all 128 of its residual stream directions, but half those values will be projected onto vectors which have nearly zero magnitude, erasing what's there.

This is what SVD Truncation takes advantage of--we can trim off these directions which seem to be contributing very little.

A common heuristic for choosing how many vectors to remove is to measure the "cummulative energy".

<!-- md -->
### 1.4. Effective Rank via Cumulative Energy

<!-- md -->
A common method for quantifying the "effective rank" of a matrix (and deciding how many dimensions to drop / keep), is to keep the smallest $r$ such that the retained singular values account for some fraction of the total “energy”:

$$
\frac{\sum_{j=1}^r \sigma_j^2}{\sum_{j=1}^{d_v} \sigma_j^2} \ge \tau
$$

with $\tau$ set to something like $0.99$ or $0.999$ (99% or 99.9%).

Here is that metric applied to query head 50.

<!-- md -->
<img src='https://lh3.googleusercontent.com/d/12eOZ8tFZOBTFqz12-niKTYaZrzeKRPN8' alt='Plot showing the overlay of this query head cumulative energy and singular values along with the locations of the 99 and 99.9 energy points' width='800' />

<!-- md -->
Using the 99.9% threshold, the effective rank of this query head is 66.

The below demonstrates how we calculate the effective rank of a query head, where you would replace `W_q` with actual weight values.

<!-- code -->
```python
# Step 1: Define the cumulative energy function
def cumulative_energy(sigma, tau):
    sigma_e = sigma ** 2 # 'Energy' from each sigma
    total_e = np.sum(sigma_e) # Total energy
    cumulative_e = np.cumsum(sigma_e) # Cumulative energy by sigma
    sigma_ratios = cumulative_e / total_e # Fraction of total
    return np.argmax(sigma_ratios >= tau) + 1 # Index of sigma which exceeds tau

# Step 2: Retrieve the query head
W_q = torch.randn(d_model, d_head)

# Step 3: Decompose the query head
# Shapes
#       W_q  (d_model, d_head)
#      W_qa  (d_model, r     )
#      W_qb  (      r, d_head)
#   sigma_q  (r,)
# W_qb is a square matrix with shape (d_head, d_head). The use of `r` helps
# clarify the orientation of the singular vectors (they are the rows) and
# indicates the dimension along which we'll truncate.
W_qa, sigma_q, W_qb = torch.linalg.svd(W_q)

# Step 4: Get the effective rank of this query head.
eff_rank_q = cumulative_energy(sigma_q, 0.999)
```

<!-- md -->
**Note: Cumulative energy is a heuristic, not a guarantee**

While this is a useful shorthand for “most of the matrix’s action is in these directions,” it’s not a performance guarantee.

* Singular value energy is an *average-case* measure over all possible inputs.
* A rare-but-important direction could correspond to a small singular value but still be semantically critical.
* Energy ignores **structure** in the singular vectors — losing a low-energy direction could disproportionately affect specific tasks.

So while “99.9% energy retained” *sounds* like “99.9% performance retained,” the correlation is imperfect and needs to be validated empirically.

The goal here is not to redefine SVD truncation’s trade-offs — that’s been studied extensively — but to **improve how we identify latent bottlenecks** so that truncation targets the right dimensions.

<!-- md -->
### 1.5. Query-Key Truncation Example

<!-- md -->
The query head we explored earlier has a key head partner with similarly low rank.

Using the cumulative energy metric with a threshold of 99.9%, the Layer 13 Query Head 50 has an effective rank of 66, and the corresponding Key Head 50 has an effective rank of 72.

The size of the two heads needs to match, so if we want to retain our threshold then the lowest we can truncate them is to the top 72 singular vectors, (which may not be enough to reduce the total number of parameters?)

The below code assumes we've repeated the above calculations for the key head, and now we'll truncate.

<!-- code -->
```python
# Given the effective ranks of the query and key head, we can choose a rank
# that we will truncate to, 'r', which retains 99.9% energy. To do this, we
# need to pick the larger of the two. This gives r = 72.
r = max(eff_rank_q, eff_rank_k)

# Truncate by taking the first `r` columns or rows of the decomposed matrices.
W_qar = W_qa[:, :r]
W_qbr = W_qb[:r, :]

W_kar = W_ka[:, :r]
W_kbr = W_kb[:r, :]

sigma_q = sigma_q[:r]
sigma_k = sigma_k[:r]

# Fold the sigma values into their respective 'A' matrix; start by turning them
# into row vectors, then element-wise multiply with their respective columns.
W_qar = W_qar * sigma_q.unsqueeze(0)
W_kar = W_kar * sigma_k.unsqueeze(0)

# Now query and key vectors are calculated via:
q = x @ W_qar @ W_qbr
k = x @ W_kar @ W_kbr
```

<!-- md -->
We can calculate the parameter savings to determine if we've improved things.

The head shapes in Kimi-K2 are unusual because the input to them is a latent vector rather than the token vector. The query heads are $(1536, 128)$ and the key heads are $(512, 128)$.

  * **Original Parameters**:
      * $W^Q$: $1536 \times 128 = 196,608$
      * $W^K$: $512 \times 128 = 65,536$
      * **Total: 262,144 (256K)**

We've decomposed and truncated these to $(1536, 72) \times (72, 128)$ and $(512, 72) \times (72, 128)$.

  * **New Parameters (Individual Truncation)**:
      * $W^Q \rightarrow W^{QA_r} \times W^{QB_r}$: $(1536 \times 72) + (72 \times 128) = 110,592 + 9,216 = 119,808$
      * $W^K \rightarrow W^{KA_r} \times W^{KB_r}$: $(512 \times 72) + (72 \times 128) = 36,864 + 9,216 = 46,080$
      * **Total: 165,888 (162K)**

This is a **36.7%** reduction in parameters for this head, but we now have four matrices instead of two, and we can do better.

By fusing the heads together before applying SVD, we can reveal a much lower joint effective rank—just **41**—for their combined operation.

<!-- md -->
## S2. Fusing Attention Heads

<!-- md -->

The key insight of FaRR is to analyze the rank of the *combined linear map* rather than its individual components. When two matrices are multiplied, the rank of the resulting matrix can be much lower than the rank of either original matrix, especially if their latent spaces are misaligned or one is a subspace of the other.

<!-- md -->
### 2.1. How to Fuse

<!-- md -->
In multi-head attention, each head contains two projection matrices that form a low-rank linear map. For example, for the value and output matrices:

  * The **Value projection** $W^V_i$ maps from the model's embedding space ($d_\text{model}$) into a smaller per-head value space ($d_v$).
  * The **Output projection** $W^O_i$ maps from that per-head value space back into the model's embedding space.

These two matrices are usually implemented as part of larger concatenated matrices for all heads for GPU efficiency. But if you conceptually “break apart” the concatenation, each head's contribution to the residual stream is just:

$$\text{output}_i = (\alpha_i V_i) W^O_i$$

where $\alpha_i$ are the attention weights and $V_i = X W^V_i$.

Mechanistic interpretability researchers—notably in Anthropic’s *Transformer Circuits* framework—often work with the fused matrix, as it represents the head's complete information pathway:

$$W^{VO}_i = W^V_i \, W^O_i$$

The same fusion idea applies to the query/key side, where the matrices combine to produce the attention scores:

$$W^{QK}_i = W^Q_i \, (W^K_i)^\top$$

In both cases, this “fused” view lets us apply SVD to study a head’s true rank and subspace alignment—revealing inefficiencies and redundancy that aren’t visible when looking at just one side.

(A more detailed derivation of these fused matrices can be found in the appendix.)

<!-- md -->

**RoPE Caveat**

Most modern models prevent this simple fusion of the Query-Key matrices because they apply Rotary Positional Embeddings (RoPE) in between the projections. We *can*, however, apply FaRR to models like Kimi-K2, which only apply RoPE to a fraction of the head's dimensions, allowing us to fuse the non-RoPE part. More on that topic in the appendix.

<!-- md -->
### 2.2. Query-Key Example (Revisited with FaRR)

<!-- md -->

Let's return to our Kimi-K2 Layer 13 Head 50 example. Instead of decomposing $W^Q$ and $W^K$ separately, we first fuse them.

```python
# Step 1: Fuse the query and key heads
# W_q shape: (1536, 128)
# W_k shape: (512, 128)
W_qk = W_q @ W_k.T  # Fused shape: (1536, 512)

# Step 2: Decompose the fused matrix
W_qka, sigma_qk, W_qkb = torch.linalg.svd(W_qk)
# W_qka: (1536, 512)
# sigma_qk: (512,)
# W_qkb: (512, 512)

# Step 3: Get the effective rank of the fused matrix
eff_rank_fused = cumulative_energy(sigma_qk, 0.999)
# For this head, eff_rank_fused = 41
```

The fused effective rank is only **41**, far lower than the individual effective ranks of 66 (Query) and 72 (Key).

We can now decompose back into new query and key matrices, and truncate them both to this lower rank.

First, applying SVD to the fused matrix and folding the singular values into one of the matrices (how/where we fold them does not matter), gives back new query and key projections:

$\text{SVD}_{QK}(W^{QK}) = W^{QKA} \sigma^{QK} W^{QKB} = \hat{W}^{Q} (\hat{W}^{K})^\top $

These new matrices have the same shape and accuracy as the originals, but now they are each orthonormal and sorted.

Truncating them to their top 41 vectors yields a query matrix of size $(1536, 41)$ and a key matrix of size $(512, 41)$.

  * **New Parameters (FaRR)**:
      * New $W^Q$: $1536 \times 41 = 62,976$
      * New $W^K$: $512 \times 41 = 20,992$
      * **Total: 83,968 (82K)**

This is a **67.9%** reduction from the original 256K parameters, and a further **49.4%** reduction compared to the 162K parameters from individual truncation. We're now back to having just two matrices, but with far fewer parameters.

<!-- md -->
### 2.3. FaRR in Practice: Illustrative Layers

<!-- md -->

We've seen how FaRR can compress a single head. Now, let's examine how these low-rank opportunities manifest across a model. By analyzing different layers from DeepSeek-R1, we can identify two primary patterns where FaRR proves effective, as well as the more common case where it does not.

**Case 1: The Bottleneck Effect**

The rank of a fused matrix is mathematically constrained by the ranks of its constituent matrices:
$$\operatorname{rank}(W^{VO}) \le \min\big(\operatorname{rank}(W^V), \operatorname{rank}(W^O)\big)$$
This means that if one matrix in a pair already has a low effective rank, it creates a bottleneck that the fused matrix cannot exceed. This is clearly visible in Layer 8 of DeepSeek-R1. While the Output heads (blue dots) are consistently high-rank, many Value heads (orange dots) have a very low effective rank, dragging the fused rank (green line) down with them.

<!-- md -->

<img src='https://lh3.googleusercontent.com/d/1A4whsqz0PRgxR0fa3n73XTpSR0XfwUo9' alt='Plot of the effective ranks of the Value, Output, and fused VO matrix in DeepSeek-R1 showing the combined effective rank being dragged down by the value matrix' width='900' />

<!-- md -->

**Case 2: Only Low When Fused**

More interestingly, it's possible for both matrices in a pair to have high effective rank individually, yet produce a fused matrix with a much lower rank.

Layer 2 of DeepSeek-R1 provides a prominent example. Both the Value and Output heads appear to be high-rank, but their combined effective rank is dramatically lower.

<!-- md -->

<img src='https://lh3.googleusercontent.com/d/1iBfzQ1qLt6p5jhHMGaBtD3vob3RAxTZh' alt='Plot of the effective ranks of the Value, Output, and fused VO matrix in DeepSeek-R1 layer 2 showing the combined effective rank being much lower than either of the v or o separately' width='900' />

<!-- md -->
This has at least two possible causes, and deserves more investigation.

The example query-key head pair that we explored (Layer 13 Head 50 of Kimi-K2) exhibited this, and it could predominantly (but not entirely) be attributed to the singular values of the key head being more uniformly low--it appears to be suppressing much of the pair's behavior.

Cumulative energy only looks at the magnitude of the singular values relative to one another, so it doesn't capture this when looking at the key head independently.

The remainder of the rank drop comes from the interaction between the $W^{QB}(W^{KB})^\top$ matrices, which define the heads subspaces, and can be "misaligned". We'll explore this more in part 2.

<!-- md -->

**The Common Case: High-Rank Layers**

While the above cases present opportunities, it is crucial to note that they are the exception. In most layers, both the individual and fused matrices are full or near-full rank. Layer 28, shown below, is representative of this common scenario, where FaRR would offer no parameter savings.

<img src='https://lh3.googleusercontent.com/d/1FAwlOuRATdVS3xJ1c44FK3_S2nnBpGo8' alt='Plot of the effective ranks of the Value, Output, and fused VO matrix in layer 28 of DeepSeek-R1 showing high rank for all three' width='700' />

<!-- md -->
### 2.4. A Model-Wide View

<!-- md -->
Having examined individual layer patterns, we now zoom out to view the entire DeepSeek-R1 model. The grid plots below show the effective rank (at 99.9% energy) for all Value-Output and Query-Key head pairs. Each plot sorts the heads within a layer by their fused effective rank (the green line) to reveal the distribution of compression opportunities. The horizontal axis represents the 128 heads in the layer, and the vertical axis is the effective rank, capped at 128.

<!-- md -->
**Value-Output (VO) Heads**

<img src='https://mccormickml.com/assets/svd_fusion/vo_heads_all_layers_ds-r1.png' alt='Plot showing, for all 61 layers of DeepSeek R1, the effective rank of the value and output heads and their fused form. Sorted by fused effective rank. Early layers show low rank, and some at the end, but the bulk of the middle appears very high.' width='900' />

<!-- md -->

Several key patterns emerge from this model-wide view:

  * **Most layers are high-rank.** The central block of the model, roughly layers 20–50, consists almost entirely of full-rank heads where FaRR would be ineffective.
  * **Opportunities are concentrated in early and late layers.** Significant low-rank structure is present in the first \~15 layers and again near the final layer.
  * **A curious split emerges in early layers.** Layers 4–8 exhibit a consistent split where roughly half the heads show the "bottleneck effect" (low-rank Value head) and the other half show the "redundant subspace" pattern (both V/O high-rank, but low fused rank). This bimodal distribution of behavior within a single layer is remarkable.

<!-- md -->
**Query-Key (QK) Heads**

<!-- md -->
![Grid plot showing all 61 layers of DeepSeek-R1, plotting the effective rank of all query and key heads and their fused form. Might be more low rank heads overall than for value-output, but still plenty of layers with high rank heads.](https://mccormickml.com/assets/svd_fusion/qk_heads_all_layers_ds-r1.png)

<!-- md -->
The Query-Key heads show even more pronounced low-rank structure in the early layers. Layer 8, for instance, appears to have a uniformly low effective rank across all heads, making it an ideal candidate for FaRR. However, other layers like 5 and 7 are frustratingly close, with a small fraction of high-rank heads preventing a simple, uniform truncation across the entire layer.

<!-- md -->

**The Practical Challenge: Mismatched Head Sizes and Parallelism**

These plots reveal the primary obstacle to applying FaRR for inference speedups--heterogenous rank.

FlashAttention requires that the query, key, and value heads in a layer all be of identical size.

To adhere to our chosen energy threshold, then, we cannot truncate any lower than the _single highest rank head_ in a layer, whether that's on the $W^{QK}$ side or the $W^{VO}$ side.

To apply FaRR cleanly, we would like to see _all_ heads within a layer have low rank.

<!-- md -->
**The Opportunities**

Opportunities still exist, but require compromise.

Note that:

1. From a compute perspective, any parameter reduction achieved via FaRR is _free_, because we are not introducing any new calculations. There is no "break-even" point we need to pass, and no added overhead we need to amortize.
2. 99.9% is a fairly conservative threshold, and fine-tuning can help recover lost performance.
3. We can also deviate from point 1--In the same way that a tipping point exists for breaking a large matrix into two smaller ones, there is a break-even point for running two attention passes at two sizes.

<!-- md -->
## S3. Conclusion

<!-- md -->
(TODO)

<!-- md -->
# Appendix

<!-- md -->
## A.1. Mapping to Standard SVD

<!-- md -->

Standard compact SVD is defined as:

$$
\text{SVD}(W) = U\,\Sigma\,V^\top
$$

Where,
* $U \in \mathbb{R}^{d_\text{output}\times r} \quad$ the left singular vectors, spanning the **output** space of $W$
* $\Sigma \in\mathbb{R}^{r\times r} \quad$ the singular values of $W$.
* $V^\top \in\mathbb{R}^{r\times d_\text{input}} \quad$ the right singular vectors, spanning the **input** space of $W$.

<!-- md -->
**Mapping to $SVD_\text{QKV}$**

Our A/B notation corresponds to:

$$
\boxed{\,W^{QA} = U,\quad \sigma^Q = \mathrm{diag}(\Sigma),\quad W^{QB} = V^\top\,}
$$

<br/>

and we write $W = W^{QA}\,\sigma^Q\,W^{QB}$ instead of $U\Sigma V^\top$.


All row/column scaling by $\sigma$ is implicit based on its placement relative to the matrix.

<!-- md -->
**In Deep Learning Libraries**

<!-- md -->
PyTorch returns `U, S, Vh` with shapes:

* `U` = `[d_model, r]` (orthonormal columns) → $W^{QA}$
* `S` = `[r]` (descending singular values) → $\sigma$
* `Vh` = `[r, d_head]` (orthonormal rows) → $W^{QB}$

So:

```python
W_qa, sigma_q, W_qb = torch.linalg.svd(W_q, full_matrices=False)
# W_q  =  W_qa @ (W_qb * sigma.unsqueeze(1))  =  (W_qa * sigma.unsqueeze(0)) @ W_qb
```

<!-- md -->
TODO - We could further simplify by defining a function:

<!-- md -->
```python
@torch.no_grad()
def svd_qkv(W: torch.Tensor):
    """
    Decompose a query, key, or value attention head into A/B + sigma, using our notation.
    Returns: W_a, sigma_w, W_b  (columns-ortho, vector, rows-ortho)
    Example:
    W_qa, sigma_q, W_qb = svd_qkv(W_q)
    """
    W_a, sigma_w, W_b = torch.linalg.svd(W, full_matrices=False)
    # (Optional) sanity checks you might enable during development:
    # assert torch.allclose(W_qa.T @ W_qa, torch.eye(W_qa.shape[1]), atol=1e-5)
    # assert torch.allclose(W_qb @ W_qb.T, torch.eye(W_qb.shape[0]), atol=1e-5)
    return W_a, sigma_w, W_b
```

<!-- md -->
## A.2. Fusion Derivation

<!-- md -->
Let’s start from the standard single-head attention equations for a query token embedding $x_q \in \mathbb{R}^{d_\text{model}}$ and a sequence of token embeddings $X \in \mathbb{R}^{T \times d_\text{model}}$:

**Queries, Keys, Values:**

$$
q = x_q W^Q_i
$$

$$
K = X W^K_i
$$

$$
V = X W^V_i
$$

**Attention scores and weighted sum:**

$$
\alpha = \mathrm{softmax}\!\left(\frac{q K^\top}{\sqrt{d_k}}\right)
$$

$$
z = \alpha V
$$

where $z \in \mathbb{R}^{1 \times d_v}$ is in the head’s value space.

**Output projection:**

$$
o = z W^O_i
$$

with $o \in \mathbb{R}^{1 \times d_\text{model}}$ in the full embedding space.

<!-- md -->
**Moving $W^O_i$ before the attention weights**

<!-- md -->
Because matrix multiplication is associative, we can fuse $W^V_i$ and $W^O_i$ before applying the attention weights:

$$
o = \alpha \, (V W^O_i)
$$

$$
\phantom{o} = \alpha \, (X W^V_i W^O_i)
$$

This gives us the **Value–Output fused matrix**:

$$
\boxed{W^{VO}_i = W^V_i W^O_i}
$$

which maps directly from the model space to the model space through a $d_v$-dimensional bottleneck.

<!-- md -->
**Query–Key fusion**

<!-- md -->
Similarly, in the score computation:

$$
\alpha = \mathrm{softmax}\!\left(\frac{(x_q W^Q_i) (X W^K_i)^\top}{\sqrt{d_k}}\right)
$$

we can regroup:

$$
(x_q W^Q_i)(X W^K_i)^\top
= x_q \, \big[ W^Q_i (W^K_i)^\top \big] \, X^\top
$$

so:

$$
\boxed{W^{QK}_i = W^Q_i (W^K_i)^\top}
$$

<!-- md -->
## A.3. RoPE Head Fusion

<!-- md -->
If we want to analyze the shared subspace between $W^Q$ and $W^K$, we might try to fuse them into a single matrix:

$$
W^{QK}_i = W^Q_i \cdot W^{K\top}_i
$$

That works fine **if** $Q$ and $K$ are computed directly from the same linear projection of $X$.

But with RoPE, the process is:

$$
Q_\text{rot} = \text{RoPE}(X W^Q), \quad K_\text{rot} = \text{RoPE}(X W^K)
$$

RoPE is a **token-position–dependent rotation** in each query/key vector’s 2D subspaces.
It’s **not linear**, so you can’t simply push it through or ignore it:

* The rotation is different for each token position $p$.
* It changes the basis directions themselves, meaning the fused $W^{QK}$ would need to “know” about position $p$ to be correct.

Can we work around this?

If a model uses **partial RoPE** (e.g., only the first $d_\text{rope}$ dims per head), you can fuse and analyze the remaining “NoPE” dimensions exactly like $W^{VO}$. These dimensions see no rotation, so QK fusion is valid there.

This is the approach we took for this article.

Also, a technique which can still allow for fused rank analysis (but not FaRR) is to fuse with a fixed-offset.

If you choose a fixed position offset $\Delta p$, you can apply RoPE with that offset to both $W^Q$ and $W^K$ **before** fusing:

   $$
   W^{QK}_\text{offset}(\Delta p) = \text{RoPE}_{p=0}(W^Q) \cdot \text{RoPE}_{p=\Delta p}(W^K)^\top
   $$

The result is offset-specific rather than global, so we can't use it for parameter reduction on RoPE heads.

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

