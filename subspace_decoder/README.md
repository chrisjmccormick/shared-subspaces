# Subspace Decoder: Shared and Private Attention Spaces

## Understanding Attention Head Space Decomposition

This project explores the decomposition of attention head spaces into **"shared"** and **"private"** components, extending the Multi-head Latent Attention (MLA) architecture popularized by models like DeepSeek-V3 and Kimi-K2.

### What is a "Subspace"?

In our context, a **subspace** refers to a lower-dimensional representation space that multiple attention heads can share. This is mathematically equivalent to what others might call a **"latent space"** — the terminology used in the original MultiheadLatentAttention implementation. 

Think of the attention mechanism as having multiple "rooms" where different heads can work:
- **Private spaces**: Each attention head has its own dedicated room (traditional multi-head attention)
- **Shared spaces**: Multiple heads share a common room, forcing them to coordinate and potentially learn more efficiently

### Shared vs. Private Space Decomposition

#### Traditional Multi-Head Attention (MHA)
In standard attention, each head operates in its own private space:
```
Query_i = W^Q_i × Input    (separate W^Q_i for each head i)
Key_i   = W^K_i × Input    (separate W^K_i for each head i) 
Value_i = W^V_i × Input    (separate W^V_i for each head i)
Output  = W^O × concat(head_1, head_2, ..., head_n)
```

#### Multi-Head Latent Attention (MLA)
MLA introduces shared subspaces for queries and keys/values:
```
# Shared projections (all heads share these spaces)
Q_shared = W^Q_shared × Input     (single shared projection)
KV_shared = W^KV_shared × Input   (single shared projection)

# Private projections (per-head, but from shared space)
Query_i = W^Q_private_i × Q_shared
Key_i   = W^K_private_i × KV_shared
Value_i = W^V_private_i × KV_shared
```

#### MLA with Output Subspace (MLA-O)
Our extension adds a shared output subspace:
```
# After attention computation...
Output_latent_i = W^O_private_i × AttentionOutput_i  (per-head to shared space)
Output = W^O_shared × sum(Output_latent_1, ..., Output_latent_n)  (shared to model space)
```

### Alternative Nomenclature

Other common naming conventions for this decomposition include:

- **LoRA-style naming**: `A` and `B` matrices (where A projects down, B projects up)
- **Down-Up notation**: `D` and `U` matrices (emphasizing the dimensional reduction and expansion)
- **Bottleneck terminology**: "Compression" and "decompression" matrices

However, we've chosen the **"shared vs. private"** terminology in this project to emphasize the core conceptual difference: whether attention heads operate in isolation (private) or coordinate through common representational spaces (shared).

### Why This Matters

The shared/private distinction captures something fundamental about how we want attention heads to learn:

- **Shared spaces** receive gradients from every token and every head that uses them, potentially enabling faster learning and better generalization
- **Private spaces** allow heads to specialize without interference, but may learn more slowly and redundantly
- **The balance** between shared coordination and private specialization may be key to efficient attention mechanisms

---

*For more background and discussion of the motivation, see the original blog post [here](https://mccormickml.com/2025/07/28/output-latent-spaces-in-multihead-attention/).*
