# Handoff: Fused Attention Rank Analysis on Muse Glimmer 30B

**Status:** context-gathering complete, no experiment run yet.
**Branch:** `glimmer-fused-vo` in `shared-subspaces`.
**Target hardware:** one 80 GB A100 (to be reserved).
**Date compiled:** 2026-08-11.

This document is *context*, not a procedure. It collects what the prior `fused_attn_svd`
work established, what Muse Glimmer's architecture actually is (verified against the HF
config and the `transformers` modeling source, not from memory), which fusions are
mathematically valid in this model and which are blocked, and the open questions worth
attacking. Design the experiment yourself.

---

## 1. Why this project exists now

Chris previously found that in large MLA models (DeepSeek-V3/R1, Kimi-K2) the **fused**
attention matrices — $W^{VO}_i = W^V_i W^O_i$ and $W^{QK}_i = W^Q_i (W^K_i)^\top$ — often
have substantially lower effective rank than either constituent matrix, concentrated in
early layers. See `Fuse and Rank Reduce - Part 1 - Truncation.md` for the full write-up.

Kaiyue Wen (@wen_kaiyue) posted about the just-released Meta **Muse Glimmer 30B**
(`x.com/wen_kaiyue/status/2086869377442369620`, 2026-08-10):

> "A hidden detail in the recently released Muse Glimmer model: its per-matrix weight RMS
> norm is pinned almost exactly at ~6e-3! If you're curious why fixing weight RMS like this
> can work, feel free to checkout Hyperball: https://arxiv.org/abs/2606.16899"
>
> "Why 6e-3? This is actually just fixing the matrix weight to have RMS $0.5/\sqrt{d}$ with
> $d = 6656$ being the embedding dimension of the network."
>
> "Could this be a coincidence? It is impossible to tell for sure but the Weight RMS
> distribution for Qwen3 8B is vastly different."

The arithmetic checks out exactly: $0.5/\sqrt{6656} = 6.1286\text{e-}3$.

**Note the direction of the Qwen3-8B comparison:** Glimmer's per-matrix RMS is *flat* —
pinned across layers and matrices. Qwen3-8B's is *not*, and that varying shape is what
Chris recognized as resembling his effective-rank-by-layer curves. Qwen3-8B is the
**contrast**, not a second example.

Chris replied to the tweet:

> "Matrix norm and variance stuff makes me itch since, here, attn_q is 32 functionally
> independent matrices that just happen to sit next to one another on the GPU. Is the stat
> true at a head level? (and does that even matter? I don't actually know)"

So there are **two** deliverables in scope: the fused-rank analysis, and the head-level
norm decomposition.

> ⚠️ Meta's blog and model card disclose **nothing** about the training recipe — no
> optimizer, no weight-norm constraint, no weight decay — and there is no tech report.
> That Glimmer used Hyperball (or Muon at all) is Wen's *inference from the weights*, not a
> disclosed fact. Treat it as a hypothesis the weights can test, and re-derive the RMS
> numbers here rather than citing them.

### 1.1 What Hyperball actually does, and why it matters here

**"Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization"** —
Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang ([arXiv:2606.16899](https://arxiv.org/abs/2606.16899)).

> Matrix-based optimizers such as Muon can substantially speed up LM pretraining, but their
> gains over AdamW shrink at scale under standard constant decoupled weight decay. Hyperball
> is an optimizer *wrapper*: given a base optimizer (Adam or Muon), it **sets the Frobenius
> norms of weight matrices and of their corresponding updates to fixed constants**. On
> Qwen3-style models up to 1.2 B, Muon+Hyperball gets a 20–30% token-equivalent speedup over
> weight-decay baselines, and improves LR transfer across widths and depths. Motivation:
> training with weight decay leads to an equilibrium weight norm that depends only on the
> training hyperparameters, and that norm then decides the **angular learning rate** — how
> fast the *direction* of the weight matrix changes. Hyperball makes that control explicit
> rather than an emergent side effect.

Two consequences that shape this experiment:

**(a) A Frobenius constraint pins total energy, not its distribution.** It says nothing
about the *spectrum*. A matrix can sit exactly on the Hyperball and still be badly
low-rank — all its energy concentrated in 40 of 128 singular directions. So norm-pinning
does **not** on its own predict "the low-rank pathology goes away." What it *does* block is
the whole matrix drifting toward zero, i.e. Case-1 suppression executed at the
matrix level.

**(b) It says nothing about per-head allocation either — which is exactly Chris's point.**
`q_proj` is one 4096×6656 tensor holding 32 independent heads. A constraint on
$\lVert W \rVert_F$ leaves the model completely free to move energy *between* heads. If
anything, once global scale is fixed, redistribution across heads is one of the few
remaining free directions, so a pinned per-matrix RMS is not merely *compatible* with
extreme per-head variation — it may actively encourage it. Chris's tweet reply is therefore
the right question, and §5 is the measurement that answers it.

### 1.2 Four candidate mechanisms, only some of which Glimmer has

Worth keeping distinct, because they make different predictions and the experiment can
partly separate them:

| Mechanism | Effect on the low-rank pathology | In Glimmer? |
|---|---|---|
| **Hyperball / RMS pinning** | Blocks whole-matrix magnitude drift. Does **not** flatten spectra. | Inferred from weights (~6e-3) |
| **Muon (orthogonalized updates)** | Pushes updates toward flat spectra → directly fights low effective rank. | Unknown; Hyperball wraps it, so plausible |
| **Scale-free QK-norm** | Makes $W^Q$/$W^K$ magnitude functionally *unobservable* — suppression-by-shrinkage is inexpressible. | **Yes**, confirmed in code (§3.3) |
| **Attention output gate** | Provides a cheap data-dependent suppression channel, removing the *incentive* to suppress via weights. | **Yes**, confirmed in code (§3.3) |

Note that Muon, not Hyperball, is the mechanism that would actually attack effective rank.
If Glimmer's spectra look flatter than DeepSeek's, that's evidence about the *optimizer*;
if only the magnitudes look controlled, that's evidence about the *wrapper*.

Two alternative readings of the 6e-3 observation came up in the replies and are worth
ruling out early, since they're cheap to distinguish (§5.3): **nGPT** (@rokuJitsu) and a
general **ℓ2-unit-ball constraint on all layers** (@powns_ai).

### 1.3 Already measured (2026-08-11) — the norm question is settled

**Full sweep complete:** every layer of both models — 416 Glimmer matrices (52 × 8) and
252 Qwen3-8B matrices (36 × 7). Raw data in `results/*_norms.csv` (+ `.npz` with per-head
and per-row arrays, gitignored — regenerate with `norm_survey.py --layers all`). Figures:
`fig1_rms_pinned.png`, `fig2_per_head_spread.png`, `fig3_gate_head_band.png`.

Three results, all clean:

**(1) The claim is true, and it is a per-matrix RMS target — not spectral, not nGPT.**
All 416 Glimmer matrices — attention *and* MLP, across five distinct shapes (4096×6656,
256×6656, 6656×4096, 19968×6656, 6656×19968) — sit at weight RMS **6.1292e-3**.
Ratio to $0.5/\sqrt{6656}$ spans **0.99982 → 1.00024**: a total spread of **0.042% across
the entire model**. Shape-independence at that precision rules out a spectral condition,
which would scale with $\sqrt{d_\text{out}/d_\text{in}}$ and split `down_proj` from
`q_proj` by tens of percent. Per-**row** norm CV is 0.05–0.40, nowhere near zero, which
rules out nGPT-style row normalization. This is exactly the Hyperball signature: fixed
$\lVert W\rVert_F$ per matrix, rows unconstrained. (The consistent ~1.0001 centre is
almost certainly bf16 rounding bias, not a real offset.)

**(2) Qwen3-8B confirms the contrast, and validates the measurement.** Its ratio to
$0.5/\sqrt{d}$ spans **1.877 → 4.264** — a 127% spread, no pinning whatsoever. The same
code reports variation where variation exists. Its early layers (1–5) show a large
excursion: `mlp.up`/`mlp.down` dive to ~1.9 while `mlp.gate` spikes to 4.26 at layer 5.
That early-layer irregularity is the feature Chris recognized as resembling the shape of
the effective-rank curves.

**(3) Chris's tweet question is answered: NO, the statistic does not hold per head.**
Per-head RMS max/min within a matrix, over all 52 layers:

| matrix | median | worst | at layer |
|---|---|---|---|
| **attn gate_proj** | **1.73** | **2.69** | 23 |
| attn q_proj | 1.34 | 1.69 | 43 |
| attn o_proj | 1.29 | 1.66 | 34 |
| attn v_proj | 1.09 | 2.01 | 48 |
| attn k_proj | 1.07 | 1.52 | 44 |

Exactly what §1.1(b) predicts: with whole-matrix scale pinned, cross-head redistribution
is one of the few remaining free directions, and the model uses it — heavily.

Three things the full sweep shows that the 4-layer sample did not:

- **`gate_proj` dominates, and the shape is a mid-model hump, not monotonic growth.** It
  rises from ~1.2 at layer 0 to a peak of **2.69 at layer 23**, then stays elevated
  (1.6–2.2) through the back half. That is not where §5.2 predicted the widest spread —
  the prediction was `q_proj`, since `qk_norm` makes q/k scale unobservable so nothing
  downstream constrains it. The gate winning is *more* interesting: per-head gate weight
  norm controls **sigmoid saturation** (large $\lVert G_i\rVert$ ⇒ a decisive per-token
  on/off switch; small ⇒ a gate stuck near $\sigma(0)=0.5$ that barely gates). The largest
  head-level differentiation in the model sits in the *head-suppression channel* — direct
  support for the §4.3 thesis.
- **Correction to the 4-layer reading: `v_proj` does *not* stay flat.** The sample
  (layers 0/3/25/51, all ≈1.11–1.15) suggested GQA left nothing to differentiate on the
  V side. Over all layers, v and k are indeed flat (~1.05–1.15) through roughly layer 40 —
  and then become erratic, v spiking to **2.01 at layer 48** and k to 1.52 at layer 44.
  Something changes in the last ~10 layers on the KV side specifically. Worth a look; the
  late layers are also where the original fused-rank work found a second low-rank region.
- **The constraint holds the centre and lets the spread run** (`fig3`). Glimmer's *median*
  head RMS for `gate_proj` sits at ~0.0061 — the pinned target — essentially constant
  across all 52 layers, while the min→max band widens from ±0.001 at layer 0 to ±0.0025
  from layer 20 on. The Frobenius constraint fixes the mean of the squares; everything
  about how that budget is distributed across heads is free, and the model spends it.

Qwen3-8B for comparison has per-head spread of similar *magnitude* (o 1.52 median, v 1.40,
q 1.30, k 1.23) but **no depth structure** — it is noisy at every layer. So the difference
is not that Glimmer's heads vary more; it is that Glimmer's variation is *organized by
depth and concentrated in the gate*.

Remaining on the norm question: whether per-head gate norm predicts fused VO rank. That
link is the actual payoff, and it is now a join between `results/glimmer_norms.npz` and
the fused-rank work that has not been done yet.

---

## 2. What the prior work established (recap)

Definitions used throughout, matching the existing notebooks:

- **Effective rank** = smallest $r$ such that retained singular-value *energy* (sum of
  $\sigma^2$) is $\ge \tau$ of the total. The notebooks report $\tau = 0.99$ and $0.999$
  (i.e. 1% and 0.1% error). `get_rank_for_error_threshold()` is the canonical
  implementation and is duplicated in both notebooks.
- **Fused matrix** = the product of the two matrices the model already multiplies together
  with no nonlinearity between them. Fusing costs no extra multiplies, so any parameter
  saved by truncating the fused form is free (no break-even point).
- **Stacked heads** = concatenate all heads of one type in a layer along the head axis and
  take the SVD of the result, to find structure shared across heads. The notebooks apply
  **spectral normalization** (divide each head by its own $\sigma_1$) before stacking, so
  loud heads don't dominate. This raises effective ranks and is a known caveat.

Findings that motivate this experiment:

| Pattern | Description | Where seen |
|---|---|---|
| **Case 1 — bottleneck** | One side (usually $W^V$) has low effective rank; the fused rank is dragged down to match, even though $W^O$ is full rank. The high-rank side has more parameters than it can use. | DS-R1 layer 8 |
| **Case 2 — misaligned** | *Both* sides are high rank individually, but the fused form is much lower. The two matrices disagree about the definition of the shared $d_\text{head}$-dim space. | DS-R1 layer 2, Kimi-K2 L13 H50 |
| **Common case** | Both sides and the fusion are near-full rank. This is most of the model (roughly layers 20–50 of DS-R1). | DS-R1 layer 28 |

Chris's current reading of the pathology, and the thing to test:

> "One side of the head pair had very low magnitude along many of its bases, so it was
> effectively shutting off the head (even though the other side might be fuller rank)."

Note that cumulative-energy effective rank is **scale-invariant per matrix** — it only sees
*relative* singular values. So a uniformly-small side looks full-rank in isolation and only
reveals itself in the fusion. That is exactly why the Kimi-K2 L13 H50 key head appeared
high-rank alone but crushed the fused rank.

Also worth carrying forward from the notebooks: DeepSeek-V3-Base and DeepSeek-R1 have
**nearly identical** effective ranks everywhere — RL post-training barely moves the spectra.

### Existing assets

- Pre-computed singular values for DS-R1 and DS-V3-Base live in the HF dataset repo
  `ChrisMcCormick/svd-attn-singvals` (files `ds-r1_singular_values.pkl`,
  `ds-v3-base_singular_values.pkl`). Kimi-K2 was computed but is marked
  `vals_available = False` in the notebooks.
- Data structure (pickled 3-tuple `(S_subspaces, S_heads, S_stacked_heads)`):
  ```python
  S_subspaces[layer_i][W_name]           -> 1-D array of sigmas   (one matrix per layer)
  S_heads[layer_i][W_name][head_i]       -> 1-D array of sigmas   (per head)
  S_stacked_heads[layer_i][W_name]       -> 1-D array of sigmas   (heads concatenated)
  ```
  This schema is MLA-shaped (`KVA`, `QA`, `KVA_pe`, `Q_pe` keys). Glimmer has no latent
  projections, so it will need adapting — but keeping `S_heads` / `S_stacked_heads`
  compatible means the plotting notebook's functions (`plot_headwise_rank`,
  `plot_layerwise_rank`, and the 11×6 grid plots) work with minimal edits.
- The three source notebooks are now committed as `.md` alongside the `.ipynb`
  (converted with `colab_utils.py to-md`).

---

## 3. Muse Glimmer 30B — architecture

**Repo:** `meta-models/Muse-Glimmer-30B` on HF. Apache-2.0, ungated, 59.6 GB in two
safetensors shards (50 GB + 9.6 GB), bf16. No text-only checkpoint. Vision tower is bundled
in the same files.

All of the following is verified against `config.json` and
`transformers/models/muse_glimmer/modeling_muse_glimmer.py` on `main`.

### 3.1 Text decoder config (the part that matters)

| Field | Value |
|---|---|
| `hidden_size` | **6656** |
| `num_hidden_layers` | **52** |
| `num_attention_heads` | **32** |
| `num_key_value_heads` | **2** (GQA, 16 query heads per KV head) |
| `head_dim` | **128** |
| `intermediate_size` | 19968 (dense MLP — **there is no MoE**) |
| `vocab_size` | 202048, `tie_word_embeddings: false` |
| `sliding_window` | 2048 |
| `qk_scale_factor` | **3.87** |
| `rms_norm_eps` / `post_norm_eps` | 1e-5 / 1e-8 |
| `final_logit_softcapping` | 20.0 |
| `output_multiplier` | 0.196116… = $1/\sqrt{26}$ = $1/\sqrt{L/2}$ |
| `attention_bias` | false (no biases anywhere in attention) |

Text-side parameter count works out to ≈27.9 B (25.2 B decoder + 2×1.34 B embeddings);
the remainder of the advertised ~30 B is the 2 B vision tower.

### 3.2 Layer type map — **this is the key structural fact**

`layer_types` repeats `[sliding, sliding, sliding, full]` 13 times. `layer_rope_theta`
repeats `[500000, 500000, 500000, 0]`.

```
full_attention (NoPE) layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51   (13 layers)
sliding_attention (RoPE, w=2048): all others                                      (39 layers)
```

Confirmed in `MuseGlimmerTextModel.forward`:

```python
position_embeddings=position_embeddings if self.config.layer_rope_theta[i] else None,
```

and in the attention `__init__` comment: *"NoPE layers (layer_rope_theta == 0) get no
position embeddings."*

**Consequence: on those 13 global layers, $W^{QK}$ fusion is exactly valid** — not
"partial-RoPE valid" like Kimi-K2, where only the NoPE slice of each head could be fused.
The *entire* 128-dim head is NoPE. This is the cleanest QK-fusion setting available in any
model analyzed so far. On the 39 sliding layers, full RoPE blocks exact QK fusion as usual
(the fixed-offset $\Delta$ trick from Appendix A.3 is still available for analysis).

### 3.3 Attention module — exact code

```python
self.q_proj    = nn.Linear(6656, 32*128 = 4096, bias=False)
self.k_proj    = nn.Linear(6656,  2*128 =  256, bias=False)
self.v_proj    = nn.Linear(6656,  2*128 =  256, bias=False)
self.o_proj    = nn.Linear(4096, 6656,          bias=False)
self.gate_proj = nn.Linear(6656, 32*128 = 4096, bias=False)   # <-- attention output gate
self.qk_norm   = MuseGlimmerRMSNorm(eps=1e-5, with_scale=False)   # NO learnable weight
self.qk_scale_factor = 3.87
```

```python
q = q_proj(x).view(..., 32, 128).transpose(1,2)
k = k_proj(x).view(..., 2, 128).transpose(1,2)
v = v_proj(x).view(..., 2, 128).transpose(1,2)

q = qk_norm(q) * 3.87          # per-head RMS norm, then a global scalar
k = qk_norm(k)                 # per-head RMS norm

if position_embeddings is not None:          # sliding layers only
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

attn_output = attention(q, k, v, scaling=128**-0.5, sliding_window=...)
attn_output = attn_output.reshape(..., 4096)
attn_output = attn_output * torch.sigmoid(gate_proj(x))       # <-- data-dependent gate
attn_output = o_proj(attn_output)
```

Three things sit *between* the matrices we want to fuse. Each one needs its own treatment
(§4).

### 3.4 Checkpoint tensor names and shapes

Text decoder tensors are under `model.language_model.*`:

```
model.language_model.embed_tokens.weight                              (202048, 6656)
model.language_model.layers.{i}.input_layernorm.weight                (6656,)
model.language_model.layers.{i}.self_attn.q_proj.weight               (4096, 6656)
model.language_model.layers.{i}.self_attn.k_proj.weight               ( 256, 6656)
model.language_model.layers.{i}.self_attn.v_proj.weight               ( 256, 6656)
model.language_model.layers.{i}.self_attn.gate_proj.weight            (4096, 6656)
model.language_model.layers.{i}.self_attn.o_proj.weight               (6656, 4096)
model.language_model.layers.{i}.post_attention_layernorm.weight       (6656,)
model.language_model.layers.{i}.pre_feedforward_layernorm.weight      (6656,)
model.language_model.layers.{i}.mlp.{gate,up}_proj.weight             (19968, 6656)
model.language_model.layers.{i}.mlp.down_proj.weight                  (6656, 19968)
model.language_model.layers.{i}.post_feedforward_layernorm.weight     (6656,)
model.language_model.norm.weight                                      (6656,)
lm_head.weight                                                        (202048, 6656)
```

Shapes above are inferred from the module definitions, not read off the index file — worth a
one-line assert on first load. `nn.Linear.weight` is `(out_features, in_features)`.

**Head slicing** (note the two orientations):

```python
# rows are heads:
Q_i = q_proj.weight[i*128:(i+1)*128, :]        # (128, 6656),  i in 0..31
K_g = k_proj.weight[g*128:(g+1)*128, :]        # (128, 6656),  g in 0..1
V_g = v_proj.weight[g*128:(g+1)*128, :]        # (128, 6656)
G_i = gate_proj.weight[i*128:(i+1)*128, :]     # (128, 6656)

# columns are heads:
O_i = o_proj.weight[:, i*128:(i+1)*128]        # (6656, 128)
```

**GQA mapping:** `num_key_value_groups = 32 // 2 = 16`, and HF's `repeat_kv` repeats each KV
head contiguously, so

```
kv head 0  ->  query heads  0..15
kv head 1  ->  query heads 16..31
g(i) = i // 16
```

### 3.5 Norm folding — **gotcha, different from the DeepSeek notebook**

`MuseGlimmerTextCenteredRMSNorm` uses **zero-centered gamma** (Gemma-style):

```python
self.weight = nn.Parameter(torch.zeros(dim))
...
output = self._norm(x.float()) * (1.0 + self.weight.float())
```

Despite the name, `_norm` does **not** subtract the mean — it is a plain RMS norm. What is
"centered" is the *weight*, initialized at zero and used as $(1 + w)$.

So the fold into `q/k/v/gate_proj` (which all consume `input_layernorm(x)`) is

```python
W_folded = W * (1.0 + input_layernorm_weight).unsqueeze(0)     # broadcast over columns
```

The MLA notebook does `W = W * norm_weight.unsqueeze(0)`. **Copying that line verbatim here
is wrong** and would zero out any coordinate where the learned $w \approx 0$ — which, for a
zero-initialized gamma, is most of them.

`qk_norm` has `with_scale=False`, so there is **nothing to fold** on the QK side.
`o_proj` sits inside the residual branch and is followed by `post_attention_layernorm`,
which is applied to the *sum*, so it cannot be folded into `o_proj`.

---

## 4. What fuses cleanly in Glimmer, and what doesn't

Using the row-vector convention of the notebooks ($x W$), define per-head matrices
$W^Q_i = Q_i^\top \in \mathbb{R}^{6656 \times 128}$ and likewise for $K, V, G$; and
$W^O_i = O_i^\top \in \mathbb{R}^{128 \times 6656}$.

### 4.1 QK — exactly fusible on the 13 NoPE layers, and *nothing needs folding*

$$q = \frac{x W^Q_i}{\mathrm{rms}(x W^Q_i)}\cdot 3.87, \qquad
  k = \frac{x' W^K_{g(i)}}{\mathrm{rms}(x' W^K_{g(i)})}$$

$$q \cdot k^\top \;=\; \underbrace{\frac{3.87}{\mathrm{rms}(\cdot)\,\mathrm{rms}(\cdot)}}_{\text{positive per-token scalar}} \;\cdot\; x \Big[ W^Q_i (W^K_{g(i)})^\top \Big] x'^\top$$

$$\boxed{\;W^{QK}_i = W^Q_i (W^K_{g(i)})^\top \in \mathbb{R}^{6656\times6656},\quad \operatorname{rank}\le 128\;}$$

The QK-norm contributes only a positive **scalar** per (token, head), so it leaves the
subspace structure of the fused matrix untouched. This is a cleaner situation than
DeepSeek/Kimi, where `q_a_layernorm` / `kv_a_layernorm` weight vectors had to be folded in.

**But there is a much more interesting consequence.** Because `qk_norm` renormalizes $q$ and
$k$ to unit RMS *at runtime*, the **overall magnitude of $W^Q_i$ and $W^K_g$ is functionally
irrelevant**. The model *cannot* suppress a head by shrinking its query or key weights — the
norm scales it straight back up. Only the *shape* of the spectrum (relative $\sigma$)
survives.

That is a sharp, falsifiable prediction: **the Case-1 "one side is uniformly small"
pathology cannot exist on Glimmer's QK side.** Any low fused-QK rank observed there must be
genuine subspace misalignment (Case 2). This alone makes the 13 NoPE layers worth the trip.

### 4.2 VO — blocked by the gate, in exactly the way RoPE blocks QK

$$o \;=\; \Big(\underbrace{(\alpha V)}_{z} \odot \sigma(x W^G_i)\Big) W^O_i
      \;=\; \alpha\,\Big( X\, W^V_{g(i)} \operatorname{diag}\!\big(\sigma(x W^G_i)\big) W^O_i \Big)$$

The sigmoid gate is a **data-dependent diagonal matrix in the 128-dim head space** — the
same structural position RoPE occupies between $W^Q$ and $W^K$. There is no exact
input-independent $W^{VO}$.

Three ways to proceed, all defensible, ideally all three:

1. **Ungated fusion** $W^{VO}_i = W^V_{g(i)} W^O_i$ (i.e. gate $\equiv 1$). This is the
   *capacity* view: the subspace the head is wired to route, before the gate decides how
   much of it to use. Directly comparable to the DS-R1/K2 numbers.
2. **Mean-gate fusion** $W^V_{g(i)} \operatorname{diag}(\bar g_i) W^O_i$ where
   $\bar g_i = \mathbb{E}_x[\sigma(x W^G_i)]$ over a calibration set. The *typical operating
   point* view — directly analogous to the fixed-offset RoPE trick in Appendix A.3.
3. **Gate statistics as a first-class result.** $\operatorname{diag}(g)$ is a.s. full rank,
   so gating cannot change the rank *bound* — but it can re-weight which of the 128 inner
   directions carry energy, which *is* the head-suppression mechanism. Per-head
   distributions of $\sigma(x W^G_i)$ (mean, fraction of channels with mean < 0.1, etc.) are
   cheap and probably as informative as the spectra.

### 4.3 The thesis this all points at

Glimmer has architecturally removed **both** weight-magnitude routes to head suppression:

| Suppression route | In DeepSeek/Kimi | In Glimmer |
|---|---|---|
| Shrink $W^Q$ or $W^K$ | available | **removed** by scale-free `qk_norm` |
| Shrink $W^V$ or $W^O$ | available | still available, but **the gate is cheaper** |
| Explicit gate | — | present (`gate_proj` + sigmoid) |

If the tweet's "weight RMS pinned at ~6e-3" is real, that is a third piece of the same
story: the recipe removes magnitude as a free parameter and forces the model to express
"turn this head down" somewhere other than the weights.

**So the headline question for this experiment is:** does Glimmer still show the low-rank
early-layer pattern? If it doesn't, the pattern in DeepSeek/Kimi was an *optimization
artifact* — a byproduct of leaving magnitude unconstrained — rather than a statement about
what the task needs. That would be the most valuable thing this could establish, and it
reframes the whole "wasted capacity" reading.

### 4.4 GQA changes the shared-subspace story

With only **2** V matrices and **32** O matrices per layer:

- All 16 query heads in a group read the value path through the *same* 128-dim slice of the
  residual stream. Row-space of $W^{VO}_i$ = row-space of $W^V_{g(i)}$, shared by 16 heads.
- A whole layer's value path reads the 6656-dim residual stream through at most
  **256 dimensions** (2 × 128). That is a structural bottleneck MLA didn't have.
- **Case 1 becomes correlated:** if $W^V_g$ has low effective rank, all 16 of its heads are
  dragged down simultaneously. Per-head scatter plots will show it as two flat plateaus
  rather than the smooth sorted curve seen in DS-R1.
- The interesting per-head variation is therefore on the $W^O$ / gate side. The stacked-head
  analysis is really only meaningful for $O$, $Q$, and $VO$ now.

---

## 5. The head-level norm question

Chris's tweet reply is a separate, cheap, self-contained measurement.

### 5.1 The identity that makes the question precise

Heads partition the rows of `q_proj` / `k_proj` / `v_proj` / `gate_proj` exactly, and the
columns of `o_proj` exactly, into equal-sized blocks. Therefore

$$\mathrm{RMS}(W)^2 \;=\; \frac{1}{n_h}\sum_{i} \mathrm{RMS}(W_i)^2$$

The whole-matrix RMS is exactly the **quadratic mean of the per-head RMS**. A constant
whole-matrix RMS is therefore perfectly compatible with per-head RMS varying by an order of
magnitude — the aggregate statistic simply cannot see it. That's the answer to "is the stat
true at a head level?": it has to be checked, and the aggregate gives no evidence either
way.

Worth reporting per layer, per matrix: per-head RMS spread (min / median / max, coefficient
of variation, max/min ratio), plus per-head $\sigma_1$ and stable rank
$\lVert W_i\rVert_F^2 / \sigma_1^2$.

### 5.2 "…and does that even matter?"

It matters differently for each matrix, and the architecture makes this unusually clean:

| Matrix | Does per-head scale matter downstream? |
|---|---|
| `q_proj`, `k_proj` | **No.** `qk_norm` divides it straight out. Only the spectrum *shape* survives. Scale is a pure gauge freedom here. |
| `v_proj` | **Yes.** Scale passes through attention and the gate into `o_proj` and the residual stream. |
| `o_proj` | **Yes.** Directly sets each head's write amplitude into the residual stream. |
| `gate_proj` | **Yes, and non-linearly.** $\lVert G_i \rVert$ sets how saturated the sigmoid is: large ⇒ the gate is a hard on/off switch; small ⇒ it sits near 0.5 and barely gates at all. |

Note there is an exact **gauge freedom** on the VO path: $W^V_g \to cW^V_g$,
$W^O_i \to W^O_i/c$ leaves the function unchanged (for all 16 heads of group $g$ at once).
So any per-matrix norm constraint the recipe imposes is *choosing a gauge*, and whether the
observed 6e-3 is a meaningful constraint or an artifact of the chosen gauge is itself a
question. The $q/k$ case is even more extreme — the scale is entirely unobservable.

Predicted-but-untested: if the recipe genuinely pins RMS per *matrix*, the per-head spreads
for `q_proj` should be the widest of the five (nothing downstream cares), and `o_proj` /
`gate_proj` the narrowest.

### 5.3 Three hypotheses about the 6e-3, and the statistic that separates them

The claim is that RMS $= 0.5/\sqrt{d_\text{model}}$ — note this depends **only on $d = 6656$**,
not on the individual matrix's fan-in/fan-out. That already discriminates a lot. Checking
the same statistic at three granularities separates the live hypotheses cheaply:

| Hypothesis | Prediction |
|---|---|
| **Hyperball** — fixed $\lVert W\rVert_F$ per matrix | Whole-matrix RMS constant at $0.5/\sqrt{d}$ **regardless of shape**; per-row and per-head RMS free to vary widely |
| **nGPT-style** — rows/columns normalized | Per-**row** (or per-column) norms nearly constant; whole-matrix RMS constant only as a consequence |
| **Spectral-norm condition** (e.g. plain Muon) | RMS varies with shape, roughly $\propto \sqrt{d_\text{out}/d_\text{in}}/\sqrt{\max(d_\text{out},d_\text{in})}$ — **not** a single constant |

The decisive comparison is `mlp.down_proj` (6656 × 19968) against `q_proj` (4096 × 6656):
wildly different shapes and fan-ins. If both land on 6.13e-3, it's an explicit RMS target.
If they differ in the direction $\sqrt{d_\text{out}/d_\text{in}}$ predicts, it's spectral.
(For reference: $1/\sqrt{6656} = 1.23\text{e-}2$; $1/\sqrt{19968} = 7.08\text{e-}3$ — close
enough to 6e-3 that fan-in-based schemes are *not* excluded by magnitude alone, which is
exactly why the cross-shape comparison is needed rather than eyeballing one matrix.)

Then the per-head decomposition of §5.1 sits underneath all three: whichever holds at the
matrix level, the head-level spread is an independent and unconstrained degree of freedom
(§1.1b).

### 5.4 Baseline / control

**Qwen3-8B** (`Qwen/Qwen3-8B`) is the natural control — it's the model in the tweet's
comparison plot, it's ~16 GB, and its architecture isolates variables nicely:

| | Glimmer 30B | Qwen3-8B |
|---|---|---|
| hidden / layers | 6656 / 52 | 4096 / 36 |
| q / kv heads, head_dim | 32 / 2, 128 | 32 / 8, 128 |
| RoPE | sliding layers only; 13 NoPE global layers | all layers, theta 1e6 |
| QK-norm | scale-**free**, shared | `q_norm`/`k_norm`, **with** learnable weight (length 128, foldable) |
| Attention output gate | **yes** | **no** |

So Qwen3-8B gives a **clean, exact $W^{VO}$ fusion** with no gate in the way — the direct
apples-to-apples comparison against the DS-R1 curves. It cannot give a QK fusion (full
RoPE). Glimmer gives a clean QK fusion (13 layers) but a gate-obstructed VO. Between the two
you can cover both sides.

---

## 6. The vocabulary-mean question

Chris:

> "We've noticed before in gpt-2 that in the early layers it rather quickly rotates the
> residual stream away from the strong vocabulary mean. That 'strong mean' detail makes me
> wonder if I'm misattributing the low rank to 'poor use of capacity' vs. there being a
> strong direction of some kind where understanding the behavior requires first viewing the
> vectors in terms of their offset from that strong shared direction."

This is a real confound, and the math resolves cleanly. Writing it out because the naive
version of the fix doesn't work.

**What doesn't work:** subtracting the mean from $W$. The mean is a property of the *input
distribution*, not of the weight matrix. $W$'s SVD doesn't know $\mu$ exists, and removing a
single direction from $W$ can change its effective rank by at most ~1.

**What the mean actually does** is make the input distribution anisotropic. Let
$x = \mu + \delta$ with $\lVert\mu\rVert \gg \lVert\delta\rVert$. The head's output is
$xW = \mu W + \delta W$: a **constant vector** plus the part that varies across tokens.

- Current metric — spectrum of $W$ — is an average-case measure over an **isotropic** input
  prior. It implicitly assumes every input direction is equally likely.
- Activation-weighted — spectrum of $M^{1/2}W$ where $M = \mathbb{E}[xx^\top] = \mu\mu^\top + \Sigma_\delta$ —
  **degenerates**. If $\mu$ dominates, $W^\top M W \approx (\mu W)^\top(\mu W)$, which is
  rank 1 *for every head*. This tells you nothing except that the mean must be removed.
- The meaningful metric is the **centered, activation-weighted** spectrum:
  $\Sigma_\delta^{1/2} W$. "How many directions does this head actually use across real
  tokens?"

And it cuts **both ways**, which is the interesting part:

1. If a head's *small* singular directions align with *high-variance* input directions, its
   true functional rank is **higher** than the weight-space number — the current metric is
   too pessimistic, and Chris's instinct is right.
2. If a head's *large* singular directions align with $\mu$ (nearly constant across tokens),
   then a chunk of its apparent rank is spent producing a **near-constant output vector**
   added to the residual stream at every position — a learned bias implemented inside the
   head. The head is *more* wasteful than it looks, not less.

Case 2 is very plausible for early layers and is the more interesting outcome, because it
converts "this head is low rank" into "**this head is mostly a bias**" — a far more
interpretable, more publishable claim, and one that connects to the attention-sink /
default-output literature. Two numbers per head capture it:

- **bias fraction:** $\lVert \mu W \rVert^2 \,/\, \mathbb{E}\lVert xW \rVert^2$
- **centered effective rank:** effective rank of $\Sigma_\delta^{1/2} W$

Practical notes: the relevant $x$ is the residual stream **after `input_layernorm`**, since
that is what `q/k/v/gate_proj` actually consume — and RMSNorm does *not* remove the mean, so
$\mu$ survives into the projection. A 6656×6656 covariance is 177 MB in fp32; 52 of them is
9 GB, fine on the A100 but stream them. A few hundred sequences of calibration text is
plenty for a 6656-dim second moment to be usable for this purpose.

There's also a Glimmer-specific wrinkle worth checking first: the **embedding mean** itself.
`tie_word_embeddings: false` with a 202k vocab — measure $\lVert\mu_\text{embed}\rVert$
relative to the mean row norm before assuming there's a strong shared direction at all.
Recipes that control weight norms may also have suppressed it.

---

## 7. Practical notes

### Compute

The fused matrices are $6656 \times 6656$ — much bigger than the DeepSeek case (where
$W^{VO}$ was $512 \times 7168$ because of the MLA latent). **Do not materialize them.**

For $A \in \mathbb{R}^{m\times r}$, $B \in \mathbb{R}^{r\times n}$ with $m, n \gg r$:
thin-QR both sides, $A = Q_A R_A$ and $B^\top = Q_B R_B$. Then
$AB = Q_A (R_A R_B^\top) Q_B^\top$ with $Q_A, Q_B$ orthonormal, so

$$\operatorname{svdvals}(AB) \;=\; \operatorname{svdvals}(R_A R_B^\top)$$

an **$r \times r$ = 128×128 SVD**. Two 6656×128 QRs plus a 128×128 SVD per head, instead of
a 6656×6656 SVD. This applies to both $W^{VO}$ and $W^{QK}$, and makes the whole model a
few minutes of GPU rather than hours. (Materializing all 32 fused VO matrices for one layer
in fp32 would be 5.7 GB; with the QR trick it's negligible.)

For the **stacked-head** analysis, accumulate the Gram matrix
$\sum_i (W^{VO}_i)^\top W^{VO}_i$ (6656×6656, 177 MB fp32) and take eigenvalues — sigmas are
the square roots. Remember the notebooks spectrally normalize each head ($/\sigma_1$) before
stacking.

Do the SVDs in **fp32 or fp64**, not bf16. The existing plotting notebook has a visible
`RuntimeWarning: overflow encountered in square` from squaring fp32 sigmas — worth computing
energy in float64.

### Weights

Attention-only tensors are ~4.4 B params ≈ 8.9 GB in bf16 across all 52 layers, so unlike
the DeepSeek work there is **no disk-management problem** — an 80 GB A100 can hold the whole
59.6 GB checkpoint, and `safetensors.safe_open` can pull individual tensors without loading
a shard fully. All the `free_up_space` / `retrieve_and_dequant_weight` machinery in
`Calculating Singular Values in Large MLA Models.md` is unnecessary here. Weights are plain
bf16 — **no FP8 dequantization needed** either.

Use `HF_TOKEN` from `env.sh` (sourced in the same command; never echo it).

### The box

A gpu-sniper profile exists: **`fused-attn-svd`** (`gpu-sniper/profiles.yaml` +
`profiles/fused_attn_svd_setup.sh`). 1×A100_80GB, pure-PyPI uv venv at
`~/shared-subspaces/.venv` — torch, safetensors, `huggingface_hub[hf_transfer]`, numpy,
scipy, matplotlib, pandas, accelerate. Nothing compiles; no vLLM, no flash-attn.

- `transformers` is installed **from git main** because Muse Glimmer support is
  5.15.0.dev0-only and in no release. Its install is best-effort and non-fatal: the SVD path
  never imports it. It *is* required for the activation-based work (gate statistics, §4.2/§6
  covariance), so check it landed before planning that phase.
- Both models are pre-pulled into `$HF_HOME` (`~/hf-cache`), guarded on free disk —
  the setup log says which ones actually downloaded, so read it rather than assuming.
- Setup asserts the layer-0 and layer-3 tensor shapes in §3.4 and prints the NoPE layer
  list and the layer-0 `q_proj` RMS vs. $0.5/\sqrt{d}$. If those asserts printed `PASS`,
  §3 of this document is confirmed against the actual checkpoint.
- Deliberately does **not** clone `agent-ops` — Chris scoped this project out of that
  methodology.

### Repo

- Branch `glimmer-fused-vo` off `main` in `shared-subspaces`.
- Notebooks live in `fused_attn_svd/`, authored as `.md` and converted with
  `colab_utils.py to-nb` / `upload` (see the `colab-utils` skill). `to-md` round-trips.
- Agent-ops methodology is explicitly **skipped** for this sub-project (Chris's call).

---

## 8. Open questions, roughly in order of value

1. **Does Glimmer show the low-rank early-layer pattern at all?** Ungated $W^{VO}$ across
   all 52 layers vs. the DS-R1 curves. This is the whole ballgame.
2. **Fused QK on the 13 NoPE layers.** First exact, full-head QK fusion available in any
   model studied here. Given scale-free `qk_norm`, any low rank found there is *necessarily*
   Case-2 misalignment. Clean result either way.
3. ~~Is the 6e-3 claim true, and is it true per head?~~ **Done — see §1.3.** Confirmed as a
   shape-independent per-matrix RMS target (Hyperball signature); *not* true per head, with
   the spread growing with depth and concentrated in `gate_proj`. What remains is the full
   depth profile (`norm_survey.py --layers all`) and, more importantly, whether per-head
   gate norm predicts fused VO rank.
4. **Gate statistics per head.** Mean/spread of $\sigma(x W^G_i)$ on calibration text. Are
   there heads that are simply switched off? Does gate-off correlate with low fused VO rank,
   or has it *replaced* low rank as the suppression mechanism?
5. **The bias/mean decomposition** of §6, at least on early layers, for both Glimmer and one
   older model where the low-rank pattern is known to be present (DS-R1 layer 2 or 8 is the
   obvious target, and its sigmas are already computed).
6. **GQA plateau structure.** Do the 16 heads sharing a KV head behave as a block?
7. **Qwen3-8B as the ungated VO control**, to separate "gated architecture" from
   "norm-controlled recipe."

---

## 9. Chris's standing view, for calibration

He has never been sure what the low-rank finding is *for*, beyond "seems like a bit of a
waste of capacity." Worth knowing when deciding what to report: the deliverable that would
actually move his thinking is **evidence about *why* the low rank is there**, not another
compression ratio. Specifically —

- Is it *functional* (early layers genuinely need few directions)?
- Is it *suppression* (the model turning a head off through weight magnitude)?
- Is it *bias* (the head's dominant directions are constant-output, per §6)?
- Is it an *optimization artifact* that a norm-controlled recipe removes?

Glimmer is unusually well-suited to separating these because it has independently removed
two of the four mechanisms by construction.
