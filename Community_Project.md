# Adding an Output Latent to MLA

I've made good progress on this project, but there's still a lot to explore, and I'd love to collaborate on it.

Something that fascinates me about Multihead Latent Attention in models like DeepSeek-R1 and Kimi-K2 is how aggressively they manage to bottleneck the inputs to the attention layer--from their model size of 7,168 down to just 512 for all of the key and value heads in a layer, and 1,536 for the query heads.

So here's my research question: If these bottlenecks work well on the input of attention, what impact does it have on performance if we add one to the output, too? 

![Block diagram of output latent](https://lh3.googleusercontent.com/d/1Hh95gVcbyyWpn-vNo6Mx4equVBdjjLcZ)

In the diagram, the trapezoids represent shared "latent space" projections. MLA has a projection for a query latent space (on top), and a projection for a key-value latent space (on bottom). 

The idea here is to complete the symmetry and add one to the output projection as well. We split the output matrix W^O, into:

1. A per-head (i) projection, W^OA_i, that projects the value vector up to an output latent space.
    * e.g., from `head_dim → (model_dim / 4)`.
2. The output latents are summed across the heads.
3. The combined latent is projected back to model space, via a shared projection W^OB 
    * e.g., from `(model_dim / 4)  →  model_dim`.

I'm fond of the language of mechanistic interpretability--they frame the token vector as a communication bus that all of the model components are attached to. The specific term for the token vector is "the residual stream". 

In that framing, MLA restricts how much of the stream the query, key, and value heads can read from. Here, we're also constricting where they can write to.

### What's the Point?

Shared subspaces are interesting; on the one hand they risk:

* Reducing head diversity--not enough input for the heads to learn to specialize.
* Destructive interference--too many heads trying to work in too small of a space, stepping on each other.

But at the same time, they can mean:

* Shared learning--shared subspaces receive a weight update from every token, whereas heads need to be attended to to learn.
* Re-use--it may motivate the model to re-use existing pathways, helping avoid redundancy and leading to more shared learning.

I'm wondering if, under the right circumstances and configuration, an output latent space could actually benefit the model in some way beyond just reducing parameter count.

Here's where I've gotten so far.

### SVD Analysis

First, I looked at the singular values for all of the output heads in DeepSeek-V3 and Kimi-K2 to get a sense of their "effective rank". Maybe we could decompose them and find a shared subspace?

These are the "effective ranks" of the concatenated output heads for each of the 61 layers in Kimi-K2. The top line is "how many of the dimensions do you have to keep in order to have less than 0.1% reconstruction error?", and the bottom is if you loosen that to 1% error.

![Plot of Kimi-K2 output head effective ranks](https://lh3.googleusercontent.com/d/1d7ohGvkOonAaq9NCGso9W8Se4wOrQU_a)

Other than the early layers, I'm not sure the amount we could safely truncate is worthwhile--you have to cut the rank down to ~5,000 just to break even on parameter count.

That could be worth exploring more, but I'd also expect this idea to work best with the constraint put in place prior to pre-training, (instead of trying to impose it on an existing model like Kimi-K2, where the output heads learned without any constraint on where they could write to), so that's where my energy's gone mostly.

### Pre-Training Experiments

Next, I tried implementing this on some tiny-scale encoder models. They seem like a good starting point because they can train faster and (I think?) are easier to benchmark. The goal is to explore the concept enough to justify the more expensive Decoder experiments.

I modified a Vision Transformer to use MLA, and trained it from scratch on CIFAR-10. It did well enough there to warrant looking at further, so I moved on to training a tiny BERT model from scratch.

I created a custom encoder model to test out standard Multihead Attention (MHA) vs. MLA vs. "MLA-o" (MLA with an output latent space). 

The variants I tested all had 6-layers, embedding length of 256, 8 attention heads, and a head size of 32. I ran them for 50,000 steps with a batch size of 256 on WikiText-v103, and then fine-tuned and tested them on SST-2.

| # | Attention | Test Accuracy | Parameters | Query Latent | Key-Value Latent | Output Latent | Position Encoding | # of RoPE Dims |
|:-:|:---------:|:-------------:|:----------:|:------------:|------------------|---------------|:-----------------:|:--------------:|
| 1 | MHA       | 85.67         | 13.50M     | n/a          | n/a              | n/a           | RoPE              | 32             |
| 2 | MLA       | 84.75         | 12.67M     | 64           | 32               | n/a           | RoPE              | 16             |
| 3 | MLA-o     | 84.63         | 12.48M     | 64           | 32               | 64            | RoPE              | 32             |

Number 1 is standard multihead attention, it scored the highest. 

For the MLA variants, I chose the latent space sizes based on:

* "What seemed to work well" for the Vision Transformer (where I was able to iterate faster)
* The loose pattern from DeepSeek that:
  * (1) The latents can be much smaller than the token vector, and 
  * (2) That the query space should be larger than the key-value space.
* The conjecture that the query and output spaces might need similar capacity.

I haven't gotten to explore these choices much. These are just the best performing configurations of MLA and MLA-o that I have results for.

The output subspace slightly decreased the accuracy, and only reduced the parameter count by ~1.5%. Also, at this scale, I'm guessing the overhead of the additional kernel launch outweights the reduction in operations, because MLA-o was slower, too. 

So it's not looking particularly exciting from the numbers I have so far. But I'd still like to explore more parameter combinations, and identify what needs to be true about the architecture in order for this to increase throughput.

## Community Project

**What am I hoping for?**

A paper would be great if the results are worthy of one, but I don't want to try and do that alone. If not a paper, I'll post about it. 

(Also, to be blunt--I'm hoping to land a research role somewhere, so this is also about building experience and credibility!)

**What's next?**

I'm open to input! 

Some ideas:

1. Move on to pre-training a tiny Decoder model. 
    * It's only a few lines of code to add an output subspace to the DeepSeekV3 implementation in the transformers library. 
    * That could be a cleaner way to get results than something more custom.
2. Throughput experiments.
    * Take a single `DeepSeekV3Attention` layer and run random data through it for a few seconds under different configurations (sequence lengths, embedding size, head count / size, latent space size) and see what impact the output projection has.
3. Subspace alignment.
    * Kind of a tangent into interpretability...
    * If the key and value heads do well working together within a common subspace, is there similar alignment between the output heads and other spaces? 
4. Rank Reduction on Kimi-K2 (separate project?)
    * It's not a huge win, but it does seem like we could shave off a decent percentage of attention head parameters in the first ~15 layers. Take away its "1 trillion parameter" trophy. :-P
    * Also, while a single shared output subspace might be too constricting, clustering the heads and grouping them into multiple shared subspaces per layer might work.


I'd welcome any feedback, perspecitves, and general discussion of the ideas. And if you're really interested, any analysis or experimental results you're willing to contribute would be awesome. Looking forward to it!