# SharedSubspaceEncoder


Implements a custom Encoder model to explore the effects of adding an Output subspace to MLA.

The project includes code for pre-training the model and evaluating it on GLUE tasks (currently I've only evaluated on SST-2).

The implementation is compatible with the HuggingFace Transformers library and uses their Trainer class for pre-training, with results logged to Weights & Biases



## Architecture


**Overall**

Compared to BERT, the `SharedSpaceEncoder`:

* Uses RoPE for position information.
* Uses MLA in place of standard MHA.
* Uses SwiGLU for the FFNs (which adds gate neurons).
* Following DeepSeek-V3, LayerNorm is replaced by RMSNorm.

**Modifications to MLA**

The MLA implementation is based on the DeepSeekV3Attention class from HuggingFace transformers, but with several modifications:

* Optionally includes an output latent space (the primary experiment for this architecture).
* Removal of Decoder-specific details such as the key-value cache.
* Removal of the "MQA" configuration of RoPE; i.e., this model has a one-to-one ratio of queries and keys receiving position information. See the discussion on this further down.

The class also allows for running some or all layers as standard MHA to allow for:
* Comparison between MLA and MHA.
* Evaluating the benefits of including early MHA layers.


**Additional Subspace Options**

The model also includes additional optional shared-subspace-learning techniques to explore:

1. Linear decomposition of neuron weights in the FFNs.
2. Linear decomposition of the vocabulary embeddings.
3. Leaving a specified number of early layers as "dense":
    * Standard MHA instead of MLA
    * No decomposition 

Numbers (1) and (2) aren't currently explored in the experimental results, but there are a few runs relating to (3).




## Status of Experiments

The below should not be viewed as claims--there needs to be further validation of the code and results, and ablations need to be run.


In all experiments, we're using a 6-layer encoder with an embedding length of 256. There are 8 attention heads, all of size 32.

Below are the highest scoring configuriations of the three variants we're comparing.


| Baseline Name | Test Accuracy | Parameters | Latent Spaces | Position Encoding | # of RoPE Dims |
|---------------|---------------|------------|---------------------------------------------|-------------------|---------------------------|
| MHA         | 85.32         | 13.50M     | n/a                                         | PEVs              | n/a                       |
| MLA           | 84.75         | 12.67M     | 64 - 32                                     | RoPE              | 16                        |
| MLA-o         | 84.63         | 12.48M     | 64 - 32 - 64                                | RoPE              | 32                        |



"Latent Spaces" provides the size of the latent projections in the format

`(Query) - (Key-Value) - (Output)`

i.e., the current highest performing variant of MLA-o has:
* A shared query latent projection: `256 → 64`
* A shared key-value latent projection: `256 → 32`
* A shared output projection: `64 → 256`


Here is a summary of the current observations:

**Impact of Output Latent**

I only have two data points so far which directly compare MLA and MLAo under the same configurations. MLAo achieved higher SST-2 accuracy in one and MLA performed better in the other--see the section below on RoPE dimensions. 

In both, MLAo was slower (in terms of training samples per second), despite having fewer parameters. 

Before drawing any conclusions, I want to continue to explore:
* Different model sizes, latent space dimensions, and sequence lengths.
* The efficiency of the current code.

**RoPE vs. PEVs**

When using MLA, the scores on SST-2 increase significantly with the use of RoPE instead of learned position encoding vectors as in the original BERT. 

First, this needs to be compared to how much the baseline BERT benefits from RoPE. If the difference is significant, it implies that the PEVs do not work well with the added subspace projections. This would be a valuable insight to researchers exploring the use of attention latent spaces in encoders (such as the PoorViT paper).

**Number of RoPE Dimensions**

The scores change significantly based on the number of head dimensions which receive RoPE information.

* We're using a head size of 32, and have tried applying RoPE to all 32 dimensions, and to just 16 of them. 
* Oddly, the current results are conflicting regarding MLA vs. MLAo:
    * MLA scores higher when only 16 of 32 dimensions receive RoPE.
    * MLAo performs better when it's applied to all 32.
* This seems quite interesting if it isn't simply due to a bug or mistake, and if it holds up under multiple configurations.



### Experiment Notebook

Run results and discussion can be found in the `experiments.ipynb` notebook.



### Plans & Next Steps

**Sanity-Checking vs. HuggingFace BERT**

Before attempting any sweeps over the latent space sizes, I'd like to sanity check this model's baseline implementation against the BERT implementation in HuggingFace Transformers.

How do they compare in speed, and accuracy on SST-2? If there are substantial differences, those seem worth understanding before going further. 

**Varying Latent Sizes**

A logical next step would be to explore different combinations of the query, key-value, and output subspace sizes to see how performance changes. 

This would require: 
- Finish adding support for sweeps, and resuming from checkpoitns.
- Preferably, running on a dedicated instance, such as at Lambda, to avoid Colab disconnect issues.

It would also be nice to invest in a more efficient implementation, if possible, to speed up the experiments and better evaluate that aspect of MLAo.

**Decoder Experiments**

Before investing further in this custom implementation, it may be simplest to move on to an experiment based on an existing MLA-based Decoder model.

The output subspace is a very small change to the implementation of DeepSeek-V3 in the HuggingFace transformers library. Making this change and performing pre-training experiments on this known-good implementation may be a safer and more direct next step. 

The primary concern has been cost--I've stayed with small Encoder architectures so far because they can train in fewer steps, and it seems easier to measure the impact to downstream performance by evaluating on simpler benchmarks like GLUE. 


## Experiment Design

Pre-training experiments are run on wikitext-v103 for 50,000 steps with a batch size of 256. This equates to about 1.6 billion tokens seen, though I'm unclear on what precentage of these are padding.

In addition to standard metrics such as training loss and samples per second, the scripts report MLM accuracy as a potential measure of pre-training quality.

Downstream performance is measured by fine-tuning and testing on SST-2 using standard hyperparameter choices for that benchmark.

The experiments were run on Colab with a 40GB A100.

Depending on the experiment configuration, the pre-training runs take around 75 minutes, followed by SST-2 tuning which takes less than 10. 

## Structure


The files and folders are organized as below.




```python
subspace_encoder/
├── configs/             # Model and training hyperparameters
│   ├── mha_baseline.json    
│   ├── best_mla.json
│   └── best_mla_o.json
├── layers/              
│   ├── feedforward.py   # FFN with optional linear decomposition
│   ├── mla.py           # Customized version of Multihead Latent Attention
│   └── task_heads.py    # Heads for MLM training, classification, etc.
├── models/
│   ├── shared_space_encoder.py   # Model and Layers
│   └── shared_space_config.py    # SharedSubspaceConfig
├── scripts/
│   ├── fine_tune_glue.py      # Benchmark on GLUE -- currently just SST-2
│   ├── train.py               # Pre-training
│   └── run_experiment.ipynb   # Runs the training scripts within Colab
├── tests/
│   ├── test_config.json
│   └── test_shared_encoder.py    # Validates individual components
├── utils.py                      # Helper functions for printouts
```


## Usage


All of the parameters for model architecture and training configuration are stored in / communicated by the `.json` configuration files.

Launch a pre-training run with:

```bash
python scripts/train.py --config configs/mha_baseline.json
```

Or a fine-tuning run with:

```bash
python scripts/fine_tune_glue.py --config configs/mha_baseline.json
```

_Current configurations:_

These are the highest scoring configurations of the three variants:

- `best_mha.json` - Our encoder model configured with standard MHA and PEVs.
- `best_mla.json`  - Multihead Latent Attention (MLA)
- `best_mla_o.json`    - MLA with a shared output subspace.

Pre-training metrics are logged to wandb under the project `encoder-pretrain`, and fine-tuning metrics to `encoder-pretrain-sst2`.



