"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# `test_shared_encoder.py`
"""

import pytest
import sys
import json
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.shared_space_config import (
    SharedSpaceEncoderConfig,
    make_shorthand,
    get_config
)
    
from layers.mla import MultiheadLatentAttention, RotaryEmbedding

from layers.feedforward import SubspaceFeedForward

from layers.task_heads import (
    SharedSpaceEncoderForSequenceClassification,
    SharedSpaceEncoderForMaskedLM
)

from models.shared_space_encoder import (
    SharedSpaceEncoderModel,
    SharedSpaceEncoderLayer,
    DeepseekV3RMSNorm
)


"""

TODO - Re-implement these tests.

def test_mla_init_with_output_latent(cfg):
    # TODO - Adjust config to create output latent.
    attn = MultiheadLatentAttention(cfg, layer_idx=0)
    assert attn.o_a_proj.weight.shape == (cfg.o_latent_dim, cfg.num_attention_heads * cfg.v_head_dim)
    assert attn.o_b_proj.weight.shape == (cfg.hidden_size, cfg.o_latent_dim)


def test_layer_initialization_dense(cfg):
    # TODO - Adjust config as needed

    layer = SharedSpaceEncoderLayer(cfg, layer_idx=0)

    # attention block
    assert isinstance(layer.self_attn, MultiheadLatentAttention)
    assert layer.attn_dropout.p == cfg.hidden_dropout_prob
    assert isinstance(layer.attn_norm, torch.nn.LayerNorm)

    # dense attention uses a single output projection
    assert hasattr(layer.self_attn, "o_proj")
    assert not hasattr(layer.self_attn, "o_a_proj")

    # dense FFN should not define shared weights
    assert not hasattr(layer.mlp, "W_in_shared")


def test_layer_with_subspaces(cfg):
    
    # TODO - Adjust config as needed.

    layer = SharedSpaceEncoderLayer(cfg, layer_idx=1)

    # output subspace creates the two projection matrices
    assert hasattr(layer.self_attn, "o_a_proj")
    assert hasattr(layer.self_attn, "o_b_proj")
    assert not hasattr(layer.self_attn, "o_proj")

    # decomposed FFN creates shared weights
    assert hasattr(layer.mlp, "W_in_shared")
    assert layer.mlp.W_in_shared.weight.shape == (cfg.ffn_rank, cfg.hidden_size)
"""

import os

base_config = "test_config.json"

# Track down the base_confg.
if os.path.exists(base_config):
    pass 
elif os.path.exists("tests/" + base_config):
    base_config = "tests/" + base_config 
elif os.path.exists("encoder-pretrain/tests/" + base_config):
    base_config = "encoder-pretrain/tests/" + base_config 
elif os.path.exists("shared-subspaces/encoder-pretrain/tests" + base_config):
    base_config = "shared-subspaces/encoder-pretrain/tests" + base_config


def main():
    """
    #### MLA
    """

    import torch
    import torch.nn.functional as F

    # Load the .json file with all settings.
    # full_cfg contains additional settings for training.
    # model_config is an instance of subencoder `*Config`
    full_cfg, model_cfg = get_config(base_config)

    # Print out its shorthand name.
    print(f"\nModel config:")
    print(f"  {full_cfg['shorthand']}\n")

    # 1. Forward‑shape test
    print(f"Creating data...", flush=True)
    batch_size = 2
    seq_len = 65
    layer_idx = 4

    model_cfg.num_hidden_layers = layer_idx + 1
    model_cfg.num_dense_layers = 1

    # Random input
    x = torch.randn(
        batch_size,
        seq_len,
        model_cfg.hidden_size
    )

    # Position embeddings are a tuple of
    #     R_cos [seq_len, rope_dims]
    #     R_sin [seq_len, rope_dims]
    #R_cos = self.rope.cos[:seq_len]
    #R_sin = self.rope.sin[:seq_len]

    # 65 tokens, 128 rope dims
    R_cos = torch.randn(seq_len, model_cfg.rope_dims)
    R_sin = torch.randn(seq_len, model_cfg.rope_dims)

    position_embeddings = (R_cos, R_sin)

    # [batch_size,    1,    1, seq_len]
    attention_mask = torch.ones(batch_size, 1, 1, seq_len)

    # Instantiate.
    print("Creating attention layer...", flush=True)
    attn = MultiheadLatentAttention(
        config = model_cfg,
        layer_idx = layer_idx,
    )

    # Evaluate.
    print("Evaluating...")
    y = attn(
        x,
        position_embeddings = position_embeddings,
        attention_mask = attention_mask
    )

    # The output shape should match the input shape.
    assert y.shape == x.shape

    # 2. Gradient test
    print("Verifying gradients...")
    y.mean().backward()

    # For each parameter,
    for p in attn.parameters():

        # Ensure it received a gradient.
        assert p.grad is not None

    print("Done!")

    """#### FFN"""



    # Load the .json file with all settings.
    # full_cfg contains additional settings for training.
    # model_config is an instance of subencoder `*Config`
    full_cfg, model_cfg = get_config(base_config)

    # Print out its shorthand name.
    print(f"\nModel config:")
    print(f"  {full_cfg['shorthand']}\n")

    # 1. Forward‑shape test
    print(f"Creating data...", flush=True)
    batch_size = 2
    seq_len = 65

    print(model_cfg.ffn_decompose)

    # Random input
    x = torch.randn(
        batch_size,
        seq_len,
        model_cfg.hidden_size
    )

    print("======== Decomposed FFN ========")

    model_cfg.ffn_decompose = True
    model_cfg.ffn_rank = 128
    model_cfg.num_hidden_layers = 5
    model_cfg.num_dense_layers = 1

    # Instantiate.
    print("Creating Decomposed FFN...", flush=True)
    ffn = SubspaceFeedForward(
        config = model_cfg,
        layer_idx = 4, # Should be decomposed
    )

    # Evaluate.
    print("Evaluating...")
    y = ffn(x)

    # The output shape should match the input shape.
    assert y.shape == x.shape

    # 2. Gradient test
    print("Verifying gradients...")
    y.mean().backward()

    # For each parameter,
    for p in ffn.parameters():
        print(p.size())

        # Ensure it received a gradient.
        assert p.grad is not None

    print("Done!")

    print("======== Dense FFN ========")

    print("Creating Decomposed FFN...", flush=True)

    model_cfg.num_dense_layers = 1
    model_cfg.ffn_decompose = True

    ffn = SubspaceFeedForward(
        config = model_cfg,
        layer_idx = 0,
    )

    # Evaluate.
    print("Evaluating...")
    y = ffn(x)

    # The output shape should match the input shape.
    assert y.shape == x.shape

    # 2. Gradient test
    print("Verifying gradients...")
    y.mean().backward()

    # For each parameter,
    for p in ffn.parameters():
        # Print the parameter name and shape
        print(p.size())
        # Ensure it received a gradient.
        assert p.grad is not None

    print("Done!")

    """#### Layer"""

    # Load the .json file with all settings.
    # full_cfg contains additional settings for training.
    # model_config is an instance of subencoder `*Config`
    full_cfg, model_cfg = get_config(base_config)

    # Print out its shorthand name.

    #print(f"  {full_cfg['shorthand']}\n")

    # ======== Configuration ========
    batch_size = 2
    seq_len = 65

    model_cfg.ffn_decompose = True
    model_cfg.ffn_rank = 128
    model_cfg.num_hidden_layers = 5
    model_cfg.num_dense_layers = 1

    print(f"\nModel config:")
    print(f"    {make_shorthand(model_cfg)}\n")

    # ======== Input Data ========

    print(f"Creating data...", flush=True)

    # Position embeddings are a tuple of
    #     R_cos [seq_len, rope_dims]
    #     R_sin [seq_len, rope_dims]
    #R_cos = self.rope.cos[:seq_len]
    #R_sin = self.rope.sin[:seq_len]

    # 65 tokens, 128 rope dims
    R_cos = torch.randn(seq_len, model_cfg.rope_dims)
    R_sin = torch.randn(seq_len, model_cfg.rope_dims)

    position_embeddings = (R_cos, R_sin)

    # [batch_size,    1,    1, seq_len]
    attention_mask = torch.ones(batch_size, 1, 1, seq_len)

    # Random input
    x = torch.randn(
        batch_size,
        seq_len,
        model_cfg.hidden_size
    )

    # ======== Create Layer ========
    layer = SharedSpaceEncoderLayer(
        model_cfg,
        layer_idx = 2
    )

    # ======== Evaluate ========

    # Evaluate.
    print("Evaluating...")
    y = layer(
        x,
        position_embeddings,
        attention_mask
    )

    # The output shape should match the input shape.
    assert y.shape == x.shape

    # 2. Gradient test
    print("Verifying gradients...")
    y.mean().backward()

    # For each parameter,
    for p in layer.parameters():
        # Print the parameter name and shape
        print(p.size())
        # Ensure it received a gradient.
        assert p.grad is not None

    print("Done!")

    """#### Model"""

    # Load the .json file with all settings.
    # full_cfg contains additional settings for training.
    # model_config is an instance of subencoder `*Config`
    full_cfg, model_cfg = get_config(base_config)

    # Print out its shorthand name.

    #print(f"  {full_cfg['shorthand']}\n")

    # ======== Configuration ========
    batch_size = 2
    seq_len = 65

    model_cfg.ffn_decompose = True
    model_cfg.ffn_rank = 128
    model_cfg.num_hidden_layers = 5
    model_cfg.num_dense_layers = 1

    print(f"\nModel config:")
    print(f"    {make_shorthand(model_cfg)}\n")

    # ======== Input Data ========

    print(f"Creating data...", flush=True)

    # [batch_size,    1,    1, seq_len]
    #attention_mask = torch.ones(batch_size, 1, 1, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)

    # Random input_ids [batch_size, seq_len]
    input_ids = torch.randint(model_cfg.vocab_size, (batch_size, seq_len))

    # ======== Create Model ========
    model = SharedSpaceEncoderModel(
        model_cfg
    )

    # ======== Evaluate ========

    # Evaluate.
    print("Evaluating...")
    y = model(
        input_ids,
        attention_mask
    )

    # The output shape should match the input shape.
    assert y.shape == x.shape

    # 2. Gradient test
    print("Verifying gradients...")
    y.mean().backward()

    # For each parameter,
    for p in model.parameters():
        # Print the parameter name and shape
        #print(p.size())
        # Ensure it received a gradient.
        assert p.grad is not None

    print("Done!\n")

    print(model)

    """#### `MaskedLM`"""

    # ======== Load Configuration ========
    full_cfg, model_cfg = get_config(base_config)

    # Modify the config for test
    model_cfg.ffn_decompose = True
    model_cfg.ffn_rank = 128
    model_cfg.num_hidden_layers = 5
    model_cfg.num_dense_layers = 1

    print(f"\nModel config:")
    print(f"    {make_shorthand(model_cfg)}\n")

    # ======== Input Data ========
    batch_size = 2
    seq_len = 65

    print("Creating data...", flush=True)

    # Attention mask [batch_size, 1, 1, seq_len]
    #attention_mask = torch.ones(batch_size, 1, 1, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)

    # Random input_ids and labels [batch_size, seq_len]
    input_ids = torch.randint(model_cfg.vocab_size, (batch_size, seq_len))
    labels = torch.randint(model_cfg.vocab_size, (batch_size, seq_len))

    # ======== Create Model ========
    model = SharedSpaceEncoderForMaskedLM(model_cfg)

    # Put model in eval mode (optional for test)
    model.eval()

    # ======== Forward Pass ========
    print("Evaluating...")

    # Forward pass (loss + logits are returned in a dict-like object)
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask
    )

    # Get the loss and logits
    loss = outputs.loss
    logits = outputs.logits

    # Check shape: logits should match [batch_size, seq_len, vocab_size]
    assert logits.shape == (batch_size, seq_len, model_cfg.vocab_size), f"Unexpected shape: {logits.shape}"

    # ======== Gradient Check ========
    print("Verifying gradients...")

    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    print("Sanity check passed!\n")

    """#### `SequenceClassification`"""

    # ======== Load Configuration ========
    full_cfg, model_cfg = get_config(base_config)

    # Modify the config for test
    model_cfg.ffn_decompose = True
    model_cfg.ffn_rank = 128
    model_cfg.num_hidden_layers = 5
    model_cfg.num_dense_layers = 1
    model_cfg.num_labels = 2  # For SST-2

    print(f"\nModel config:")
    print(f"    {make_shorthand(model_cfg)}\n")

    # ======== Input Data ========
    batch_size = 2
    seq_len = 65

    print("Creating data...", flush=True)

    attention_mask = torch.ones(batch_size, seq_len)
    input_ids = torch.randint(model_cfg.vocab_size, (batch_size, seq_len))

    # For classification: labels should be [batch_size], one class per sequence
    labels = torch.randint(0, model_cfg.num_labels, (batch_size,))

    # ======== Create Model ========
    model = SharedSpaceEncoderForSequenceClassification(model_cfg)

    # Put model in eval mode (optional for test)
    model.eval()

    # ======== Forward Pass ========
    print("Evaluating...")

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask
    )

    loss = outputs.loss
    logits = outputs.logits

    # For classification: [batch_size, num_labels]
    assert logits.shape == (batch_size, model_cfg.num_labels), f"Unexpected shape: {logits.shape}"

    # ======== Gradient Check ========
    print("Verifying gradients...")

    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"

    print("Sanity check passed!\n")


if __name__ == "__main__":
    main()
