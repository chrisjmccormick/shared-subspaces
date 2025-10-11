# -*- coding: utf-8 -*-

"""# linkspace_decoder.py

LinkedSpaceDecoder model with flexible space-to-module mappings.
"""

from typing import Optional

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

from layers.linkspace_mla import RotaryEmbedding
from layers.linkspace_layer import LinkedSpaceDecoderLayer, DeepseekV3RMSNorm
from models.linkspace_config import LinkedSpaceDecoderConfig

"""#### *PreTrainedModel"""

class LinkedSpaceDecoderPreTrainedModel(PreTrainedModel):
    """
    The **PreTrainedModel object for LinkedSpaceDecoder.
    """

    config_class = LinkedSpaceDecoderConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module) -> None:
        """Weight initialization hook used by :class:`PreTrainedModel`.

        ``PreTrainedModel.post_init`` will recursively apply this function to
        every submodule right after construction.
        """

        if isinstance(module, nn.Linear):
            # Standard linear layer initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
        elif isinstance(module, DeepseekV3RMSNorm):
            # RMSNorm initialization: weight to 1.0, no bias term
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm initialization: bias to 0, weight to 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# Classes
"""

"""#### *Model"""

class LinkedSpaceDecoderModel(LinkedSpaceDecoderPreTrainedModel):
    """
    The LinkedSpace decoder model (without language modeling head).
    
    Implements flexible space-to-module mappings for attention and FFN.
    """

    def __init__(self, config: LinkedSpaceDecoderConfig) -> None:
        super().__init__(config)

        # ============================
        #    Vocabulary Embeddings
        # ============================

        # If we're decomposing the token embeddings,
        if config.vocab_subspace:

            # Create the embedding table. Vocabulary embeddings are learned
            # in a lower dimensional latent space.
            self.vocab_embed = nn.Embedding(
                config.vocab_size, # Number of tokens
                config.vocab_rank  # Subspace dimension
            )

            # Selected token latents will be projected up to model size.
            # vocab_proj has shape [vocab_rank x model_size]
            self.vocab_proj = nn.Linear(
                config.vocab_rank,  # Size of latents
                config.hidden_size, # Model size
                bias=False
            )

        # Otherwise, for a dense vocabulary,
        else:
            # Create the dense embedding table in model space.
            self.vocab_embed = nn.Embedding(
                config.vocab_size,  # Number of tokens
                config.hidden_size  # Model size
            )

            self.vocab_proj = None

        # =====================
        #   RoPE Embeddings
        # =====================

        # Pre-computes the table of RoPE embeddings, leaving them in
        # GPU memory.
        self.rope = RotaryEmbedding(config)

        # ===================
        #    Create Layers
        # ===================

        layers = []

        # For each layer,
        for i in range(config.num_hidden_layers):
            # Create a **Layer, providing the config and indicating its number.
            layers.append(
                LinkedSpaceDecoderLayer(
                    config,
                    layer_idx = i
                )
            )

        # Wrap in torch ModuleList
        self.layers = nn.ModuleList(layers)

        # Whatever huggingface does behind the scenes...
        self.post_init()


    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Return token embeddings for input ids.
        This will perform the up projection to model space if the vocabulary is
        decomposed.

        input_ids have shape [batch_size, seq_len]
        """

        # If the vocabulary is decomposed,
        if self.vocab_proj is not None:

            # Retrieve the latents
            #  input_ids: [batch_size, seq_len]
            #          x: [batch_size, seq_len, latent_dim]
            x = self.vocab_embed(input_ids)

            #  Project the latents back to model space and return.
            return(self.vocab_proj(x))

        # If the vocabulary is dense,
        else:
            # Just return the embeddings.
            return self.vocab_embed(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the full decoder stack with causal attention.

        Inputs:
            input_ids       [batch_size, seq_len]
            attention_mask  [batch_size, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            Final decoder layer output   [batch_size, seq_len, model_size]
        """

        # Retrieve the token embeddings for this sequence.
        # These are model_size, regardless of whether the vocab is decompd.
        hidden_states = self.embed(input_ids)

        # Retrieve the rotary position embeddings for all of the positions in
        # our current input sequence.

        seq_len = hidden_states.size(1)

        # Retrieves just the ones necessary for the sequence length of the
        # input. These are vectors, two per token. Their length is the
        # number of head dimensions we're applying RoPE to.
        R_cos = self.rope.cos[:seq_len]
        R_sin = self.rope.sin[:seq_len]


        # ===============================
        #   Attention Mask Conversion
        # ===============================

        # Expand the attention mask
        if True:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask,
                hidden_states.dtype,
                tgt_len = seq_len
            )
            attention_mask = extended_attention_mask


        # Run the model!

        # For each decoder layer,
        for layer_i, layer in enumerate(self.layers):

            # Evaluate the layer
            hidden_states = layer(
                hidden_states,       # Token embeddings
                (R_cos, R_sin),      # Rope embeddings, passed as a tuple.
                attention_mask,      # Attn mask
            )

        # Return the final output of the decoder stack.
        return hidden_states

