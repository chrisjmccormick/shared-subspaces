# -*- coding: utf-8 -*-

"""# sparse_moe_decoder.py"""

from typing import Optional

import torch
from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

from ..layers.mla import MultiheadLatentAttention, RotaryEmbedding
from ..layers.feedforward import SparseMoEFeedForward
from .shared_space_config import SparseMoEDecoderConfig
from ..utils import DeepseekV3RMSNorm, create_norm_layer as base_create_norm_layer

def build_norm_layer(hidden_size: int, config: SparseMoEDecoderConfig) -> nn.Module:
    """
    Helper wrapper around :func:`~moe.utils.create_norm_layer` that plugs in the
    correct epsilon value based on the configuration.
    """
    if config.norm_type == "layernorm":
        eps = config.layer_norm_eps
    elif config.norm_type == "rmsnorm":
        eps = config.rms_norm_eps
    else:
        raise ValueError(f"Unknown norm_type: {config.norm_type}")
    return base_create_norm_layer(hidden_size, config.norm_type, eps)

"""#### *PreTrainedModel"""


class SparseMoEDecoderPreTrainedModel(PreTrainedModel):
    """
    The **PreTrainedModel object:
      - Is instantiated when TODO
      - Initializes:
        - TODO
      - Provides access to TODO
      - Executes TODO
    """

    config_class = SparseMoEDecoderConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module) -> None:
        """Weight initialization hook used by :class:`PreTrainedModel`.

        ``PreTrainedModel.post_init`` will recursively apply this function to
        every submodule right after construction.  HuggingFace models override
        it so that creating a model from scratch yields the same initialization
        as ``from_pretrained`` when no checkpoint is supplied.

        This decoder-specific initialization strategy includes:
        - Proper handling of configurable normalization layers (LayerNorm or RMSNorm)
        - Special initialization for language modeling heads
        - Considerations for causal attention and autoregressive modeling
        - Support for both dense and decomposed vocabulary embeddings
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

"""#### `*Layer`"""


class SparseMoEDecoderLayer(nn.Module):
    """
    The **Layer object:
      - Is instantiated by :class:`SparseMoEDecoder` for each
        Transformer block in the decoder.
      - Initializes:
        - ``self_attn`` – multi-head latent attention implementing either
          dense or latent projections depending on the configuration.
        - ``ffn`` – a :class:`SparseMoEFeedForward` block replacing the dense FFN.
        - RMSNorm layers for pre-attention and pre-FFN normalization.
      - Provides access to the attention and feed-forward submodules via the
        attributes ``self_attn`` and ``ffn``.
      - Executes a single decoder block in :meth:`forward`.
    """

    def __init__(self, config: SparseMoEDecoderConfig, layer_idx: int) -> None:

        super().__init__()

        # Norm applied prior to attention.
        self.attn_input_norm = build_norm_layer(config.hidden_size, config)
        
        # Attention block
        self.self_attn = MultiheadLatentAttention(config, layer_idx)

        # Norm applied prior to FFN
        self.ffn_input_norm = build_norm_layer(config.hidden_size, config)

        # Feed-forward network used after attention
        self.ffn = SparseMoEFeedForward(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor], # RoPE embeddings
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:

        # ========================
        #     Self Attention
        # ========================
        residual_strm = hidden_states

        # Normalize the hidden states to create the input to attention.
        attn_input = self.attn_input_norm(hidden_states)

        # Evaluate
        attn_output = self.self_attn(
            attn_input,
            position_embeddings,
            attention_mask,
        )

        # Add the attention output (the residual) back to the non-normalized
        # hidden_states.
        hidden_states = residual_strm + attn_output

        # ===========================
        #     Feed-Forward Network
        # ===========================
        residual_strm = hidden_states

        # Normalize the updated hidden states prior to the FFN
        ffn_input = self.ffn_input_norm(hidden_states)

        # Evaluate
        ffn_output = self.ffn(ffn_input)

        # Add the output the un-normalized hidden states.
        hidden_states = residual_strm + ffn_output

        return hidden_states

"""#### *Model"""


class SparseMoEDecoder(SparseMoEDecoderPreTrainedModel):
    """
    The **Model object:
      - Initializes:
        - The vocabulary embeddings (and optional decomposition)
        - Position embeddings (calculated in RotaryEmbedding)
        - All of the **Layer objects.
      - Provides interface to vocab embeddings.
      - Executes the whole decoder model in `forward` with causal attention.

      This is the base decoder without the language modeling head.
      Attach an LM head or task-specific head as needed.
    """

    def __init__(self, config: SparseMoEDecoderConfig) -> None:
        super().__init__(config)

        # ============================
        #    Vocabulary Embeddings
        # ============================
        # Decomposing the vocabulary (if enabled) defines a shared projection
        # which constrains the model to store semantic information (and
        # whatever other static token knowledge) into a limited set of
        # feature directions.

        # If we're decomposing the token embeddings,
        # TODO - Rename to vocab_subspace.
        if config.vocab_subspace:

            # Create the embedding table. Vocabulary embeddings are learned
            # in a lower dimensional latent space.
            self.vocab_embed = nn.Embedding(
                config.vocab_size, # Number of tokens
                config.vocab_rank  # Subspace dimension
            )

            # Create a
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
                SparseMoEDecoderLayer(
                    config,
                    layer_idx = i
                )
            )

        # Wrap in torch ModuleList
        self.layers = nn.ModuleList(layers)

        # Whatever huggingface does behind the scenes...
        self.post_init()

    # Agents: Do not define boilerplate helpers, e.g., get/set_input_embeddings


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
        #  Input
        #     cos: [max_seq_len, rope_dims]
        #     sin: [max_seq_len, rope_dims]
        #  Outputs:
        #     R_cos [seq_len, rope_dims]
        #     R_sin [seq_len, rope_dims]
        R_cos = self.rope.cos[:seq_len]
        R_sin = self.rope.sin[:seq_len]


        # ===============================
        #   Attention Mask Conversion
        # ===============================

        """
        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )
        """

        # Expand the attention mask
        #if use_sdpa_attention_masks and attention_mask.dim() == 2:
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
