# -*- coding: utf-8 -*-

"""# shared_subspace_encoder.py"""

from typing import Optional

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

from layers.mla import MultiheadLatentAttention, RotaryEmbedding
from layers.feedforward import SubspaceFeedForward
from models.shared_space_config import SharedSpaceEncoderConfig

"""`RMSNorm`

From:
https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py

TODO - May not need?
"""

class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

"""#### *PreTrainedModel"""

class SharedSpaceEncoderPreTrainedModel(PreTrainedModel):
    """
    The **PreTrainedModel object:
      - Is instantiated when TODO
      - Initializes:
        - TODO
      - Provides access to TODO
      - Executes TODO
    """

    config_class = SharedSpaceEncoderConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module) -> None:
        """Weight initialization hook used by :class:`PreTrainedModel`.

        ``PreTrainedModel.post_init`` will recursively apply this function to
        every submodule right after construction.  HuggingFace models override
        it so that creating a model from scratch yields the same initialization
        as ``from_pretrained`` when no checkpoint is supplied.

        The modules themselves come with PyTorch defaults; this method simply
        enforces the initializer scheme used throughout the library.  It is not
        required, but leaving it out would lead to slightly different weight
        statistics.
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# Classes
"""

"""#### `*Layer`"""

class SharedSpaceEncoderLayer(nn.Module):
    """
    The **Layer object:
      - Is instantiated by :class:`SharedSpaceEncoderModel` for each
        Transformer block in the encoder.
      - Initializes:
        - ``self_attn`` – multi-head latent attention implementing either
          dense or latent projections depending on the configuration.
        - ``mlp`` – a :class:`SubspaceFeedForward` block.
        - Dropout and LayerNorm used for the residual connection around the
          attention block.
      - Provides access to the attention and feed-forward submodules via the
        attributes ``self_attn`` and ``mlp``.
      - Executes a single encoder block in :meth:`forward`.
    """

    def __init__(self, config: SharedSpaceEncoderConfig, layer_idx: int) -> None:

        super().__init__()

        # Norm applied prior to attention.
        self.attn_input_norm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Attention block
        self.self_attn = MultiheadLatentAttention(config, layer_idx)

        # Norm applied prior to FFN
        self.ffn_input_norm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Feed-forward network used after attention
        self.ffn = SubspaceFeedForward(config, layer_idx)

        # Currently going with RMSNorm
        #self.attn_norm = nn.LayerNorm(
        #    config.hidden_size,
        #    eps=getattr(config, "layer_norm_eps", getattr(config, "eps", 1e-5)),
        #)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor, # RoPE embeddings
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:


        # ========================
        #     Self Attention
        # ========================

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
        hidden_states = hidden_states + attn_output

        # ===========================
        #     Feed-Forward Network
        # ===========================

        # Normalize the updated hidden states prior to the FFN
        ffn_input = self.ffn_input_norm(hidden_states)

        # Evaluate
        ffn_output = self.ffn(ffn_input)

        # Add the output the un-normalized hidden states.
        hidden_states = hidden_states + ffn_output

        return hidden_states

"""#### *Model"""

class SharedSpaceEncoderModel(SharedSpaceEncoderPreTrainedModel):
    """
    The **Model object:
      - Initializes:
        - The vocabulary embeddings (and optional decomposition)
        - Position embeddings (calculated in RotaryEmbedding)
        - All of the **Layer objects.
      - Provides interface to vocab embeddings.
      - Executes the whole model in `forward`.

      TODO / Note - The projection needs to be retained for creating the
      residual stream, but otherwise can be fused into the attention input
      heads after training. Or maybe the layer 0 attention heads should have
      a different input size from the start.

    """

    def __init__(self, config: SharedSpaceEncoderConfig) -> None:
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
                SharedSpaceEncoderLayer(
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

    # Comment--evaluates the model on (describe expected input shape) and returns (describe)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        # TODO - the attention mask is... "additive mask (0 for keep, −inf for pad)"
        # TODO - For coding style--don't use symbols unless it's a big, dense function.

        Inputs:
            input_ids       [batch_size, seq_len]
            attention_mask  [batch_size,    1,    1, seq_len]

        Returns:
            Final encoder layer output   [batch_size, seq_len, model_size]
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

        # For each encoder layer,
        for layer_i, layer in enumerate(self.layers):

            # Evaluate the layer
            hidden_states = layer(
                hidden_states,       # Token embeddings
                (R_cos, R_sin),      # Rope embeddings, passed as a tuple.
                attention_mask,      # Attn mask
            )

        # Return the final output of the encoder stack.
        return hidden_states

