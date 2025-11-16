"""
# `task_heads.py`

Language modeling heads and downstream adapters constructed on top of the
Sparse MoE decoder. This mirrors the structure of
`subspace_decoder/layers/task_heads.py` while wiring in the MoE-aware model
and configuration classes.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..models.shared_space_config import SparseMoEDecoderConfig
from ..models.shared_space_decoder import (
    SparseMoEDecoderPreTrainedModel,
    SparseMoEDecoder,
)
from ..utils import create_norm_layer as base_create_norm_layer


def build_norm_layer(hidden_size: int, config: SparseMoEDecoderConfig) -> nn.Module:
    """
    Create a normalization layer based on the config.norm_type.

    Args:
        hidden_size: Dimension to normalize.
        config: Decoder configuration containing norm settings.
    """
    if config.norm_type == "layernorm":
        eps = config.layer_norm_eps
    elif config.norm_type == "rmsnorm":
        eps = config.rms_norm_eps
    else:
        raise ValueError(f"Unknown norm_type: {config.norm_type}")
    return base_create_norm_layer(hidden_size, config.norm_type, eps)


class SparseMoEDecoderForCausalLM(SparseMoEDecoderPreTrainedModel):
    """
    Sparse MoE decoder model with a causal language modeling head.

    Extends :class:`SparseMoEDecoder` with:
    - Final normalization prior to logits.
    - Language modeling head that projects hidden states to vocabulary scores.
    - Compatibility helpers for HuggingFace Trainer APIs.
    """

    def __init__(self, config: SparseMoEDecoderConfig) -> None:
        super().__init__(config)

        # ===============================
        #      Base Decoder Backbone
        # ===============================
        self.model = SparseMoEDecoder(config)

        # Final norm applied before projecting to logits.
        self.norm = build_norm_layer(config.hidden_size, config)

        # Linear projection from model dimension back to the vocabulary.
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Post initialization aligns behavior with HuggingFace pretrain flows.
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """
        Customize initialization for the LM head while delegating the rest to the
        parent implementation.
        """
        super()._init_weights(module)

        if module is self.lm_head:
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.model.vocab_proj is not None:
                module.weight.data.normal_(
                    mean=0.0,
                    std=self.config.initializer_range * 0.5,
                )

    # HuggingFace compatibility helpers ---------------------------------- #
    def get_input_embeddings(self):
        return self.model.vocab_embed

    def set_input_embeddings(self, value):
        self.model.vocab_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        """
        Tie input and output embeddings when vocab embeddings live in model space.
        """
        if getattr(self.model, "vocab_proj", None) is None:
            self._tie_or_clone_weights(self.lm_head, self.model.vocab_embed)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, tuple]:
        """
        Args:
            input_ids: `[B, T]` token ids.
            attention_mask: Optional `[B, T]` attention mask.
            labels: Optional targets for cross-entropy loss.
        """
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states
            if kwargs.get("output_hidden_states", False)
            else None,
            attentions=None,
        )
