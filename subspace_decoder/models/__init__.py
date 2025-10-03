# -*- coding: utf-8 -*-

"""
Shared Subspace Decoder Models

This module contains the implementation of the Shared Subspace Decoder architecture,
including Multi-Head Latent Attention (MLA) and decomposed MLP layers.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .shared_space_config import SharedSpaceDecoderConfig
from .shared_space_decoder import (
    SharedSpaceDecoderPreTrainedModel,
    SharedSpaceDecoderModel,
)

# Import from task_heads in layers directory  
from ..layers.task_heads import SharedSpaceDecoderForCausalLM

# Register the configuration class with AutoConfig
AutoConfig.register("shared_space_decoder", SharedSpaceDecoderConfig)

# Register the model classes with AutoModel
AutoModel.register(SharedSpaceDecoderConfig, SharedSpaceDecoderModel)
AutoModelForCausalLM.register(SharedSpaceDecoderConfig, SharedSpaceDecoderForCausalLM)

__all__ = [
    "SharedSpaceDecoderConfig",
    "SharedSpaceDecoderPreTrainedModel", 
    "SharedSpaceDecoderModel",
    "SharedSpaceDecoderForCausalLM",
]
