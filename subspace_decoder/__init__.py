# -*- coding: utf-8 -*-

"""
Subspace Decoder Package

A Transformer decoder implementation with Multi-Head Latent Attention (MLA)
and decomposed MLP layers for efficient parameter usage.
"""

# Import all the main classes from models
from .models import (
    SharedSpaceDecoderConfig,
    SharedSpaceDecoderPreTrainedModel,
    SharedSpaceDecoderModel, 
    SharedSpaceDecoderForCausalLM,
)

__version__ = "0.1.0"

__all__ = [
    "SharedSpaceDecoderConfig",
    "SharedSpaceDecoderPreTrainedModel",
    "SharedSpaceDecoderModel",
    "SharedSpaceDecoderForCausalLM",
]
