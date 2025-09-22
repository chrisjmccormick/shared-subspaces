
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from transformers.modeling_outputs import CausalLMOutputWithPast

from models.shared_space_config import SharedSpaceDecoderConfig
from models.shared_space_decoder import (
    SharedSpaceDecoderPreTrainedModel,
    SharedSpaceDecoderModel,
    DeepseekV3RMSNorm
)

def create_norm_layer(hidden_size: int, config: SharedSpaceDecoderConfig) -> nn.Module:
    """
    Create a normalization layer based on the config norm_type.
    
    Args:
        hidden_size: The dimension to normalize over
        config: Configuration containing norm_type and epsilon values
    
    Returns:
        Either a LayerNorm or RMSNorm layer
    """
    if config.norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
    elif config.norm_type == "rmsnorm":
        from models.shared_space_decoder import DeepseekV3RMSNorm
        return DeepseekV3RMSNorm(hidden_size, eps=config.rms_norm_eps)
    else:
        # This should be caught by config validation, but being defensive
        raise ValueError(f"Unknown norm_type: {config.norm_type}")


class SharedSpaceDecoderForCausalLM(SharedSpaceDecoderPreTrainedModel):
    """
    Subspace Decoder model with a causal language modeling head.
    
    This model extends the SharedSpaceDecoderModel with:
    - A language modeling head that projects hidden states to vocabulary logits
    - Support for computing cross-entropy loss for language modeling
    - Proper HuggingFace compatibility for causal language modeling tasks
    - Decoder-specific initialization strategies
    
    The model can be used for:
    - Text generation
    - Language modeling pretraining  
    - Fine-tuning on downstream tasks
    """

    def __init__(self, config: SharedSpaceDecoderConfig) -> None:
        super().__init__(config)
        
        # Initialize the base decoder model
        self.model = SharedSpaceDecoderModel(config)
        
        # Final layer norm before the language modeling head
        self.norm = create_norm_layer(config.hidden_size, config)
        
        # Language modeling head
        # Projects from hidden_size to vocab_size to get logits for each token
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False  # Following common practice in modern LMs
        )
        
        # Initialize weights with decoder-specific strategy
        # Note: tie_weights() will be called automatically by post_init() if config.tie_word_embeddings=True
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """
        Decoder-specific weight initialization with special handling for language modeling head.
        
        Key differences from encoder initialization:
        - Language modeling head gets specialized initialization for stability
        - Configurable normalization layers (LayerNorm or RMSNorm) are properly handled  
        - Weight tying considerations for embedding/lm_head relationship
        """
        
        # Use the base class initialization for most modules
        super()._init_weights(module)
        
        # Special handling for language modeling head
        if module is self.lm_head:
            # Use smaller initialization for the language modeling head
            # This helps with training stability in autoregressive generation
            # Common practice is to use std=initializer_range or smaller
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            
            # If weight tying is not used, we might want even smaller init
            if self.model.vocab_proj is not None:
                # For vocab subspace models where weights aren't tied,
                # use a smaller scale to prevent initial logits from being too large
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range * 0.5)

    def get_input_embeddings(self):
        """Return the input embedding layer for compatibility with HuggingFace."""
        return self.model.vocab_embed

    def set_input_embeddings(self, value):
        """Set the input embedding layer for compatibility with HuggingFace."""
        self.model.vocab_embed = value

    def get_output_embeddings(self):
        """Return the output embedding layer (lm_head) for compatibility."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set the output embedding layer for compatibility."""
        self.lm_head = new_embeddings

    def tie_weights(self):
        """
        Tie the input and output embedding weights.
        
        This method sets the language modeling head's weight to be the same as 
        the input embedding weight. This reduces the number of parameters and
        is a common practice in modern language models.
        
        Note: For vocab subspace models, we need to handle the case where
        input embeddings go through a projection layer.
        """
        # Only tie when embeddings live in model space (no vocab_proj)
        if getattr(self.model, "vocab_proj", None) is None:
            # Use HF utility for correct tying/cloning semantics
            self._tie_or_clone_weights(self.lm_head, self.model.vocab_embed)
        # else: leave untied for subspace case


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, tuple]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len] 
                           (1 for real tokens, 0 for padding)
            labels: Ground truth token ids for computing loss. Same shape as input_ids.
                   If provided, loss will be computed. Typically input_ids shifted by 1.
            
        Returns:
            CausalLMOutputWithPast containing:
            - logits: Prediction logits of shape [batch_size, seq_len, vocab_size]  
            - loss: Cross-entropy loss if labels provided, else None
            - hidden_states: Final layer hidden states [batch_size, seq_len, hidden_size]
        """
        
        # Run the base decoder model
        # This applies all the transformer layers with causal attention
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply final layer normalization
        # This normalizes the final hidden states before the language modeling head
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary logits
        # Shape: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        # Previously, we had custom loss computation here, but now we use the 
        # standard HuggingFace loss function.
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        
        # Return in HuggingFace format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Not implementing KV cache yet
            #hidden_states=hidden_states,
            hidden_states=hidden_states if kwargs.get("output_hidden_states", False) else None,
            attentions=None,
        )

