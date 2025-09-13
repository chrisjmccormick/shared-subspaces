
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
        
        # Tie input and output embeddings
        # This shares the weight matrix between input embeddings and the language modeling head
        # The lm_head uses the transpose of the input embedding matrix
        self.tie_weights()
        
        # Initialize weights with decoder-specific strategy
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
        if self.model.vocab_proj is not None:
            # For decomposed vocabulary embeddings, we cannot directly tie weights
            # because the embedding path is: token_id -> vocab_embed(vocab_rank) -> vocab_proj(hidden_size)
            # and the lm_head path is: hidden_states(hidden_size) -> lm_head -> logits(vocab_size)
            # 
            # To properly tie weights, we'd need to restructure the computation.
            # For now, we'll leave them untied in the subspace case.
            # TODO: Implement proper weight tying for vocab subspace models
            pass
        else:
            # For dense vocabulary embeddings, directly tie the weights
            # lm_head.weight should be the transpose of vocab_embed.weight
            self.lm_head.weight = self.model.vocab_embed.weight

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
        loss = None
        if labels is not None:
            # Shift logits and labels for causal language modeling
            # We predict the next token, so compare logits[..., :-1, :] with labels[..., 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy computation
            # cross_entropy expects [N, C] logits and [N] targets
            batch_size, seq_len = shift_logits.shape[:2]
            vocab_size = shift_logits.shape[-1]
            
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            
            # Compute cross entropy loss
            # ignore_index=-100 ignores padding tokens in loss computation
            loss = F.cross_entropy(
                flat_logits, 
                flat_labels, 
                ignore_index=-100
            )
        
        # Return in HuggingFace format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Not implementing KV cache yet
            hidden_states=hidden_states,
            attentions=None,
        )

