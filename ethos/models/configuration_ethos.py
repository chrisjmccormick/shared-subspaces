"""
ETHOS Configuration

Configuration class for the ETHOS model.

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

from transformers import PretrainedConfig


class EthosConfig(PretrainedConfig):
    """
    Configuration class for ETHOS model.
    
    This is the configuration class to store the configuration of an ETHOS model.
    It is used to instantiate an ETHOS model according to the specified arguments,
    defining the model architecture.
    """
    
    model_type = "ethos"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_dense_layers=2,
        num_moe_layers=14,
        q_lora_rank=768,
        kv_lora_rank=256,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        num_experts=512**2,
        d_latent=128,
        d_intermediate_hypernet=512,
        top_k=16,
        num_routing_heads=8,
        d_query=512,
        intermediate_size=4096,
        max_seq_len=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_triton=False,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        **kwargs,
    ):
        # Model architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_dense_layers = num_dense_layers
        self.num_moe_layers = num_moe_layers
        
        # Attention parameters  
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        
        # MoE parameters
        self.num_experts = num_experts
        self.d_latent = d_latent
        self.d_intermediate_hypernet = d_intermediate_hypernet
        self.top_k = top_k
        self.num_routing_heads = num_routing_heads
        self.d_query = d_query
        
        # FFN parameters
        self.intermediate_size = intermediate_size
        
        # Position embedding parameters
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        
        # Normalization
        self.rms_norm_eps = rms_norm_eps
        
        # Implementation options
        self.use_triton = use_triton
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        