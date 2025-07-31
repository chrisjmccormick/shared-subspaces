
class MultiheadLatentAttention(nn.Module):
    """
    A variant of MLA with:
    - Simplified RoPE handling:
      - A portion of the head dimensions are used for position information.
      - Same number of queries as keys. (no MQA)
    - Optional output subspace
    """

    def __init__(self, config: SubspaceBertConfig, layer_idx: int):
        super().__init__()
        
        self.config = config
        
        # Used to determine if this layer is dense or uses latents.
        self.layer_idx = layer_idx 
        self.attention_dropout_prob = config.attention_dropout_prob
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim # New / Add
        
        self.rope_theta = config.rope_theta
        self.rope_dims = config.rope_dims # New / Add

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        # Explicit dimensional attributes for clarity
        self.hidden_size = config.hidden_size
        self.v_head_dim = config.v_head_dim
        self.q_lora_dim = config.q_lora_rank
        self.kv_lora_dim = config.kv_lora_rank

        #self.qk_rope_head_dim = config.qk_rope_head_dim -- Remove
        #self.v_head_dim = config.v_head_dim # Remove
        #self.qk_nope_head_dim = config.qk_nope_head_dim -- Remove
        
        # =========================
        #     Input Projections
        # =========================
        
        # If this is one of the dense layers,
        if self.layer_idx < config.num_dense_layers:
            
            # =========================
            #     Dense Attention
            # =========================

            # No latent projections.
            self.latent_spaces = False

            # Define the standard QKV projection
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                self.num_heads * (self.head_dim * 3),
                bias=config.attention_bias,
            )
        
        # If we're past the dense layers,
        else:

            # =========================
            #     Latent Attention
            # =========================

            # Use latent projections.
            self.latent_spaces = True

            # Input latent projections
            self.qkv_a_proj = nn.Linear(
                config.hidden_size,
                self.q_lora_rank + self.kv_lora_rank,
                bias=config.attention_bias,
            )

            # TODO - Decide whether to share or split.
            self.qkv_a_layernorm = DeepseekV3RMSNorm(
                self.q_lora_rank + self.kv_lora_rank,
                eps=config.rms_norm_eps,
            )

            # Query heads
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, 
                self.num_heads * self.head_dim, 
                bias=False # TODO
            )

            # Key and Value heads, concatenated
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_heads * (self.head_dim * 2),
                bias=False,
            )
            
        # ==========================
        #     Output Projections
        # ==========================

        self.output_subspace = config.output_subspace

        if self.output_subspace:
            
            # ==========================
            #     Output Subspace
            # ==========================

            self.o_lora_rank = config.o_lora_rank 

            # Per-head output projections
            # (Similar to original W^O, but projects the scored value vectors
            #  into a latent space instead of back to the model)
            self.o_a_proj = nn.Linear(
                self.num_heads * self.v_head_dim,
                self.o_lora_rank, 
                bias=False
            )

            # Regarding bias terms:
            #   - The thought here is to mirror the behavior on the input 
            #     latents, where only one of the two projections receives a
            #     bias term. 
            #     - Haven't yet experimented with this (i.e., whether to place
            #       it on a or b or neither)
            #
            # Regarding Layernorm:
            #   - In the ViT experiments, the addition of a layernorm between 
            #     the o_a and o_b projections (i.e., applying it to the output
            #     of o_a) hurt performance.
            #   - I have not tried applying it to the output of o_b.
            # 
            #self.o_a_layernorm = DeepseekV3RMSNorm(
            #    self.o_lora_rank, 
            #    eps=config.rms_norm_eps
            #)

            # Shared output projection
            # The head outputs from `o_a_proj` are first summed together (across
            # heads) in the latent space.
            # Then we project their combined outputs (a single vector per token)
            # back to model space via `o_b_proj`.
            self.o_b_proj = nn.Linear(
                self.o_lora_rank, 
                self.hidden_size, 
                bias=config.attention_bias
            )
   
        # Original output matrix
        else:
            # ========================
            #     Dense Output 
            # ========================

            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                config.hidden_size,
                bias=config.attention_bias,
            )

        
        #self.qk_head_dim = config.qk_nope_head_dim  -- Remove    
        #self.q_combined_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim -- Remove    
        # Define separate variables just for clarity in code on which one we're
        # actually dealing with, and to avoid the impression that it's the 
        # concatenation of the two.
        #self.q_head_dim = self.qk_head_dim - Remove
        #self.k_head_dim = self.qk_head_dim - Remove
        #self.v_head_dim -- Remove
        
        # This is not a decoder model
        self.is_causal = False # TODO - Is this needed by huggingface?
        
        # Softmax scaling factor.
        self.scaling = self.head_dim ** (-0.5)

        # TODO...
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None, # TODO - Can I remove this?
        cache_position: Optional[torch.LongTensor] = None, # TODO - Can I remove this?
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # === Tensor Dimension Symbols ===
        # B: batch_size     — number of samples in the batch
        # T: seq_len        — number of tokens per sample
        # H: n_heads        — number of attention heads
        # D: hidden_dim     — model embedding size
        # Dh: head_dim      — per-head projection dimension
        # Dc: compress_dim  — dimension of latent (e.g., LoRA or MLA) subspace
        # R: rope_dim       — rotary positional embedding size
        # C: cache_dim      — compressed key/value dimension

        B, T = hidden_states.shape[:2]
        

        # ==============================
        #     QKV Head Projections
        # ==============================

        # If this layer uses latent projections,
        if self.latent_spaces:

            # Project token embeddings into shared latents
            # Input:  hidden_states [B, T, D]
            # Output: input_latents [B, T, Dc_q + Dc_kv]
            input_latents = self.qkv_a_proj(hidden_states)

            # Normalize latent vectors
            # Input:  input_latents [B, T, Dc_q + Dc_kv]
            # Output: input_latents [B, T, Dc_q + Dc_kv]
            input_latents = self.qkv_a_layernorm(input_latents)

            # Split latents for queries and keys/values
            # Input:  input_latents [B, T, Dc_q + Dc_kv]
            # Outputs:
            #   q_latents  [B, T, Dc_q]
            #   kv_latents [B, T, Dc_kv]
            q_latents, kv_latents = torch.split(
                input_latents, [self.q_lora_dim, self.kv_lora_dim], dim=-1
            )

            # Linear projection of query latents
            # Input:  q_latents [B, T, Dc_q]
            # Output: queries [B, T, H * Dh]
            queries = self.q_b_proj(q_latents)

            # Linear projection of key/value latents
            # Input:  kv_latents [B, T, Dc_kv]
            # Output: keysvalues [B, T, H * 2 * Dh]
            keysvalues = self.kv_b_proj(kv_latents)

            # Split into key and value tensors
            # Each: [B, T, H * Dh]
            keys, values = keysvalues.chunk(2, dim=-1)
            # TODO - Can einsum project and split?

        # If this is a dense attention layer (no latent projections),
        else:
            # Standard QKV projection
            # Input:  hidden_states [B, T, D]
            # Output: querieskeysvalues [B, T, H * 3 * Dh]
            querieskeysvalues = self.qkv_proj(hidden_states)

            # Separate query, key, and value vectors
            # Each: [B, T, H * Dh]
            queries, keys, values = querieskeysvalues.chunk(3, dim=-1)
        
        # ==================
        #        RoPE
        # ==================

        # Apply rotary position embeddings to a portion of Q/K
        # queries, keys: [B, T, H * Dh]
        # TODO: implement RoPE across the final `self.rope_dims` dims

        # ===================
        #      Attention
        # ===================

        # Reshape Q, K, V for attention computation
        # Inputs:
        #   queries [B, T, H * Dh]
        #   keys    [B, T, H * Dh]
        #   values  [B, T, H * Dh]
        # TODO: split head dimension and call backend

        # Invoke

        # TODO...

        # Reshape outputs if needed (TODO)

        # =========================
        #     Output Projection
        # =========================

        # If we are using an output latent projection,
        if self.output_subspace:

            # First, project the scored value vectors onto `o_a_proj`. This is
            # equivalent to projecting onto W^O in standard attention, except 
            # that here we are projecting into an intermediate latent space. 
            # This projection is unique per-head, preserving head diversity, and
            # then sums the results into a single vector per token.
            attn_output = self.o_a_proj(attn_output)

            # MLA uses RMSNorm on the query and key-value latents. It's not
            # clear yet whether this is helpful for the output.
            #attn_output = self.o_a_layernorm(attn_output)

            #print(f"attn_output after o_a_proj: {attn_output.shape}")

            # The input to `o_b_proj` is the summed output latents of the 
            # attention heads. This step re-projects this single per-token 
            # latent back to model space.
            attn_output = self.o_b_proj(attn_output)

        # If this is a dense layer,
        else:
            # Project the values back into model space.
            attn_output = self.o_proj(attn_output)

        # -----------------------------------------

        return attn_output  #, attn_weights - TODO - does transformers require these?
