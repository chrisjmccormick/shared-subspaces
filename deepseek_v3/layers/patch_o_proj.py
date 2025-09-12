# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple, Optional, Dict
from pathlib import Path
import json
from safetensors.torch import load_file as safe_load_file
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM 

# ============================================================================
# IMPORTANT: STYLE GUIDE
# This is a **prototype**. Don't polish it prematurely.
# 
# This means:
# - Prioritize legibility over re-use and robustness. 
#     - DO NOT use get_attr. All settings are required. If something is missing,
#       that was a programmer error, and we want the code to **crash**. 
#     - Do not clutter the code with "raise ValueError", that's premature.
#     - We can add those things in later when the code is mature. 
# 
# Minimize boiler plate and safety checks.
# Don't create silent bugs by implementing fallbacks.
# Comment heavily. 
# Avoid packing operations into dense, single lines.
# Prefer flat code to small helper functions. Factorization is for mature code,
# not prototypes. It requires the developer to invest time and energy into 
# becoming familiar with what the function does and how to use it.
#
# Note that some of the existing code is agent-written and doesn't currently
# follow these guidelines.
# ============================================================================

Variant = Literal["vanilla", "sequential", "sequential_norm", "woabnorm", "woabnorm_shared"]

# ------------------------------
#   Blocks for O-proj variants
# ------------------------------

class HeadwiseRMSNorm(nn.Module):
    """
    Head-wise RMSNorm over the last dim (Do), with per-head learnable scale.
    Expects input [B, S, H, Do].
    """
    def __init__(self, num_heads: int, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        # Per-head, per-dim gain
        self.weight = nn.Parameter(torch.ones(num_heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, Do]
        # rms over last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()

        x = x * rms  # normalize

        # Apply per-head gains: broadcast [H, Do] -> [B, S, H, Do]
        return x * self.weight.view(1, 1, *self.weight.shape)

class SharedRMSNorm(nn.Module):
    """
    Head-wise RMSNorm over the last dim (Do), but with a single
    learnable gain vector shared across all heads.
    
    Expects input [B, S, H, Do].
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # One gain vector, shared by all heads
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, Do]
        # RMS across the latent dim Do
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms  # normalize
        
        # Apply the shared gain [Do] → broadcast to [B, S, H, Do]
        return x * self.weight.view(1, 1, 1, -1)


class WOABNormProj(nn.Module):
    """
    Private-shared decomposition:
      values [B,S,H*Dv] -> W_oa (no bias) -> latents [B,S,H,Do]
      -> Headwise RMSNorm -> sum over heads -> W_ob (bias per config) -> [B,S,D]
    """
    def __init__(
        self,
        num_heads: int,
        v_head_dim: int,
        out_features: int,
        o_latent_dim: int,
        eps: float = 1e-6,
        bias: bool = False,
        shared_gain: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.o_latent_dim = o_latent_dim
        self.out_features = out_features

        # Single matmul into shared latent, no bias by design
        self.W_oa = nn.Linear(num_heads * v_head_dim, o_latent_dim, bias=False)
        
        if shared_gain:
            self.head_norm = SharedRMSNorm(o_latent_dim, eps)
        else:
            self.head_norm = HeadwiseRMSNorm(num_heads, o_latent_dim, eps)
        
        self.W_ob = nn.Linear(o_latent_dim, out_features, bias=bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, S, H*Dv]
        B, S, _ = values.shape

        # Compute head-wise latents by reshaping weight to per-head slices
        # Do it via manual matmul to keep head dimension explicit for head_norm
        # W_oa.weight: [Do, H*Dv] -> reshape to [Do, H, Dv] -> [H, Dv, Do]
        W = self.W_oa.weight.view(self.o_latent_dim, self.num_heads, self.v_head_dim).permute(1, 2, 0)
        # values -> [B,S,H,Dv]
        v = values.view(B, S, self.num_heads, self.v_head_dim)
        # einsum: [B,S,H,Dv] x [H,Dv,Do] -> [B,S,H,Do]
        o_latents = torch.einsum("bshd,hdo->bsho", v, W)

        # head-wise RMSNorm, then sum heads
        o_latents = self.head_norm(o_latents)           # [B,S,H,Do]
        o_latents = o_latents.sum(dim=-2)  / (self.num_heads ** 0.5)   # [B,S,Do]

        return self.W_ob(o_latents)                     # [B,S,D]


def make_sequential_o_proj(
    in_features: int, 
    o_latent_dim: int, 
    out_features: int, 
    bias: bool,
    add_norm: bool,
    eps: float,
) -> nn.Sequential:
    """
    Simple 2-layer decomposition, with or without RMS Norm applied to 
    sum of head outputs.
    """
    if add_norm:
        seq_o_proj = nn.Sequential(
            nn.Linear(in_features,  o_latent_dim,  bias=False),  # W^OA
            nn.RMSNorm(o_latent_dim, eps), 
            nn.Linear(o_latent_dim, out_features, bias=bias),    # W^OB
        )
    else:
        seq_o_proj = nn.Sequential(
            nn.Linear(in_features,  o_latent_dim,  bias=False),  # W^OA
            nn.Linear(o_latent_dim, out_features, bias=bias),    # W^OB
        )

    return seq_o_proj

# ==================================
#       Apply Structural Patch
# ==================================

def patch_o_proj_implementation(
    model,
    variant: Variant,
    o_latent_dim: int = None,
    eps: float = 1e-6,
):
    """
    Replace each layer.self_attn.o_proj to match the variant.
    No weight loading is done here—only the structure change.
    """

    if variant == "vanilla":
        print("Warning: patch_model_o_proj called with 'vanilla'") 
        return
    
    hidden_size = model.config.hidden_size
    bias = bool(getattr(model.config, "attention_bias", False))

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        in_features = attn.num_heads * attn.v_head_dim

        if variant == "sequential":
            attn.o_proj = make_sequential_o_proj(
                in_features, 
                o_latent_dim, 
                hidden_size, 
                bias, 
                add_norm = False,
                eps=eps,
            )

        elif variant == "sequential_norm":
            attn.o_proj = make_sequential_o_proj(
                in_features, 
                o_latent_dim, 
                hidden_size, 
                bias, 
                add_norm = True,
                eps=eps,
            )           
        
        elif variant == "woabnorm":
            attn.o_proj = WOABNormProj(
                num_heads=attn.num_heads,
                v_head_dim=attn.v_head_dim,
                out_features=hidden_size,
                o_latent_dim=o_latent_dim,
                eps=eps,
                bias=bias,
                shared_gain=False
            )
        elif variant == "woabnorm_shared":
            attn.o_proj = WOABNormProj(
                num_heads=attn.num_heads,
                v_head_dim=attn.v_head_dim,
                out_features=hidden_size,
                o_latent_dim=o_latent_dim,
                eps=eps,
                bias=bias,
                shared_gain=True
            )            

        else:
            raise ValueError(f"Unknown variant: {variant}")
        

# =================================================
#      Load Pre-Trained Model & Patch
# =================================================

def load_and_patch_model(
    model_cfg: dict, # Dictionary containing model definition, including output latent config
    ckpt_dict: dict, 
) -> DeepseekV3ForCausalLM:
    """
    """

    # Step 1: Load the base model without modifications.
    
    # Strip patch-only keys from HF config
    standard_config_dict = {
        k: v for k, v in model_cfg.items()
        if k not in ["use_output_subspace", "o_latent_dim", "o_proj_variant"]
    }

    # Load the model using the standard config class (but with our model parameters)
    config = DeepseekV3Config(**standard_config_dict)
    model = DeepseekV3ForCausalLM(config)

    variant = model_cfg['o_proj_variant']
    
    #  If it doesn't require patching, just load in the weights and return
    if variant == "vanilla":
        missing, unexpected = model.load_state_dict(ckpt_dict, strict=False)
        print("[vanilla] Missing:", missing)
        print("[vanilla] Unexpected:", unexpected)
        return model
    
    # ========================
    #    Patch Structure
    # ========================
    patch_o_proj_implementation(
        model, 
        variant = variant, 
        o_latent_dim = model_cfg['o_latent_dim'],
        eps = model_cfg['rms_norm_eps']
    )

    # ========================
    #    Patch Weights
    # ========================
    
    # Get all keys from the base model and from the checkpoint.
    base_keys = set(model.state_dict().keys())
    ckpt_keys = set(ckpt_dict.keys())

    # Get the keys that are unique to the checkpoint (i.e., the ones that need to
    # be reloaded).
    extras = sorted(k for k in ckpt_keys if k not in base_keys)

    print(extras)    
    
    missing, unexpected = model.load_state_dict(
        ckpt_dict, 
        strict=False 
    )

    print("Missing:", missing)
    print("Unexpected:", unexpected)

    return model
    
def load_checkpoint_state_dict(model_dir: str):
    d = Path(model_dir)

    # safetensors (single file)
    f = d / "model.safetensors"
    if f.exists():
        return safe_load_file(str(f), device="cpu")

    # safetensors (sharded)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        index = json.loads(idx.read_text())
        state = {}
        for shard in index["weight_map"].values():
            state.update(safe_load_file(str(d / shard), device="cpu"))
        return state

    # torch (single file)
    f = d / "pytorch_model.bin"
    if f.exists():
        return torch.load(str(f), map_location="cpu")

    # torch (sharded)
    idx = d / "pytorch_model.bin.index.json"
    if idx.exists():
        index = json.loads(idx.read_text())
        state = {}
        for shard in set(index["weight_map"].values()):
            state.update(torch.load(str(d / shard), map_location="cpu"))
        return state

    # If we haven't returned by now,
    raise FileNotFoundError(f"No known model weights found in {model_dir}")

class WOABNormProj_Mine(nn.Module):
    """
    Implements the private-shared (W^OA, W^OB) decomposition of the output heads,
    and applies RMSNorm to each head's output latent prior to the 
    final projection back to model space.

    DeepSeekV3's attention block applies RMSNorm:
      - At the input to the attention block
      - After the query latent projection and after the kv latent projection.
    For the output, we are doing this process in reverse.
    There are various ways we could try applying RMSNorm.
    I'm going to try applying RMSNorm to the individual output heads.

    """
    
    def __init__(self, num_heads, v_head_dim, out_features, o_latent_dim, eps, bias=False):
        super().__init__()
        self.num_heads = num_heads
        
        # Per-head projection into latent space.
        # No bias for latent spaces.
        self.W_oa = nn.Linear(num_heads * v_head_dim, o_latent_dim, bias=False)

        # Head-wise RMSNorm over the latent dim
        self.head_norm = HeadwiseRMSNorm(num_heads, o_latent_dim, eps)
        
        """
        # Create an RMSNorm for each head.
        self.norms = []
        
        # For each head,
        for head_i in range(num_heads):
        
            # Add an RMSNorm which can be applied to a head's output latent.
            self.norms.append(DeepSeekV3RMSNorm(o_latent_dim))
        """
        
        # Final shared output projection back to model space.
        # Use a bias if that's how the base model is configured.
        self.W_ob = nn.Linear(o_latent_dim, out_features, bias=bias)
        
        # Store the sizes.
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.o_latent_dim = o_latent_dim
        self.out_features = out_features
            
        
    def forward(values):
        """
        Implements the private-shared decomposition of the output heads,
        and applies RMSNorm to each head's output latent prior to the 
        final projection back to model space.
        
        The inputs are the concatenated value vectors for each token.
        """
    
        # Shapes:
        #   - values   (batch_size, seq_len, num_heads * v_head_dim)
        batch_size, seq_len, _ = values.shape
       
        # Break apart the value vectors and the output latent heads.
        #    - values   (batch_size, seq_len, num_heads, v_head_dim)  
        values = values.view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        
        # Linear stores the matrix with transposed shape.
        # Shapes:
        #   W_oah   (o_latent_dim, num_heads * v_head_dim)
        W_oah = self.W_oa.weight
        
        #   W_oah   (o_latent_dim, num_heads, v_head_dim)
        W_oah = W_oah.view(self.o_latent_dim, self.num_heads, self.v_head_dim)
        
        #   W_oah   (num_heads, v_head_dim, o_latent_dim)
        W_oah = W_oah.permute(1, 2, 0)
        
        # Add a dimension to align the vector-matrix multiplication
        # Output shape:
        #    values   (batch_size, seq_len, num_heads, 1, v_head_dim)
        values = values.unsqueeze(-2)
        
        # Project each head's value vector onto its output head.
        # Shapes:
        #    values   (batch_size, seq_len, num_heads,          1, v_head_dim)  
        #    W_oah                         (num_heads, v_head_dim, o_latent_dim)      
        #
        #  o_latents  (batch_size, seq_len, num_heads, 1, o_latent_dim)
        o_latents = values @ W_oah
        
        # Drop the empty dimension
        # Output shape:
        #  o_latents   (batch_size, seq_len, num_heads, o_latent_dim)
        o_latents = o_latents.squeeze(-2)
        
        # Head-wise RMSNorm on latents
        o_latents = self.head_norm(o_latents)  # [B,S,H,Do]
        
        """
        # Apply the RMSNorms to their heads.
        # For each head,
        for head_i in range(self.num_heads):
            # Get the head's latent.
            o_latent = o_latents[:, :, head_i, :]
            
            # Apply the norm
            o_latent = self.norms[head_i](o_latent)
            
            # Write it back
            o_latents[:, :, head_i, :] = o_latent
        """
        
        # Sum the output latents together
        # dim = -2 means collapse the head dimension.
        # Output shape:
        #    o_latents   (batch_size, seq_len, o_latent_dim)
        o_latents = o_latents.sum(dim=-2)
        
        # Perform the final output projection.
        # Inputs:
        #   o_latents   (batch_size, seq_len, o_latent_dim)
        #   W_ob        (o_latent_dim, out_features)
        output = self.W_ob(o_latents) 
        
        return output
