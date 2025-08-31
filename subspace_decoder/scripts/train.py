# -*- coding: utf-8 -*-
# Updated training script for DeepSeek V3 with attention output subspace

"""# subspace_decoder/scripts/train.py"""

print("Importing Packages...\n")

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from utils import summarize_parameters, format_size
# To disable a warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make sure we can import modules from the decoder package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT", PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn
from transformers import DeepseekV3ForCausalLM

def check_bf16_support():
    """Check if BFloat16 is supported on the current hardware and PyTorch version."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. BFloat16 training requires CUDA.")
        return False
    
    # Check if the GPU supports BFloat16
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        print("✓ BFloat16 is supported on this hardware")
        return True
    
    # Fallback check for older PyTorch versions
    try:
        # Try to create a small BFloat16 tensor on GPU
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
        print("✓ BFloat16 is supported on this hardware")
        return True
    except Exception as e:
        print(f"Warning: BFloat16 not supported on this hardware: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

def patch_attention_with_output_latent(model, o_latent_dim: int):
    """
    Replaces each attention layer's o_proj with a two-layer O-latent:
      [H*v_head_dim] -> [o_latent_dim] -> [hidden_size]
    
    """
    
    # DeepSeek-V3 in HF is usually at model.model.layers[*].self_attn
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        in_features = attn.num_heads * attn.v_head_dim
        out_features = model.config.hidden_size  # Get from model config instead
        bias = getattr(model.config, "attention_bias", False)  # Also get bias from model config

        attn.o_proj = nn.Sequential(
            nn.Linear(in_features, o_latent_dim, bias=False),   # O_a (no bias like your code)
            nn.Linear(o_latent_dim, out_features, bias=bias),   # O_b (respect config.attention_bias)
        )

def create_model_with_mla_o(config_dict):
    """
    Create a DeepSeek V3 model with output subspace using the simple patching approach.
    
    Args:
        config_dict: Dictionary with model configuration parameters
    """
    from transformers import DeepseekV3Config
    
    # Create a standard DeepSeek V3 config, excluding our custom parameters
    standard_config_dict = {k: v for k, v in config_dict.items() 
                           if k not in ['use_output_subspace', 'o_latent_dim']}
    
    # Create the config object
    config = DeepseekV3Config(**standard_config_dict)
    
    # Create the model from scratch with our custom config
    model = DeepseekV3ForCausalLM(config)
    
    # Apply output subspace patching if requested
    if config_dict.get("use_output_subspace", False) and config_dict.get("o_latent_dim") is not None:
        print(f"Patching attention layers with output latent dimension: {config_dict['o_latent_dim']}")
        originals = patch_attention_with_output_latent(model, config_dict["o_latent_dim"])
        # Store the originals in case we need to restore later
        model._original_o_projs = originals
    
    return model

def main(config_path: str):
    """Run pre-training using the provided configuration path."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        full_cfg = json.load(f)

    model_cfg = full_cfg['model']
    ptrain_cfg = full_cfg['pre_train']

    # Print out its shorthand name.
    print(full_cfg["shorthand"])

    # Initialize the optional stats dictionary so later assignments don't fail.
    if "stats" not in full_cfg:
        full_cfg["stats"] = {}
    
    # Validate mixed precision settings
    if ptrain_cfg.get("bf16", False) and ptrain_cfg.get("fp16", False):
        raise ValueError("Cannot enable both bf16 and fp16 simultaneously. Please choose one.")
    
    # Check BFloat16 compatibility if enabled
    if ptrain_cfg.get("bf16", False):
        if not check_bf16_support():
            print("BFloat16 requested but not supported. Falling back to FP16.")
            ptrain_cfg["bf16"] = False
            ptrain_cfg["fp16"] = True
    
    # Display torch.compile status
    if ptrain_cfg.get("torch_compile", False):
        print(f"✓ torch.compile enabled:")
        print(f"  Backend: {ptrain_cfg.get('torch_compile_backend', 'inductor')}")
        print(f"  Mode: {ptrain_cfg.get('torch_compile_mode', 'default')}")
        print("  Note: First training step will be slower due to compilation.")
    else:
        print("torch.compile disabled. Enable with 'torch_compile': true in config.")

    # Use the DeepSeek tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify vocab size matches
    assert model_cfg["vocab_size"] == tokenizer.vocab_size

    # Set random seed for reproducibility
    set_seed(ptrain_cfg["seed"])

    # Setup Weights & Biases
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    # ======================
    #    Prepare Dataset
    # ======================

    dataset = load_dataset(
        ptrain_cfg["dataset_name"],
        ptrain_cfg["dataset_config"]
    )

    # Print the dataset object to see sample counts.
    print(dataset)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=ptrain_cfg["max_seq_length"],
            padding="max_length"
        )

    # Tokenize the full dataset
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8, # Use more CPUs to speed it up--this helps a lot.
        remove_columns=["text"] # Comment this
    )

    # Use DataCollatorForLanguageModeling with mlm=False for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Disable masking for causal LM
    )

    # ========================
    #    Initialize Model
    # ========================

    print("Creating MLA-o model...")
    model = create_model_with_mla_o(model_cfg)

    # ================================
    #       Review Configuration
    # ================================

    # Display architecture
    print(model)

    print("\n======== Model ========")
    print(json.dumps(model_cfg, indent=2))

    print("\n======== Pre-Train ========")
    print(json.dumps(ptrain_cfg, indent=2))

    print("=============================\n")

    """## Parameter Summary"""

    print("\n======== Parameters ========")

    ## Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    full_cfg["stats"]["total_elements"] = format_size(total_params)

    print(f"Total elements: {full_cfg['stats']['total_elements']}\n")

    # Display a full parameter breakdown using the shared utility
    summarize_parameters(model)

    # ========================================
    #   Format Settings for WandB Run Name
    # ========================================

    # Format the cfg learning rate as a scientific notation string like 5e-4
    lr_str = '{:.0e}'.format(ptrain_cfg['learning_rate'])

    # Attention configuration

    ptrain_cfg["run_name"] = full_cfg["stats"]["total_elements"] + " - " + full_cfg["shorthand"]

    print(ptrain_cfg["run_name"])

    """## wandb and TrainingArguments"""

    wandb.init(
        project="decoder-pretrain-wiki103",
        name=ptrain_cfg["run_name"],
        config=full_cfg
    )

    # ===============================
    #       Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=ptrain_cfg["output_dir"],

        per_device_train_batch_size=ptrain_cfg["train_batch_size"],
        per_device_eval_batch_size=ptrain_cfg["eval_batch_size"],

        bf16=ptrain_cfg.get("bf16", False),
        fp16=ptrain_cfg.get("fp16", False),
        
        # torch.compile configuration for performance optimization
        torch_compile=ptrain_cfg.get("torch_compile", False),
        torch_compile_backend=ptrain_cfg.get("torch_compile_backend", "inductor"),
        torch_compile_mode=ptrain_cfg.get("torch_compile_mode", "default"),

        learning_rate=ptrain_cfg["learning_rate"],
        max_steps=ptrain_cfg["num_train_steps"], 

        # The dataloader is a bottleneck without these.
        dataloader_num_workers=ptrain_cfg.get("num_workers", 8),
        dataloader_pin_memory=ptrain_cfg.get("pin_memory", True),
        # The prefetch factor didn't appear to help.
        #dataloader_prefetch_factor = ptrain_cfg.get("prefetch_factor", 2),

        weight_decay=ptrain_cfg.get("weight_decay", 0.01),  

        # Learning rate warmup (10% of total steps)
        warmup_steps=int(0.1 * ptrain_cfg["num_train_steps"]),  
        lr_scheduler_type="linear",  # Linear warmup then decay

        # Evaluate every 2,000 steps
        # Note: Recent versions of Trainer changed the name from 
        # `evaluation_strategy` to `eval_strategy`.
        batch_eval_metrics = True, # To avoid OOM
        eval_strategy="steps",
        eval_steps=ptrain_cfg.get("eval_steps", 2000),
        eval_accumulation_steps=4,  # Process eval in smaller chunks to save memory

        logging_steps=50,
        metric_for_best_model="eval_loss",
        save_steps=2000,
        save_total_limit=2,           # Optional: keeps last 2 checkpoints
        save_strategy="steps",
        report_to=["wandb"],
        
        run_name=ptrain_cfg["run_name"],
        
        remove_unused_columns=False,  # Optional: avoid dropping custom model inputs
    )

    print(training_args)

    import numpy as np

    class PerplexityMetric:
        """
        A stateful class to compute perplexity in a batch-wise manner to avoid OOM.
        Similar to the MLMAccuracyMetric from the encoder training.
        """
        def __init__(self):
            # Initialize state variables to store running totals
            self.total_loss = 0.0
            self.total_tokens = 0

        def __call__(self, eval_pred, compute_result=False):
            """
            This method will be called by the Trainer.
            """
            predictions, labels = eval_pred

            # For causal LM, we compute perplexity
            # Shift predictions and labels for next token prediction
            shift_logits = predictions[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Create a mask for valid tokens (not padding, typically -100)
            mask = shift_labels != -100
            
            if mask.sum() > 0:  # Only compute if there are valid tokens
                # Compute loss only on valid tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                batch_loss = loss_fct(shift_logits[mask], shift_labels[mask])
                
                # Add to running totals
                self.total_loss += batch_loss.item()
                self.total_tokens += mask.sum().item()

            # If this is the final call after all batches are processed
            if compute_result:
                # Avoid division by zero
                if self.total_tokens == 0:
                    avg_loss = 0.0
                    perplexity = float('inf')
                else:
                    avg_loss = self.total_loss / self.total_tokens
                    perplexity = np.exp(avg_loss)

                # Prepare the final metrics dictionary
                metrics = {
                    "perplexity": perplexity,
                    "loss": avg_loss,
                }

                # Reset state for the next evaluation run
                self.total_loss = 0.0
                self.total_tokens = 0

                return metrics

            # For intermediate calls, return an empty dict
            return {}

    # Instantiate your stateful metric computer
    perplexity_metric = PerplexityMetric()

    # ===============================
    #           Trainer
    # ===============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=perplexity_metric,

        # New argument, allows for other modalities.
        processing_class=tokenizer,

        data_collator=data_collator,
    )

    """## Loop"""

    # =====================
    #     Run Training
    # =====================

    # Do inside a try/finally so that if the run aborts, we still call wandb.finish().
    try:
        trainer.train()

        metrics = trainer.evaluate()

        wandb.log(metrics)

        # Store wandb ids into the config.
        full_cfg["pre_train"]["run_id"] = wandb.run.id
        full_cfg["pre_train"]["run_url"] = wandb.run.url
        full_cfg["pre_train"]["run_name"] = wandb.run.name

        # Save the best checkpoint.
        full_cfg["pre_train"]["best_checkpoint"] = trainer.state.best_model_checkpoint

        # Save the json back to disk
        with open(ptrain_cfg["output_dir"] + "/full_config.json", "w") as f:
            json.dump(full_cfg, f, indent=2)
   

    finally:
        # End the wandb run.
        wandb.finish()

    
if __name__ == "__main__":
    args = parse_args()
    main(args.config)
