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

from layers.deepseek_mla_o import DeepseekV3Attention
from models.configuration_deepseek import DeepseekV3Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

def create_modified_config(base_config_dict):
    """
    Create a config from the base config dictionary.
    
    Args:
        base_config_dict: Dictionary with the DeepSeek V3 config parameters including output subspace parameters
    """
    # Create a new config with all the original parameters plus the new ones
    modified_config = DeepseekV3Config(
        vocab_size=base_config_dict.get("vocab_size", 129280),
        hidden_size=base_config_dict.get("hidden_size", 7168),
        intermediate_size=base_config_dict.get("intermediate_size", 18432),
        moe_intermediate_size=base_config_dict.get("moe_intermediate_size", 2048),
        num_hidden_layers=base_config_dict.get("num_hidden_layers", 61),
        num_nextn_predict_layers=base_config_dict.get("num_nextn_predict_layers", 1),
        num_attention_heads=base_config_dict.get("num_attention_heads", 128),
        num_key_value_heads=base_config_dict.get("num_key_value_heads", 128),
        n_shared_experts=base_config_dict.get("n_shared_experts", 1),
        n_routed_experts=base_config_dict.get("n_routed_experts", 256),
        ep_size=base_config_dict.get("ep_size", 1),
        routed_scaling_factor=base_config_dict.get("routed_scaling_factor", 2.5),
        kv_lora_rank=base_config_dict.get("kv_lora_rank", 512),
        q_lora_rank=base_config_dict.get("q_lora_rank", 1536),
        qk_rope_head_dim=base_config_dict.get("qk_rope_head_dim", 64),
        v_head_dim=base_config_dict.get("v_head_dim", 128),
        qk_nope_head_dim=base_config_dict.get("qk_nope_head_dim", 128),
        topk_method=base_config_dict.get("topk_method", "noaux_tc"),
        n_group=base_config_dict.get("n_group", 8),
        topk_group=base_config_dict.get("topk_group", 4),
        num_experts_per_tok=base_config_dict.get("num_experts_per_tok", 8),
        moe_layer_freq=base_config_dict.get("moe_layer_freq", 1),
        first_k_dense_replace=base_config_dict.get("first_k_dense_replace", 3),
        norm_topk_prob=base_config_dict.get("norm_topk_prob", True),
        scoring_func=base_config_dict.get("scoring_func", "sigmoid"),
        hidden_act=base_config_dict.get("hidden_act", "silu"),
        max_position_embeddings=base_config_dict.get("max_position_embeddings", 4096),
        initializer_range=base_config_dict.get("initializer_range", 0.02),
        rms_norm_eps=base_config_dict.get("rms_norm_eps", 1e-6),
        use_cache=base_config_dict.get("use_cache", True),
        pad_token_id=base_config_dict.get("pad_token_id", None),
        bos_token_id=base_config_dict.get("bos_token_id", 0),
        eos_token_id=base_config_dict.get("eos_token_id", 1),
        tie_word_embeddings=base_config_dict.get("tie_word_embeddings", False),
        rope_theta=base_config_dict.get("rope_theta", 10000.0),
        rope_scaling=base_config_dict.get("rope_scaling", None),
        attention_bias=base_config_dict.get("attention_bias", False),
        attention_dropout=base_config_dict.get("attention_dropout", 0.0),
        # output projection parameters
        use_output_subspace=base_config_dict.get("use_output_subspace", False),
        o_latent_dim=base_config_dict.get("o_latent_dim", None),
    )
    
    return modified_config

def create_model_with_mla_o(config_dict):
    """
    Create a DeepSeek V3 model with output subspace from scratch.
    
    Args:
        config_dict: Dictionary with model configuration parameters
    """
    # Create the config directly from the config_dict
    # The output subspace parameters are already included in the config_dict
    config = create_modified_config(config_dict)
    
    # Import the model class
    from transformers import DeepseekV3ForCausalLM
    
    # Create the model
    model = DeepseekV3ForCausalLM(config)
    
    # Apply our custom class to all layers
    for i, layer in enumerate(model.model.layers):
        # Create new attention layer with modified config
        new_attention = DeepseekV3Attention(
            config=config,
            layer_idx=i
        )
        
        # Copy weights from the original attention layer
        original_attention = layer.self_attn
        
        # Copy input projection weights
        if hasattr(original_attention, 'q_proj'):
            new_attention.q_proj.weight.data = original_attention.q_proj.weight.data.clone()
        else:
            # Handle LoRA case
            new_attention.q_a_proj.weight.data = original_attention.q_a_proj.weight.data.clone()
            if hasattr(original_attention.q_a_proj, 'bias') and original_attention.q_a_proj.bias is not None:
                new_attention.q_a_proj.bias.data = original_attention.q_a_proj.bias.data.clone()
            new_attention.q_a_layernorm.weight.data = original_attention.q_a_layernorm.weight.data.clone()
            new_attention.q_b_proj.weight.data = original_attention.q_b_proj.weight.data.clone()
        
        # Copy KV projection weights
        new_attention.kv_a_proj_with_mqa.weight.data = original_attention.kv_a_proj_with_mqa.weight.data.clone()
        if hasattr(original_attention.kv_a_proj_with_mqa, 'bias') and original_attention.kv_a_proj_with_mqa.bias is not None:
            new_attention.kv_a_proj_with_mqa.bias.data = original_attention.kv_a_proj_with_mqa.bias.data.clone()
        new_attention.kv_a_layernorm.weight.data = original_attention.kv_a_layernorm.weight.data.clone()
        new_attention.kv_b_proj.weight.data = original_attention.kv_b_proj.weight.data.clone()
        
        # Handle output projection weights
        if config.use_output_subspace and config.o_latent_dim is not None:
            # Initialize the new output projections
            torch.nn.init.xavier_uniform_(new_attention.o_a_proj.weight)
            torch.nn.init.xavier_uniform_(new_attention.o_b_proj.weight)
            if new_attention.o_b_proj.bias is not None:
                torch.nn.init.zeros_(new_attention.o_b_proj.bias)
        else:
            # Copy the original output projection
            new_attention.o_proj.weight.data = original_attention.o_proj.weight.data.clone()
            if original_attention.o_proj.bias is not None:
                new_attention.o_proj.bias.data = original_attention.o_proj.bias.data.clone()
        
        # Copy rotary embedding
        new_attention.rotary_emb = original_attention.rotary_emb
        
        # Replace the attention layer
        layer.self_attn = new_attention
    
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

    # Use DataCollatorForLanguageModeling for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
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

        fp16=ptrain_cfg["fp16"],

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

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # For causal LM, we compute perplexity
        # Shift predictions and labels for next token prediction
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)
        
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        return {
            "perplexity": perplexity.item(),
            "loss": loss.item(),
        }

    # ===============================
    #           Trainer
    # ===============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,

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
