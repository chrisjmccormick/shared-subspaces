# -*- coding: utf-8 -*-
# Updated training script for DeepSeek V3 with attention output subspace

"""# subspace_decoder/scripts/train.py"""


import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"              # older check some codepaths still honor
# Optional: if Keras 3 is on the system and ever gets touched, force non-TF backend
os.environ.setdefault("KERAS_BACKEND", "torch")

from transformers.utils import is_tf_available
print("TF available (Transformers thinks):", is_tf_available())  # should be False


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

from models.shared_space_config import SharedSpaceDecoderConfig, get_config
from layers.task_heads import SharedSpaceDecoderForCausalLM

import torch.nn as nn


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


def main(config_path: str):
    """Run pre-training using the provided configuration path."""
    
    # Load configuration
    full_cfg, model_cfg = get_config(config_path)

    ptrain_cfg = full_cfg['pre_train']

    # Print out its shorthand name.
    print(full_cfg["shorthand"])

    # Initialize the optional stats dictionary so later assignments don't fail.
    if "stats" not in full_cfg:
        full_cfg["stats"] = {}
    
    # Validate mixed precision settings
    if ptrain_cfg["bf16"] and ptrain_cfg["fp16"]:
        raise ValueError("Cannot enable both bf16 and fp16 simultaneously. Please choose one.")
    
    # Check BFloat16 compatibility if enabled
    if ptrain_cfg["bf16"]:
        if not check_bf16_support():
            print("BFloat16 requested but not supported. Falling back to FP16.")
            ptrain_cfg["bf16"] = False
            ptrain_cfg["fp16"] = True
    
    # Display torch.compile status
    if ptrain_cfg["torch_compile"]:
        print(f"✓ torch.compile enabled:")
        print(f"  Backend: {ptrain_cfg['torch_compile_backend']}")
        print(f"  Mode: {ptrain_cfg['torch_compile_mode']}")
        print("  Note: First training step will be slower due to compilation.")
    else:
        print("torch.compile disabled. Enable with 'torch_compile': true in config.")

    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # gpt2 has no pad by default; use EOS for padding in causal LM
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
        
    # Verify vocab size matches
    assert model_cfg.vocab_size == tokenizer.vocab_size

    # Set random seed for reproducibility
    set_seed(ptrain_cfg["seed"])

    # Setup Weights & Biases
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    # ======================
    #    Load Dataset
    # ======================
    
    dataset_name = ptrain_cfg["dataset_name"]
    dataset_config = ptrain_cfg["dataset_config"]
    
    # Check if we should load a pre-processed dataset
    if "preprocessed_dataset_path" in ptrain_cfg and ptrain_cfg["preprocessed_dataset_path"]:
        print(f"Loading pre-processed dataset from: {ptrain_cfg['preprocessed_dataset_path']}")
        from datasets import load_from_disk
        
        dataset = load_from_disk(ptrain_cfg["preprocessed_dataset_path"])
        print(f"Loaded pre-processed dataset:")
        print(f"  Train: {len(dataset['train']):,} examples")
        print(f"  Validation: {len(dataset['validation']):,} examples")
        
        # Skip tokenization and chunking since it's already done
        tokenized = dataset
        
    elif dataset_name == "wikitext":
        
        # Original logic for wikitext and other datasets
        dataset = load_dataset(dataset_name, dataset_config)
    elif dataset_name == "allenai/c4":
        raise ValueError(f"allenai/c4 requires prep-processing, but no preprocessed dataset path was provided")
            
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    print(dataset)
    
    # ========================
    #    Tokenize Wikitext
    # ========================

    if dataset_name == "wikitext":

        block_size = ptrain_cfg["max_seq_length"]
        eos_id = tokenizer.eos_token_id
        
        # 1) Tokenize without truncation/padding
        def tokenize_function(examples):
            # add_special_tokens=False keeps things raw; we'll insert EOS between docs
            return tokenizer(
                examples["text"],
                add_special_tokens=False,
            )
        
        # 2) Group into contiguous blocks (concat + chunk)
        def group_texts(examples):
            # Flatten and insert EOS between documents to avoid cross-article bleed
            input_ids = []
            for ids in examples["input_ids"]:
                if len(ids) > 0:
                    input_ids.extend(ids)
                # add an EOS fencepost between docs
                input_ids.append(eos_id)
        
            # Drop the trailing partial block so every example is full length
            total_length = (len(input_ids) // block_size) * block_size
            input_ids = input_ids[:total_length]
        
            # Split into equal blocks
            result_input_ids = [input_ids[i:i + block_size] for i in range(0, total_length, block_size)]
            # Labels are next-token targets; Trainer/model will do the shift
            return {
                "input_ids": result_input_ids,
                "labels": [ids.copy() for ids in result_input_ids],
                # Optional attention masks (all ones because no padding)
                "attention_mask": [[1] * block_size for _ in result_input_ids],
            }
        
        # Tokenize
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=dataset["train"].column_names,  # drop raw "text"
        )
        
        # Concatenate + chunk
        tokenized = tokenized.map(
            group_texts,
            batched=True,
            num_proc=8,
        )


    # Use a simple collator; we already created labels and have no pads
    from transformers import default_data_collator
    data_collator = default_data_collator


    # ========================
    #    Initialize Model
    # ========================

    print("Initializing model...")

    model = SharedSpaceDecoderForCausalLM(model_cfg)

    # ================================
    #       Review Configuration
    # ================================

    # Display architecture
    print(model)

    print("\n======== Model ========")
    print(model_cfg)

    print("\n======== Pre-Train ========")
    print(json.dumps(ptrain_cfg, indent=2))

    # Calculate and display effective batch size
    device_batch_size = ptrain_cfg["train_batch_size"]
    gradient_accumulation_steps = ptrain_cfg["gradient_accumulation_steps"]
    effective_batch_size = device_batch_size * gradient_accumulation_steps
    
    print(f"\n======== Batch Size Configuration ========")
    print(f"Device batch size: {device_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

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
        project=ptrain_cfg["wandb_project"],
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
        gradient_accumulation_steps=ptrain_cfg["gradient_accumulation_steps"],

        bf16=ptrain_cfg["bf16"],
        fp16=ptrain_cfg["fp16"],
        
        # torch.compile configuration for performance optimization
        torch_compile=ptrain_cfg["torch_compile"],
        torch_compile_backend=ptrain_cfg["torch_compile_backend"],
        torch_compile_mode=ptrain_cfg["torch_compile_mode"],

        learning_rate=ptrain_cfg["learning_rate"],
        max_steps=ptrain_cfg["num_train_steps"], 

        # TODO - Added this to recent 576 runs, but need to decide if it's needed.
        #max_grad_norm = 1.0,
        
        # The dataloader is a bottleneck without these.
        dataloader_num_workers=ptrain_cfg["num_workers"],
        dataloader_pin_memory=ptrain_cfg["pin_memory"],
        # The prefetch factor didn't appear to help.
        #dataloader_prefetch_factor = ptrain_cfg["prefetch_factor"],

        weight_decay=ptrain_cfg["weight_decay"],  

        # Learning rate warmup (10% of total steps)
        warmup_steps=int(0.1 * ptrain_cfg["num_train_steps"]),  
        lr_scheduler_type="linear",  # Linear warmup then decay

        # Evaluate every 2,000 steps
        # Note: Recent versions of Trainer changed the name from 
        # `evaluation_strategy` to `eval_strategy`.
        batch_eval_metrics = True, # To avoid OOM
        eval_strategy="steps",
        eval_steps=ptrain_cfg["eval_steps"],
        eval_accumulation_steps=4,  # Process eval in smaller chunks to save memory

        logging_steps=ptrain_cfg["logging_steps"],
        metric_for_best_model="eval_loss",
        save_steps=ptrain_cfg["save_steps"], 
        save_total_limit=2,           # Optional: keeps last 2 checkpoints
        save_strategy="steps",
        report_to=["wandb"],

        save_safetensors=False,
        
        run_name=ptrain_cfg["run_name"],
        
        remove_unused_columns=False,  # Optional: avoid dropping custom model inputs
    )

    # Print out all of the settings in TrainingArguments using tabulate.
    # Note that they are not in alphabetical order, but that the order appears
    # to be more sensible than that. (e.g., all batch size related arguments
    # are together).
    import tabulate
    print("Training Arguments:")
    print(tabulate.tabulate(vars(training_args).items(), headers=["Argument", "Value"]))

    # ==========================
    #     Perplexity Metric
    # ==========================

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
