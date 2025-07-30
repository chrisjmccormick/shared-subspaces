import json
import argparse
from pathlib import Path
import sys
import os

import torch
from torch import profiler as torch_profiler

import wandb
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# This file exists in the 'scripts' subdirectory, go up a level to find 'models'.
from models.custom_bert import SubspaceBertForMaskedLM, SubspaceBertConfig
from utils import summarize_parameters, format_size

# Make sure we can import modules from the encoder-pretrain package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Temp - Using base class for the moment.
#from models.custom_bert import CustomBertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

"""Helper utilities are imported from utils."""



def main():
    args = parse_args()
    
    with open(args.config) as f:
        cfg = json.load(f)

    # Some keys in the configs have older names. Normalize them here so the rest
    # of the script can rely on a consistent set of fields.
    cfg["ffn_decompose"] = cfg.get("ffn_decompose", cfg.get("use_decomp_mlp", False))
    cfg["output_subspace"] = cfg.get("output_subspace", cfg.get("use_output_latent", False))
    # Default max_steps for profiling; keeps runs short unless overridden
    cfg["max_steps"] = cfg.get("max_steps", 10)

    print("Transformers version:", transformers.__version__)  # Helpful sanity check

    # Set random seed for reproducibility
    set_seed(cfg.get("seed", 42))

    # Setup Weights & Biases
    # Use offline mode if the API key isn't in the environment variables.
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    
    # Use the standard BERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cfg["vocab_size"] = tokenizer.vocab_size
    
    
    dataset = load_dataset(
        cfg["dataset_name"], 
        cfg["dataset_config"]
    )
    

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=cfg["max_seq_length"]
        )

    tokenized = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"] # Comment this
    )

    # This DataCollator:
    # - Pads examples in a batch to the same length
    # - Applies BERT's Masked Language Modeling, replacing random tokens with 
    #   [MASK]. Enabled by setting `mlm_probability`.
    # - Returns input_ids, labels, and attention_mask for MLM training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=cfg["mlm_probability"] # Default is 15%
    )

    #model = CustomBertForMaskedLM.from_config(cfg)

    bert_config = SubspaceBertConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg.get("max_position_embeddings", 512),
        type_vocab_size=cfg.get("type_vocab_size", 2),
        hidden_act=cfg.get("hidden_act", "gelu"),
        hidden_dropout_prob=cfg.get("hidden_dropout_prob", 0.1),
        attention_dropout_prob=cfg.get("attention_dropout_prob", 0.1),
        use_mla=cfg.get("use_mla", False),
        kv_lora_rank=cfg.get("kv_lora_rank"),
        q_lora_rank=cfg.get("q_lora_rank"),
        o_lora_rank=cfg.get("o_lora_rank"),
        qk_rope_head_dim=cfg.get("qk_rope_head_dim"),
        v_head_dim=cfg.get("v_head_dim"),
        qk_nope_head_dim=cfg.get("qk_nope_head_dim"),
        output_subspace=cfg.get("output_subspace", False),
        num_dense_layers=cfg.get("num_dense_layers"),
        ffn_rank=cfg.get("ffn_rank")
    )

    model = SubspaceBertForMaskedLM(bert_config)
    
    model


    print("\n======== Config File ========")
    print(json.dumps(cfg, indent=2))
    print("=============================\n")
  
    """# Review

    ### Parameter Tally
    """

    print("\n======== Parameters ========")

    ## Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The model has {:} different named parameters.\n'.format(len(params)))

    # =====================
    #     Review Params
    # =====================

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    cfg["total_elements"] = format_size(total_params)

    print(f"Total elements: {cfg['total_elements']}\n")

    # Print out final config for quick verification
    for k, v in cfg.items():
        print(f"{k:>25}: {v:>10}")

    print("=============================\n")

    # Display a full parameter breakdown using the shared utility
    summarize_parameters(model)


    # ========================================
    #   Format Settings for WandB Run Name
    # ========================================
    
    # Format the cfg learning rate as a scientific notation string like 5e-4
    lr_str = '{:.0e}'.format(cfg['learning_rate'])

    # Attention configuration
    if cfg.get("use_mla"):
        dense_str = str(cfg.get("num_dense_layers")) + "mha + "

        # If no output subspace is used, the dimension will show as -1.
        attn_str = (
            dense_str
            + "mla."
            + str(cfg.get("q_lora_rank"))
            + "."
            + str(cfg.get("kv_lora_rank"))
            + "."
            + str(cfg.get("o_lora_rank"))
        )
    else:
        attn_str = "mha"

    # MLP Configuration
    if cfg.get("ffn_decompose"):
        # Specify the number of dense mlps and their size.
        dense_str = (
            str(cfg.get("num_dense_layers"))
            + "mlp."
            + str(cfg.get("intermediate_size"))
            + " + "
        )

        # Specify the neuron count and latent dimension of the decomposed mlps.
        mlp_str = (
            dense_str
            + "dcmp."
            + str(cfg.get("intermediate_size"))
            + "."
            + str(cfg.get("ffn_rank"))
    )
    else:
        mlp_str = "mlp." + str(cfg.get("intermediate_size"))


    run_name = f"{cfg['total_elements']} - {attn_str} - {mlp_str} - h{cfg['hidden_size']} - l{cfg['num_hidden_layers']} - bs{cfg['train_batch_size']} - lr{lr_str} - seq{cfg['max_seq_length']}"

    cfg["run_name"] = run_name
    model.config.run_name = run_name

    print(run_name)





    wandb.init(
        project="encoder-pretrain", 
        name=run_name,
        config=cfg
    )

    # ===============================
    #       Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        
        fp16=cfg.get("fp16", False),
        
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        # Limit steps for profiling so execution finishes quickly
        max_steps=cfg.get("max_steps", 10),
        
        # Evaluate every xx steps
        # Recent versions changed from 'evaluation_strategy'
        eval_strategy="steps", 
        eval_steps=cfg.get("eval_steps", 500), # Default to 500
        
        logging_steps=50,
        
        # Checkpoint saving
        save_steps=500,
        save_total_limit=2,           # Optional: keeps last 2 checkpoints
        save_strategy="steps",
        report_to=["wandb"],
        run_name=run_name,
        remove_unused_columns=False,  # Optional: avoid dropping custom model inputs
    )

    # ===============================
    #           Trainer
    # ===============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        
        # New argument, allows for other modalities.
        processing_class=tokenizer,
        
        data_collator=data_collator,
    )

    # =====================
    #     Run Training
    # =====================

    try:
        # Profile the training run to collect performance information.
        # The context records all operations executed by `trainer.train`.
        with torch_profiler.profile(
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.GPU],
            record_shapes=True,
        ) as prof:
            trainer.train()

        # Display a summary of the profiling results.
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="gpu_time_total", row_limit=10))

        metrics = trainer.evaluate()
        wandb.log(metrics)
    
    finally:
        wandb.finish()
        trainer.save_model(cfg["output_dir"])
    

if __name__ == "__main__":
    main()
