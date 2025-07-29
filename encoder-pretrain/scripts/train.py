import json
import argparse
from pathlib import Path
import sys
import os

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

from transformers import AutoModelForMaskedLM, AutoConfig


# Make sure we can import modules from the encoder-pretrain package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT", PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Temp - Using base class for the moment.
#from models.custom_bert import CustomBertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

"""**Helper Function for Formatting Counts**"""

def format_size(num):
    """
    This function iterates through a list of suffixes ('K', 'M', 'B') and
    divides the input number by 1024 until the absolute value of the number is
    less than 1024. Then, it formats the number with the appropriate suffix and
    returns the result. If the number is larger than "B", it uses 'T'.
    """
    suffixes = [' ', 'K', 'M', 'B'] # Return an empty space if it's less than 1K,
                                    # this helps highlight the larger values.

    base = 1024

    for suffix in suffixes:
        if abs(num) < base:
            if num % 1 != 0:
                return f"{num:.2f}{suffix}"

            else:
                return f"{num:.0f}{suffix}"

        num /= base

    # Use "T" for anything larger.
    if num % 1 != 0:
        return f"{num:.2f}T"

    else:
        return f"{num:.0f}T"



def main():
    args = parse_args()
    
    with open(args.config) as f:
        cfg = json.load(f)

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

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = load_dataset(cfg["dataset_name"], cfg["dataset_config"])

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
    config = AutoConfig.from_pretrained(cfg["model_name"])
    model = AutoModelForMaskedLM.from_config(config)
    
    model

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        
        fp16=cfg.get("fp16", False),
        
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        
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
        remove_unused_columns=False,  # Optional: avoid dropping custom model inputs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        
        # New argument, allows for other modalities.
        processing_class=tokenizer,
        
        data_collator=data_collator,
    )

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
    #     Total Params
    # =====================

    total_params = 0
    for p_name, p in params:
        total_params += p.numel()

    cfg["total_elements"] = format_size(total_params)

    print(f"Total elements: {cfg['total_elements']}\n")

    # Print out final config
    for k, v in config.items():
        print(f"{k:>15}: {v:>10}")
  
    print("=============================\n")

    # Format the cfg learning rate as a scientific notation string like 5e-4
    lr_str = '{:.4e}'.format(cfg['learning_rate'])

    run_name = f"{cfg['total_elements']} - bert scratch - h{config.hidden_size} - l{config.num_hidden_layers} - bs{cfg['train_batch_size']} - lr{lr_str} - seq{cfg['max_seq_length']}"

    wandb.init(
        project="encoder-pretrain", 
        run_name=run_name,
        config=cfg
    )

    # =====================
    #     Run Training
    # =====================

    try:
        trainer.train()
        metrics = trainer.evaluate()
        wandb.log(metrics)
    
    finally:
        wandb.finish()
        trainer.save_model(cfg["output_dir"])
    

if __name__ == "__main__":
    main()
