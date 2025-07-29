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

# Make sure we can import modules from the encoder-pretrain package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("PROJECT_ROOT", PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.custom_bert import CustomBertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()


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
    wandb.init(project="encoder-pretrain", config=cfg)

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

    model = CustomBertForMaskedLM.from_config(cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        
        # Evaluate every xx steps
        evaluation_strategy="steps",
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
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    wandb.log(metrics)

    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
