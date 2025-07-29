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
)

# Make sure we can import modules from the encoder-pretrain package even if the
# directory name contains a hyphen. When running this script directly the
# parent directory of this file is ``encoder-pretrain``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    wandb.init(project="encoder-pretrain", config=cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = load_dataset(cfg["dataset_name"], cfg["dataset_config"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg["max_seq_length"])

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg["mlm_probability"])

    model = CustomBertForMaskedLM.from_config(cfg)

    hf_version = transformers.__version__
    kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=50,
        save_steps=500,
        report_to=["wandb"],
    )
    if tuple(map(int, hf_version.split(".")[:2])) >= (3, 1):
        kwargs["evaluation_strategy"] = "steps"
    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
