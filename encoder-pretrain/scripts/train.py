import json
import argparse

import wandb
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

from encoder_pretrain.models.custom_bert import CustomBertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    wandb.init(project="encoder-pretrain", config=cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = load_dataset(cfg["dataset_name"], cfg["dataset_config"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg["max_seq_length"])

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg["mlm_probability"])

    model = CustomBertForMaskedLM.from_config(cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=50,
        evaluation_strategy="steps",
        save_steps=500,
        report_to=["wandb"],
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
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
