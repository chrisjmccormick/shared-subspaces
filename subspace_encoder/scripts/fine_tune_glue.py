"""
# ▂▂▂▂▂▂▂▂▂▂▂▂

# Fine-Tune.py (Refactored with HuggingFace Trainer)
"""

"""Fine-tune a pretrained encoder on the SST-2 task. The original script
assumed a fixed checkpoint directory. We now accept a config file on the
command line so the script can locate the appropriate `output_dir` created
during pretraining."""

import os

# Disable tensorflow to avoid noisy warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import json
import wandb
import torch
import numpy as np 

from datasets import load_dataset

from torch.utils.data import Subset

from transformers import (
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

# Import custom model classes
from layers.task_heads import SharedSpaceEncoderForSequenceClassification
from models.shared_space_config import SharedSpaceEncoderConfig, get_config

# Disable a noisy warning from huggingface caused by using multiple
# workers in the data collator.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set W&B project to report to.
os.environ["WANDB_PROJECT"] = "subspace-encoder-sst2"

# Filter out some unhelpful warnings cluttering the output.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    """Return CLI args with path to a config JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()


# Wrapt the tokenizer so that we can use dataset.map further down.
def tokenize_fn(example, tokenizer, max_length=128):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics(p: EvalPrediction):
    """Computes flat accuracy for a given prediction."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = (preds == labels).mean()
    return {"accuracy": acc}


# =================
#       Main
# =================

def main():
    args = parse_args()

    # The `full_cfg` is a multi-level dictionary.
    # The `model_cfg` is the `*Config` class.
    full_cfg, model_cfg = get_config(args.config)

    # Specify the checkpoint with the best accuracy.
    checkpoint_path = full_cfg["pre_train"]["best_checkpoint"]

    # Confirm directory exists
    assert os.path.exists(checkpoint_path), f"Directory does not exist: {checkpoint_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get a variable for just the fine-tuning stuff.
    sft_cfg = full_cfg["fine_tune"]

    # Set random seed for reproducibility
    set_seed(sft_cfg["seed"])

    # Define run name for logging
    run_name = full_cfg["pre_train"]["run_name"] + f' - pt_id.{full_cfg["pre_train"]["run_id"]} - {sft_cfg["task"]}'
    full_cfg["fine_tune"]["run_name"] = run_name
    full_cfg["fine_tune"]["model_path"] = checkpoint_path
    full_cfg["fine_tune"]["tuned_from_id"] = full_cfg["pre_train"]["run_id"]


    # ======================
    #         Dataset
    # ======================

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load the GLUE task specified in the sft_cfg.
    dataset = load_dataset("glue", sft_cfg["task"])

    print("\n======== Tokenizing Dataset ========\n")

    # Tokenize the dataset.
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, sft_cfg["max_length"]),
        batched=True,
        remove_columns=["sentence", "idx"] # Remove original columns
    )
    tokenized_dataset.set_format(type="torch")

    # Because the SST-2 dataset does not provide labels for the test sampels,
    # We'll set aside 10% of the training data to use for validation during 
    # training, (to assess overfitting and determine our best checkpoint)
    # and then use the provided "validation" set as our test set.
    train_split = int(0.9 * len(tokenized_dataset["train"]))
    train_dataset = Subset(tokenized_dataset["train"], range(train_split))
    val_dataset = Subset(tokenized_dataset["train"], range(train_split, len(tokenized_dataset["train"])))
    test_dataset = tokenized_dataset["validation"]

    # Calculate the total train steps based on number of epochs, dataset size,
    # and train batch size
    num_training_steps = sft_cfg["epochs"] * len(train_dataset) // sft_cfg["batch_size"]

    print(f"\nTotal Training Steps: {num_training_steps:,}")

    # ======================
    #         Model
    # ======================

    print("\n======== Loading Model ========\n")

    model = SharedSpaceEncoderForSequenceClassification.from_pretrained(
        sft_cfg["model_path"],
        config=model_cfg
    )

    model.to(device)

    # =========================
    #      TrainingArguments
    # =========================

    training_args = TrainingArguments(
        
        #output_dir=f"./results/{run_name}",  # TODO....
        
        run_name=run_name,

        # Hyperparameters
        num_train_epochs=sft_cfg["epochs"],

        per_device_train_batch_size=sft_cfg["batch_size"],
        per_device_eval_batch_size=sft_cfg["batch_size"],

        learning_rate=sft_cfg["lr"],
        lr_scheduler_type="linear",  # Linear warmup then decay

        # These would be good to add, but weren't present in my
        # existing runs. I'll need to redo the fine-tuning experiments.
        #warmup_ratio = 0.1, # Use the first 10% of the training steps for warmup.
        #weight_decay = 0.01, # Regularization to avoid over-fitting.
        

        # Evaluation and Logging
        eval_strategy="steps",
        eval_steps = 1000, # Eval ~11 times over the course of training
     
        logging_strategy="steps",
        logging_steps=100,
        logging_dir="./logs",
        
        # W&B Integration
        report_to="wandb",
        disable_tqdm = True, # Let's try true

        # Dataloader performance
        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        # Reproducibility
        seed=sft_cfg["seed"],
    )

    # Log the complete config file with the training run.
    wandb.init(config=full_cfg, name=run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ======================
    #        Training
    # ======================
    # We'll use a try / finally block to ensure wandb.finish() is called.
    try:
        trainer.train()
    
        # ======================
        #   Final Evaluation
        # ======================
        
        print("\n--- Evaluating on Test Set ---")
        test_results = trainer.predict(test_dataset)

        # The predict output contains metrics, which we can log and print.
        # We rename them to distinguish from validation metrics in W&B.
        final_test_metrics = {
            "final_test_accuracy": test_results.metrics["test_accuracy"]
        }
        wandb.log(final_test_metrics)
        print(f"[FINAL] Test Accuracy: {final_test_metrics['final_test_accuracy']:.4f}")

        full_cfg["fine_tune"]["run_id"] = wandb.run.id
        full_cfg["fine_tune"]["run_url"] = wandb.run.url

        # Save the json back to disk
        with open(os.path.join(checkpoint_path, "full_config.json"), "w") as f:
            json.dump(full_cfg, f, indent=2)
   
    # Ensure we get to call `finish`, even if training is interrupted.
    finally:

        wandb.finish()
    


if __name__ == "__main__":
    main()