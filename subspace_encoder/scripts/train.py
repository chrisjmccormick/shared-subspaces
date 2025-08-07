
"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# train.py

Source for DataCollator [here](https://github.com/huggingface/transformers/blob/6dfd561d9cd722dfc09f702355518c6d09b9b4e3/src/transformers/data/data_collator.py#L764)
"""

print("Importing Packages...\n")

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

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



# To disable a warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Make sure we can import modules from the encoder-pretrain package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT", PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layers.task_heads import SharedSpaceEncoderForMaskedLM
from models.shared_space_config import SharedSpaceEncoderConfig, get_config

from utils import summarize_parameters, format_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()

"""Helper utilities are imported from utils."""


def main(config_path: str):
    """Run pre-training using the provided configuration path."""
    
    
    full_cfg, model_cfg = get_config(config_path)

    ptrain_cfg = full_cfg['pre_train']

    # Print out its shorthand name.
    print(full_cfg["shorthand"])

    # Initialize the optional stats dictionary so later assignments don't fail.
    # Configuration files in this repo don't define a "stats" key yet.
    if "stats" not in full_cfg:
        full_cfg["stats"] = {}

    # Use the standard BERT tokenizer. It will use the fast variant by default.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    assert model_cfg.vocab_size == tokenizer.vocab_size

    # Set random seed for reproducibility
    set_seed(ptrain_cfg["seed"])

    # Setup Weights & Biases
    # Use offline mode if the API key isn't in the environment variables.
    
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
            max_length=ptrain_cfg["max_seq_length"]
        )

    # Tokenizes the full dataset.
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_cpu=4, # Use more CPUs to speed it up.
        remove_columns=["text"] # Comment this
    )

    # This DataCollator:
    # - Pads examples in a batch to the same length
    # - Applies BERT's Masked Language Modeling, replacing random tokens with
    #   [MASK]. Enabled by setting `mlm_probability`.
    # - Returns input_ids, labels, and attention_mask for MLM training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=ptrain_cfg["mlm_probability"] # Default is 15%

    )

    # ========================
    #    Initialize Model
    # ========================

    model = SharedSpaceEncoderForMaskedLM(model_cfg)

    # ================================
    #       Review Configuration
    # ================================

    # Display architecture
    print(model)

    print("\n======== Model ========")
    #print(json.dumps(model_cfg, indent=2))

    print("\n======== Pre-Train ========")
    print(json.dumps(ptrain_cfg, indent=2))

    # Print out final config for quick verification
    #for k, v in ptrain_cfg.items():
    #    print(f"{k:>25}: {v:>10}")


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

    #model.config.run_name = run_name

    print(ptrain_cfg["run_name"])

    """## wandb and TrainingArguments"""

    wandb.init(
        project="encoder-pretrain-wiki103",
        name=ptrain_cfg["run_name"],
        config=full_cfg # Provides the model config, pretraining, and fine-tuning.
    )

    # ===============================
    #       Training Arguments
    # ===============================

    training_args = TrainingArguments(
        output_dir=ptrain_cfg["output_dir"],

        per_device_train_batch_size=ptrain_cfg["train_batch_size"],
        per_device_eval_batch_size=ptrain_cfg["eval_batch_size"],

        fp16=ptrain_cfg.get("fp16", False),

        learning_rate=ptrain_cfg["learning_rate"],
        #num_train_epochs=ptrain_cfg["num_train_epochs"],
        max_steps=ptrain_cfg["num_train_steps"], # Changed from max_train_steps

        dataloader_num_workers=ptrain_cfg.get("num_workers", 8),
        dataloader_pin_memory=ptrain_cfg.get("pin_memory", True),
        #dataloader_prefetch_factor = ptrain_cfg.get("prefetch_factor", 2),

        weight_decay=ptrain_cfg.get("weight_decay", 0.01),  # TODO - Add to configs

        # Learning rate warmup (10% of total steps)
        warmup_steps=int(0.1 * ptrain_cfg["num_train_steps"]),  # 5000 steps for your 50k setup
        lr_scheduler_type="linear",  # Linear warmup then decay

        # Evaluate every xx steps
        # Recent versions changed from 'evaluation_strategy'
        batch_eval_metrics = True, # To avoid OOM
        eval_strategy="steps",
        eval_steps=ptrain_cfg.get("eval_steps", 2000),
        metric_for_best_model="eval_accuracy",


        logging_steps=50,

        # Checkpoint saving
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

        # Get predicted token IDs (argmax over vocab dimension)
        predictions = predictions.argmax(axis=-1)

        # Only compute accuracy for masked tokens (labels != -100)
        mask = labels != -100

        # Calculate accuracy only on masked positions
        accuracy = (predictions[mask] == labels[mask]).mean()

        return {
            "accuracy": accuracy,
        }

    import numpy as np

    class MLMAccuracyMetric:
        """
        A stateful class to compute MLM accuracy in a batch-wise manner.
        """
        def __init__(self):
            # Initialize state variables to store running totals
            self.total_correct_predictions = 0
            self.total_masked_tokens = 0

        def __call__(self, eval_pred, compute_result=False):
            """
            This method will be called by the Trainer.
            """
            predictions, labels = eval_pred

            # Get predicted token IDs (argmax over vocab dimension)
            predictions = predictions.argmax(axis=-1)

            # Create a mask for the labeled tokens (labels != -100)
            mask = labels != -100

            # Add the number of correct predictions in this batch to the total
            self.total_correct_predictions += (predictions[mask] == labels[mask]).sum()

            # Add the number of masked tokens in this batch to the total
            self.total_masked_tokens += mask.sum()

            # If this is the final call after all batches are processed
            if compute_result:
                # Avoid division by zero
                if self.total_masked_tokens == 0:
                    accuracy = 0.0
                else:
                    accuracy = self.total_correct_predictions / self.total_masked_tokens

                # Prepare the final metrics dictionary
                metrics = {"accuracy": accuracy.item()} # .item() converts numpy/torch value to a Python scalar

                # Reset state for the next evaluation run
                self.total_correct_predictions = 0
                self.total_masked_tokens = 0

                return metrics

            # For intermediate calls, return an empty dict
            return {}

    # Instantiate your stateful metric computer
    mlm_accuracy_metric = MLMAccuracyMetric()

    # ===============================
    #           Trainer
    # ===============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],

        # Add MLM accuracy as a metric
        #compute_metrics=compute_metrics,
        compute_metrics=mlm_accuracy_metric,
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
   
        #!mkdir -p /content/checkpoints/bert_baseline_wikitext103
        #!cp -r /content/checkpoints/* /content/drive/MyDrive/encoder-pretrain-wiki103/
        shutil.copytree(
            "/content/checkpoints", 
            "/content/drive/MyDrive/encoder-pretrain-wiki103/checkpoints", 
            dirs_exist_ok = True
        )

    finally:
        # End the wandb run.
        wandb.finish()

    
if __name__ == "__main__":
    args = parse_args()
    main(args.config)
