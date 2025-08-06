"""# ▂▂▂▂▂▂▂▂▂▂▂▂

# Fine-Tune.py
"""

"""Fine-tune a pretrained encoder on the SST-2 task. The original script
assumed a fixed checkpoint directory. We now accept a config file on the
command line so the script can locate the appropriate `output_dir` created
during pretraining."""

import os
import argparse
import json
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler, set_seed

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

# For testing the transformers implementation:
from transformers.modeling_outputs import SequenceClassifierOutput

#from transformers import BertForSequenceClassification

from layers.task_heads import SharedSpaceEncoderForSequenceClassification
from models.shared_space_config import SharedSpaceEncoderConfig, get_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Flat accuracy.
def compute_accuracy(preds, labels):
    return (preds == labels).sum() / len(labels)

# Evaluate model on dataset in `dataloader`
def run_eval(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
    preds = torch.tensor(all_preds)
    labels = torch.tensor(all_labels)
    acc = compute_accuracy(preds, labels).item()
    mcc = matthews_corrcoef(labels, preds)
    return acc, mcc

# =================
#      Main
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

    # Fine-tuning hyperparameters are now provided under "fine_tune".

    run_name = full_cfg["pre_train"]["run_name"] + f' - pt_id.{full_cfg["pre_train"]["run_id"]} - {full_cfg["fine_tune"]["task"]}'
    full_cfg["fine_tune"]["run_name"] = run_name
    full_cfg["fine_tune"]["model_path"] = checkpoint_path
    full_cfg["fine_tune"]["tuned_from_id"] = full_cfg["pre_train"]["run_id"]

    # Log the complete config file with the training run.
    wandb.init(
        project="encoder-pretrain-sst2",
        name=run_name,
        config=full_cfg
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get a variable for just the fine-tuning stuff.
    sft_cfg = full_cfg["fine_tune"]

    # Set random seed for reproducibility
    set_seed(sft_cfg["seed"])


    # ======================
    #       Dataset
    # ======================

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load the GLUE task specified in the sft_cfg.
    dataset = load_dataset("glue", sft_cfg["task"])


    dataset = dataset.map(lambda ex: tokenize_fn(ex, tokenizer, sft_cfg["max_length"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_split = int(0.9 * len(dataset["train"]))
    train_dataset = Subset(dataset["train"], range(train_split))
    val_dataset = Subset(dataset["train"], range(train_split, len(dataset["train"])))
    test_dataset = dataset["validation"]


    train_loader = DataLoader(
        train_dataset,
        batch_size=sft_cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=sft_cfg["batch_size"],
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=sft_cfg["batch_size"],
        num_workers=8,
        pin_memory=True
    )

    # ======================
    #       Model
    # ======================

    model = SharedSpaceEncoderForSequenceClassification.from_pretrained(
        sft_cfg["model_path"],
        config = model_cfg
    )

    model.to(device)

    # ======================
    #       Training
    # ======================

    optimizer = AdamW(
        model.parameters(),
        lr=sft_cfg["lr"],
        #weight_decay = 0.01 # This run did not have weight decay enabled.
    )
    num_training_steps = sft_cfg["epochs"] * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    try:
        for epoch in range(sft_cfg["epochs"]):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            val_acc, val_mcc = run_eval(model, val_loader, device)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_accuracy": val_acc,
                "val_mcc": val_mcc
            })

            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | MCC: {val_mcc:.4f}")
    finally:
        # Final evaluation on test set
        test_acc, test_mcc = run_eval(model, test_loader, device)

        wandb.log({
            "test_accuracy": test_acc,
            "test_mcc": test_mcc
        })

        print(f"[FINAL] Test Accuracy: {test_acc:.4f} | Test MCC: {test_mcc:.4f}")

        full_cfg["fine_tune"]["run_id"] = wandb.run.id
        full_cfg["fine_tune"]["run_url"] = wandb.run.url
        
        # Update the run with the updated config object.
        #wandb.run.config = full_cfg
        
        # Save the json back to disk
        with open(checkpoint_path + "/full_config.json", "w") as f:
            json.dump(full_cfg, f, indent=2)

        wandb.finish()


if __name__ == "__main__":
    main()


