import os
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

# For testing the transformers implementation:
from transformers.modeling_outputs import SequenceClassifierOutput

#from transformers import BertForSequenceClassification

from models.custom_bert import SubspaceBertForSequenceClassification, SubspaceBertConfig

# Filter out some unhelpful warnings cluttering the output.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

    # Checkpoints are saved here.
    model_path = "/content/shared-subspaces/encoder-pretrain/checkpoints/baseline/"

    # Confirm directory exists
    assert os.path.exists(model_path), f"Directory does not exist: {model_path}"

    config = {
        "model_path": model_path,
        "task": "cola",
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 3,
        "max_length": 128,
    }

    # Retrieve the run name used during pre-training so we can reuse it here.
    model_cfg = SubspaceBertConfig.from_pretrained(model_path)
    run_name = getattr(model_cfg, "run_name", "pretrain")
    config["run_name"] = run_name

    wandb.init(
        project="encoder-pretrain-sst2",
        name=f"{run_name}-sst2",
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # ======================
    #       Dataset
    # ======================
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    #dataset = load_dataset("glue", "cola")
    dataset = load_dataset("glue", "sst2")


    dataset = dataset.map(lambda ex: tokenize_fn(ex, tokenizer, config["max_length"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_split = int(0.9 * len(dataset["train"]))
    train_dataset = Subset(dataset["train"], range(train_split))
    val_dataset = Subset(dataset["train"], range(train_split, len(dataset["train"])))
    test_dataset = dataset["validation"]

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # ======================
    #       Model
    # ======================

    model = SubspaceBertForSequenceClassification.from_pretrained(config["model_path"])
    model.to(device)

    # ======================
    #       Training
    # ======================

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    num_training_steps = config["epochs"] * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    try:
        for epoch in range(config["epochs"]):
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
        
        wandb.finish()


if __name__ == "__main__":
    main()
