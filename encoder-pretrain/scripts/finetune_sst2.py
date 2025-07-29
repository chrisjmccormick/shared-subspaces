import os
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from transformers import BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

#from models.custom_bert_full import CustomBertForMaskedLM
from transformers.modeling_outputs import SequenceClassifierOutput

"""
class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_path, num_labels=2):
        super().__init__()
        self.bert_mlm = CustomBertForMaskedLM.from_pretrained(pretrained_path)
        self.bert = self.bert_mlm.bert
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def tokenize_fn(example, tokenizer, max_length=128):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_accuracy(preds, labels):
    return (preds == labels).sum() / len(labels)


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


def main():

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

    wandb.init(project="encoder-pretrain", name="finetune-cola", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    #model = CustomBertForSequenceClassification(pretrained_path=config["model_path"])
    model = BertForSequenceClassification.from_pretrained(config["model_path"])
    model.to(device)

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
