import os
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm

from models.custom_bert_full import CustomBertForMaskedLM  # Assuming this includes BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomBertForSequenceClassification(nn.Module):
    """Wraps a pretrained BERT encoder with a classification head for CoLA."""
    def __init__(self, pretrained_path, num_labels=2):
        super().__init__()
        self.bert_mlm = CustomBertForMaskedLM.from_pretrained(pretrained_path)
        self.bert = self.bert_mlm.bert  # Grab encoder only
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # shape: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def tokenize_fn(example, tokenizer, max_length=128):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_accuracy(preds, labels):
    return (preds == labels).sum() / len(labels)


def main():
    config = {
        "model_path": "checkpoints/mla_output",  # Adjust if needed
        "task": "cola",
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 3,
        "max_length": 128,
    }

    wandb.init(project="encoder-pretrain", name="finetune-cola", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "cola")
    dataset = dataset.map(lambda ex: tokenize_fn(ex, tokenizer, config["max_length"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(dataset["validation"], batch_size=config["batch_size"])

    model = CustomBertForSequenceClassification(pretrained_path=config["model_path"], num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    num_training_steps = config["epochs"] * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
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

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["label"].cpu().tolist())

        acc = compute_accuracy(torch.tensor(all_preds), torch.tensor(all_labels))

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "eval_accuracy": acc.item()
        })

        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Eval Acc: {acc:.4f}")


if __name__ == "__main__":
    main()
