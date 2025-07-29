from transformers import AutoTokenizer, BertForMaskedLM
import torch
from pathlib import Path

# Load model from your training checkpoint directory
model_path = "../checkpoints/baseline/"

# Use the tokenizer you originally trained with
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model_path = Path("../checkpoints/baseline").resolve()

print("Passing repo id as:", str(model_path))

# Load the checkpoint
model = BertForMaskedLM.from_pretrained(
    str(model_path),
    local_files_only=True
)
model.eval()


# Run a test input
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Identify the [MASK] token position
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Get the predicted token ID at that position
predicted_token_id = logits[0, mask_token_index].argmax(-1)

# Decode the prediction
predicted_token = tokenizer.decode(predicted_token_id)

print("Prediction:", predicted_token)
