from transformers import AutoTokenizer, BertForMaskedLM
import torch

# Load model from your training checkpoint directory
model_path = "../checkpoints/baseline/"

# Use the tokenizer you originally trained with
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ FIXED: Use `from_pretrained(...)`, not just `BertForMaskedLM(...)`
model = BertForMaskedLM.from_pretrained(model_path)
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
