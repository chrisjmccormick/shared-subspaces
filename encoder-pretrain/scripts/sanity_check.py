from transformers import AutoTokenizer
from models.custom_bert_full import CustomBertForMaskedLM
import torch

model_path = "checkpoints/mla_output"  # or whatever you trained
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = CustomBertForMaskedLM.from_pretrained(model_path)
model.eval()

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted token
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = logits[0, mask_token_index].argmax(-1)
predicted_token = tokenizer.decode(predicted_token_id)

print("Prediction:", predicted_token)
