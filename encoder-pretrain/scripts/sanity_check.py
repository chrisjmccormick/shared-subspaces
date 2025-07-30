from transformers import AutoTokenizer
import torch
from pathlib import Path
import os


from models.custom_bert import SubspaceBertForMaskedLM, SubspaceBertConfig
from utils import summarize_parameters, format_size

# Load model from your training checkpoint directory

# Use the tokenizer you originally trained with
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model_path = Path("../encoder-pretrain/checkpoints/baseline").resolve()

# Confirm directory exists
assert os.path.exists(model_path), f"Directory does not exist: {model_path}"

# Retrieve the run name used during pre-training so we can reuse it here.
model_cfg = SubspaceBertConfig.from_pretrained(model_path)
print("Run name:", getattr(model_cfg, "run_name", "pretrain"))


print("Passing repo id as:", str(model_path))

# Load the checkpoint
model = SubspaceBertForMaskedLM.from_pretrained(
    str(model_path),
    local_files_only=True,
    trust_remote_code=True,  # Usually safe for local loads
    use_safetensors=True     # 
)
model.eval()

print("\n======== Parameters ========")

# Use the shared utility to list model parameters and dimensions
summarize_parameters(model)




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
