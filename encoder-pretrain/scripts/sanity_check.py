from transformers import AutoTokenizer, BertForMaskedLM
import torch
from pathlib import Path
import os

# Load model from your training checkpoint directory
model_path = "../checkpoints/baseline/"

# Use the tokenizer you originally trained with
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model_path = Path("../encoder-pretrain/checkpoints/baseline").resolve()

# Confirm directory exists
assert os.path.exists(model_path), f"Directory does not exist: {model_path}"


print("Passing repo id as:", str(model_path))

# Load the checkpoint
model = BertForMaskedLM.from_pretrained(
    str(model_path),
    local_files_only=True,
    trust_remote_code=True,  # Usually safe for local loads
    use_safetensors=True     # 
)
model.eval()

"""**Helper Function for Formatting Counts**"""

def format_size(num):
    """
    This function iterates through a list of suffixes ('K', 'M', 'B') and
    divides the input number by 1024 until the absolute value of the number is
    less than 1024. Then, it formats the number with the appropriate suffix and
    returns the result. If the number is larger than "B", it uses 'T'.
    """
    suffixes = [' ', 'K', 'M', 'B'] # Return an empty space if it's less than 1K,
                                    # this helps highlight the larger values.

    base = 1024

    for suffix in suffixes:
        if abs(num) < base:
            if num % 1 != 0:
                return f"{num:.2f}{suffix}"

            else:
                return f"{num:.0f}{suffix}"

        num /= base

    # Use "T" for anything larger.
    if num % 1 != 0:
        return f"{num:.2f}T"

    else:
        return f"{num:.0f}T"

print("\n======== Parameters ========")

## Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The model has {:} different named parameters.\n'.format(len(params)))

# =====================
#     Review Params
# =====================

total_params = 0
for p_name, p in params:
    total_params += p.numel()

cfg["total_elements"] = format_size(total_params)

print(f"Total elements: {cfg['total_elements']}\n")

# Print out final config
for k, v in cfg.items():
    print(f"{k:>25}: {v:>10}")

print("=============================\n")

"""## Full Parameter List"""

display_bias = True # Excludes any 1-D parameters.

include_layers = []

print("Parameter Name                                              Dimensions       Total Values    Trainable\n")

for p_name, p in params:

    # Loop through the parameter dimensions and delete any == 1.
    p_size = list(p.size())

    for i in range(len(p_size) - 1, -1, -1):
        if p_size[i] == 1:
            del p_size[i]

    if len(p_size) == 1:
        if not display_bias:
            continue
        p_dims = "{:>10,} x {:<10}".format(p.size()[0], "-")

    elif len(p_size) == 2:
        p_dims = "{:>10,} x {:<10,}".format(p.size()[0], p.size()[1])
    elif len(p_size) == 3:
        p_dims = "{:>10,} x {:,} x {:<10}".format(p.size()[0], p.size()[1], p.size()[2])
    elif len(p_size) == 4:
        p_dims = "{:>10,} x {:,} x {:,} x {:<10}".format(p.size()[0], p.size()[1], p.size()[2], p.size()[3])
    else:
        print("Unexpected: ", p.size(), p_name)
        break

    print("{:<55} {:}    {:>6}    {:}".format(p_name, p_dims, format_size(p.numel()), p.requires_grad))


print(f"\nTotal elements: {format_size(total_params)}\n")





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
