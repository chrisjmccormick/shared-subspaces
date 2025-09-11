#!/usr/bin/env bash
set -euo pipefail

echo "===> Verifying system PyTorch (should be CUDA build)..."
python3 - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| cuda_available:", torch.cuda.is_available())
    print("torch path:", torch.__file__)
except Exception as e:
    print("torch import failed:", e)
    raise SystemExit(1)
PY

echo "===> Installing pinned user-site packages (no torch)..."
pip3 install --user -U \
  "transformers==4.56.0" \
  "datasets==4.0.0" \
  "accelerate==1.10.1" \
  "deepspeed==0.17.5" \
  "wandb==0.21.3" \
  "tqdm==4.67.1" \
  "pyarrow==21.0.0" \
  "huggingface-hub==0.34.4" \
  "safetensors==0.6.2" \
  "tokenizers==0.22.0"

# Optional: for some builds that need it
# sudo apt-get update && sudo apt-get install -y libaio-dev

echo "===> Writing environment guard to skip TensorFlow/Keras..."
# (You can also do this per-notebook with `%env TRANSFORMERS_NO_TF=1`)
grep -qxF 'export TRANSFORMERS_NO_TF=1' ~/.bashrc || echo 'export TRANSFORMERS_NO_TF=1' >> ~/.bashrc

echo "===> Quick sanity table..."
python3 - <<'PY'
import importlib, os
pkgs = ["torch","torchvision","transformers","datasets","accelerate","deepspeed","wandb","numpy","pandas","pyarrow","safetensors","huggingface_hub","tokenizers","flash_attn"]
print(f"{'Pkg':<16} {'Version':<14} Location")
print("-"*90)
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "unknown")
        print(f"{p:<16} {v:<14} {os.path.dirname(getattr(m, '__file__', 'n/a'))}")
    except Exception as e:
        print(f"{p:<16} {'(missing)':<14} {e}")
try:
    import torch, torch.backends.cudnn as cudnn
    print("-"*90)
    print("CUDA available:", torch.cuda.is_available(), "| CUDA:", torch.version.cuda, "| cuDNN:", cudnn.version())
except Exception as e:
    print("Torch CUDA probe failed:", e)
PY

echo "===> Done."
