# Encoder Pretraining Experiments

This directory contains experiments for pretraining small BERT models using Hugging Face Transformers and Weights & Biases for tracking.

## Structure

- `data/` – scripts or pointers for obtaining pretraining datasets.
- `configs/` – JSON configuration files defining model and training hyperparameters.
- `models/` – custom model definitions (MLA, output subspace, decomposed MLP).
- `scripts/` – training and evaluation scripts.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch training with one of the configs. For example:
   ```bash
   python scripts/train.py --config configs/baseline.json
   ```

Available configs:

- `baseline.json` – standard BERT.
- `mla.json` – baseline with Multihead Latent Attention.
- `mla_output.json` – MLA with a shared output subspace.
- `mla_output_decompose.json` – MLA, output subspace and decomposed MLP.

Training metrics are logged to wandb under the project `encoder-pretrain`.
