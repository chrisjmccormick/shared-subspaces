
The setup script will install Huggingface `transformers` and `datasets` and make sure everything is pinned to a specific working combination of versions.

Run the following in a terminal:

```bash
chmod +x setup_lambda.sh
./setup_lambda.sh
```

The script does the following installs, and also does some sanity checks and print outs.

```bash
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
```

**On NumPy**

Lambda Stack comes with NumPy 1.21.5 and a SciPy version that requires numpy <1.25. 
Those aren't included in the package install list, but I'm leaving that note here in case we ever need to make it explicit.