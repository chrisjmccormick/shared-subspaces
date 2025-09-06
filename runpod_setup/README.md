## Set up project to be run on runpod

1. Clone project repository and switch to remote branch with your changes
```
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Fetch all branches from remote
git fetch origin

# Check out the branch
git checkout <branch-name>

# (Optional) Set it to track the remote branch
git checkout -b <branch-name> origin/<branch-name>
```

2. Set up miniconda to create virtual environment
```
# Make the file executable
chmod +x runpod_setup/install_miniconda.sh

# Execute the file
./runpod_setup/install_miniconda.sh
```

3. Create conda environment and install required dependencies.
>NOTE: run `setup_dev_env.sh` from root project directory.
```
# Make the file executable
chmod +x runpod_setup/setup_dev_env.sh

# Execute the file
./runpod_setup/setup_dev_env.sh
```

4. BEFORE launching pre-training/fine-tuning runs on __runpod__, do the following:
- Set wandb token using
```
wandb login
huggingface-cli login
```
- set `WANDB_MODE` to `online` to ensure logs are sent to wandb.
```
export WANDB_MODE=online
```

5. ADD the following to the pretrain config in `best_mla.json` and `best_mla-o.json`
>NOTE: Ensure to remove them when done training from the config files.
```
"best_checkpoint": "checkpoints/mla_baseline/checkpoint-50000",
"run_name": "mla-o_baseline",
"run_id": "koko"
```