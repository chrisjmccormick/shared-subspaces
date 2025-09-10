#!/bin/bash

# Step 1: Download Miniconda installer
echo "Downloading Miniconda installer..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Step 2: Run the installer silently
echo "Installing Miniconda..."
bash miniconda.sh -b -p $HOME/miniconda

# Step 3: Initialize conda for bash
echo "Initializing conda for bash..."
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
$HOME/miniconda/bin/conda init bash

echo "✅ Miniconda installed. Restart your shell or run: source ~/.bashrc"
