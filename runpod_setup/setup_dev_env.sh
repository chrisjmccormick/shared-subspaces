#!/usr/bin/env bash
#
# One-shot script to create a conda env + install requirements
# Usage:
#   ./scripts/setup_conda.sh [requirements_file]
#
# If no requirements file is provided, defaults to ./requirements.txt
#

set -euo pipefail

# Accept ToS for Anaconda channels
# Remove these lines (not supported on macOS)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV_NAME="shared-subspace"
PY_VER="3.11"
REQUIREMENTS_FILE="${1:-requirements.txt}"

# --- sanity checks -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 'conda' not found in PATH. Ensure Anaconda/Miniconda is installed and initialized (e.g., 'conda init')."
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "❌ Requirements file not found: $REQUIREMENTS_FILE"
  echo "   Provide a path, e.g.: ./scripts/setup_conda.sh path/to/requirements.txt"
  exit 1
fi

# --- create env if needed ----------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ℹ️  Conda env '$ENV_NAME' already exists; skipping creation."
else
  echo "📦 Creating conda env '${ENV_NAME}' (Python ${PY_VER}) ..."
  conda create -y -n "$ENV_NAME" "python=${PY_VER}"
fi

# --- install requirements ----------------------------------------------------
echo "⬆️  Upgrading pip in '${ENV_NAME}' ..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "📄 Installing dependencies from '${REQUIREMENTS_FILE}' ..."
conda run -n "$ENV_NAME" pip install -r "$REQUIREMENTS_FILE"

# --- done -------------------------------------------------------------------
echo
echo "✅ Environment ready."
echo "   • Activate:   conda activate ${ENV_NAME}"
echo "   • Python:     python --version"
echo "   • Installed:  pip list | wc -l"
