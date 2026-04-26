#!/bin/bash

# Brev setup script for the CI 2026 Hackathon Starter Kit.
# Runs automatically on first boot before participants access the instance.
# Requires NVIDIA driver >= 570 (ships with Brev CUDA 12.8 base images).
set -euo pipefail

REPO_DIR="/home/ubuntu/workspace"
DATA_DIR="/home/ubuntu/workspace/data"
ENV_NAME="ci26_starter_kit"

# ── 1. Conda ──────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[setup] Installing Miniconda..."
    curl -fsSL \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
fi
CONDA_BIN=$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")
eval "$("$CONDA_BIN" shell.bash hook)"

# ── 2. Conda environment ──────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[setup] Creating conda env '${ENV_NAME}'..."
    conda env create -f "$REPO_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

# ── 3. PyTorch with CUDA 12.8 ─────────────────────────────────────────────────
echo "[setup] Installing PyTorch (CUDA 12.8)..."
conda run -n "$ENV_NAME" uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ── 4. Remaining requirements (torch already installed above) ─────────────────
echo "[setup] Installing requirements..."
grep -vE "^torch(vision|audio)?($|[>=<! ])" "$REPO_DIR/requirements.txt" \
    | conda run -n "$ENV_NAME" uv pip install -r /dev/stdin

# ── 5. Install starter kit in editable mode ───────────────────────────────────
echo "[setup] Installing starter kit package..."
conda run -n "$ENV_NAME" pip install -e "$REPO_DIR/"

# ── 6. Training data from HuggingFace ─────────────────────────────────────────
echo "[setup] Downloading training data..."
mkdir -p "$DATA_DIR"
huggingface-cli download tobifinn/CI2026Hackathon \
    --repo-type dataset \
    --local-dir "$DATA_DIR/train_data"
find "$DATA_DIR/train_data" -name "*.zip" -exec unzip -o {} -d "$DATA_DIR/train_data" \; \
    -exec rm {} \;

echo "[setup] Done. Activate the environment with: conda activate ${ENV_NAME}"
