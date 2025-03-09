#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# 0. Load modules
module load gcc/13.2.0
module load cuda/12.5
module load cudnn/9.7.1.26_cuda12

# Create a lib directory in your home if it doesn't exist
mkdir -p ~/lib64

# Create a symbolic link to the newer libstdc++
ln -sf /opt/software/gcc/13.2.0/lib64/libstdc++.so.6 ~/lib64/libstdc++.so.6

# Add your local lib directory to the front of LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$HOME/lib64:$LD_LIBRARY_PATH"

# Also keep LD_PRELOAD just in case
export LD_PRELOAD="/opt/software/gcc/13.2.0/lib64/libstdc++.so.6"

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LD_PRELOAD: $LD_PRELOAD"

# 1. Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create uv virtual environment
echo "Installing dependencies..."

uv venv

uv pip install ninja cmake pybind11 numpy psutil

cd evo2

uv pip install .

cd vortex

uv pip install -e .

uv pip install -v transformer-engine[pytorch] --no-build-isolation

cd vortex/ops/attn

export MAX_JOBS=32
uv pip install -v -e . --no-build-isolation

cd ../../../../../

uv pip install .

echo "Environment setup complete!"
