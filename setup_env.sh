#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# 0. Load modules
module load cuda/12.5
module load cudnn/9.7.1.26_cuda12
module load gcc/13.2.0

# 1. Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create uv virtual environment
echo "Installing dependencies..."

uv venv

uv pip install numpy torch==2.6.0 ninja psutil wheel setuptools pybind11 cmake

uv pip install -v transformer-engine[pytorch] --no-build-isolation

uv pip install -v https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.7.4.post1+cu124torch2.6-cp312-cp312-linux_x86_64.whl --no-build-isolation

cd evo2

uv pip install .

cd vortex

uv pip install -e .

cd ../../

uv pip install .

echo "Environment setup complete!"
