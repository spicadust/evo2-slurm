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
uv pip install --system .

# 3. Move to evo2 directory and install package
echo "Installing evo2 package..."
cd ./evo2
uv pip install --system .

# Return to original directory
cd ../

echo "Setup complete!"
