#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# 1. Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create uv virtual environment
echo "Installing dependencies..."
uv venv
uv pip install .

# 3. Move to evo2 directory and install package
echo "Installing evo2 package..."
cd ./evo2
uv pip install .

cd vortex
make setup-full

# Return to original directory
cd ../../

echo "Setup complete!"
