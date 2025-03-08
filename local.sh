# Make sure we use gcc/g++ instead of clang
export CC=gcc
export CXX=g++

# Set CUDA paths explicitly from virtual environment and NixOS
CUDA_RUNTIME_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime"
TRITON_NVIDIA_PATH="$PWD/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include"
CUSPARSE_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cusparse/include"
CUDNN_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/include"
NIX_CUDA_PATH="/nix/store/63i9isjm6wxwl9287ph0jrbbzzjz4rlk-cuda-merged-12.4/include"

# Set the CPATH for include directories
export CPATH="${CUDA_RUNTIME_PATH}/include:${TRITON_NVIDIA_PATH}:${CUSPARSE_PATH}:${CUDNN_PATH}:${NIX_CUDA_PATH}:${CPATH:-}"

# Set library paths
export LIBRARY_PATH="${CUDA_RUNTIME_PATH}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_PATH}/lib:${LD_LIBRARY_PATH:-}"

uv venv

uv pip install numpy torch==2.6.0 ninja

uv pip install transformer-engine[pytorch] --no-build-isolation

uv pip install flash-attn --no-build-isolation

cd evo2

uv pip install .

cd vortex

uv pip install -e .

cd ../../

uv pip install .
