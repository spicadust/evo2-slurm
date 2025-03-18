#!/bin/bash
#SBATCH --job-name=evo2_embedding
#SBATCH -p gpu --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
# We expect that our program should not run longer than 2 days
# Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00

export LD_LIBRARY_PATH="$HOME/lib64:$LD_LIBRARY_PATH"
export LD_PRELOAD="/opt/software/gcc/13.2.0/lib64/libstdc++.so.6"

./mount_erda.sh
./setup_env.sh
uv run generate-embedding --input ~/erda/llm_matrix/millard_phage_Jan2025_cleaned.fa --output_dir ~/erda/embeddings/ --batch_size 4 --model_name evo2_40b --layer_name blocks.44.mlp.l3
./unmount_erda.sh
