#!/bin/bash
#SBATCH --job-name=evo2_embedding
#SBATCH -p gpu --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
# We expect that our program should not run longer than 2 days
# Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00

./mount_erda.sh
./setup_env.sh
uv run generate-embedding generate-embedding --input ~/erda/llm_matrix/1Jan2025_genomes.fa --output_dir ~/erda/embeddings/evo2/
./unmount_erda.sh
