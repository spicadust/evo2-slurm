#!/bin/bash
#SBATCH --job-name=evo2_embedding
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=4
# we run on the gpu partition and we allocate 1 A100 gpu
#SBATCH -p gpu --gres=gpu:A100:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=4:00:00

./mount_erda.sh
./setup_env.sh
uv run generate-embedding generate-embedding --input ~/erda/llm_matrix/1Jan2025_genomes.fa --output_dir ~/erda/embeddings/evo2/
./unmount_erda.sh
