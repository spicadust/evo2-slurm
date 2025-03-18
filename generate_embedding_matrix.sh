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
uv run generate-embedding --input ~/erda/CCRP/MATRIX/Analyses/Microbiomics/Ref_Genomics/phages/vOTUs_MATRIX_01_2024.fasta --output_dir ~/erda/embeddings_matrix/ --batch_size 4 --model_name evo2_40b --layer_name blocks.44.mlp.l3
./unmount_erda.sh
