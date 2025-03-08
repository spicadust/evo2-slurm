import argparse
import os

import torch
from Bio import SeqIO
from tqdm import tqdm

from evo2 import Evo2


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate embeddings from sequences using Evo2 model"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument("--model_name", type=str, default="evo2_1b_base", help="Evo2 model name")
    parser.add_argument(
        "--layer_name",
        type=str,
        default="blocks.24.mlp.l3",
        help="Layer to extract embeddings from",
    )
    parser.add_argument("--prefix", type=str, default="evo2", help="Prefix for output files")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of sequences to process at once"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model_name} model...")
    evo2_model = Evo2(args.model_name)

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = list(SeqIO.parse(args.input, "fasta"))
    print(f"Loaded {len(sequences)} sequences")

    # Process all sequences in the FASTA file and extract embeddings

    # Create lists to store results and failed sequences
    results = []
    failed_sequences = []

    # Process sequences in batches
    print("Generating embeddings...")

    # If batch size is 1, process sequences one by one
    if args.batch_size == 1:
        for record in tqdm(sequences):
            try:
                # Extract sequence and header
                sequence = str(record.seq)
                header = record.description

                # Make sure GPU memory is cleared before processing
                torch.cuda.empty_cache()

                # Tokenize and get embeddings
                input_ids = (
                    torch.tensor(
                        evo2_model.tokenizer.tokenize(sequence),
                        dtype=torch.int,
                    )
                    .unsqueeze(0)
                    .to("cuda:0")
                )

                # Get embeddings with explicit dtype to avoid BFloat16 issues
                with torch.amp.autocast("cuda", enabled=False):
                    outputs, embeddings = evo2_model(
                        input_ids, return_embeddings=True, layer_names=[args.layer_name]
                    )

                # Extract the embeddings tensor and ensure it's float32
                embedding_tensor = embeddings[args.layer_name].to(torch.float32)

                # Average over the sequence length dimension to get a 1920-dim vector
                # Shape goes from [1, n, 1920] to [1, 1920] to [1920]
                avg_embedding = embedding_tensor.mean(dim=1).squeeze().cpu().numpy()

                # Store results
                results.append({"header": header, "embedding": avg_embedding})

                # Clear GPU cache to free memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing sequence {header}: {e}")
                # Record the failed sequence
                failed_sequences.append(header)
                # Force GPU memory cleanup
                torch.cuda.empty_cache()
                continue
    else:
        # Process sequences in batches
        for i in tqdm(range(0, len(sequences), args.batch_size)):
            batch = sequences[i : i + args.batch_size]
            try:
                # Extract sequences and headers
                batch_sequences = [str(record.seq) for record in batch]
                batch_headers = [record.description for record in batch]

                # Make sure GPU memory is cleared before processing
                torch.cuda.empty_cache()

                # Tokenize and get embeddings
                input_ids = torch.stack(
                    [
                        torch.tensor(
                            evo2_model.tokenizer.tokenize(seq),
                            dtype=torch.int,
                        ).to("cuda:0")
                        for seq in batch_sequences
                    ]
                )

                # Get embeddings with explicit dtype to avoid BFloat16 issues
                with torch.amp.autocast("cuda", enabled=False):
                    outputs, embeddings = evo2_model(
                        input_ids, return_embeddings=True, layer_names=[args.layer_name]
                    )

                # Extract the embeddings tensor and ensure it's float32
                embedding_tensor = embeddings[args.layer_name].to(torch.float32)

                # Average over the sequence length dimension to get 1920-dim vectors
                # Shape goes from [batch_size, n, 1920] to [batch_size, 1920]
                avg_embeddings = embedding_tensor.mean(dim=1).cpu().numpy()

                # Store results
                for header, embedding in zip(batch_headers, avg_embeddings):
                    results.append({"header": header, "embedding": embedding})

                # Clear GPU cache to free memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch starting at sequence {i}: {e}")
                # Record the failed sequences
                failed_sequences.extend(batch_headers)
                # Force GPU memory cleanup
                torch.cuda.empty_cache()
                continue
