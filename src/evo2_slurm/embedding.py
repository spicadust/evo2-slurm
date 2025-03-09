import click
import os

import torch
from Bio import SeqIO
from tqdm import tqdm

from evo2 import Evo2


@click.command()
@click.option(
    "--input",
    type=str,
    required=True,
    help="Path to input FASTA file",
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save output files",
)
@click.option(
    "--model_name",
    type=str,
    default="evo2_7b",
    help="Evo2 model name",
)
@click.option(
    "--layer_name",
    type=str,
    default="blocks.28.mlp.l3",
    help="Layer to extract embeddings from",
)
@click.option(
    "--prefix",
    type=str,
    default="evo2",
    help="Prefix for output files",
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="Number of sequences to process at once",
)
def generate(input, output_dir, model_name, layer_name, prefix, batch_size):
    """Generate embeddings from sequences using Evo2 model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading {model_name} model...")
    evo2_model = Evo2(model_name)

    # Load sequences
    print(f"Loading sequences from {input}...")
    sequences = list(SeqIO.parse(input, "fasta"))
    print(f"Loaded {len(sequences)} sequences")

    # Process all sequences in the FASTA file and extract embeddings

    # Create lists to store results and failed sequences
    results = []
    failed_sequences = []

    # Process sequences in batches
    print("Generating embeddings...")

    # If batch size is 1, process sequences one by one
    if batch_size == 1:
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
                        input_ids, return_embeddings=True, layer_names=[layer_name]
                    )

                # Extract the embeddings tensor and ensure it's float32
                embedding_tensor = embeddings[layer_name].to(torch.float32)

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
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i : i + batch_size]
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
                        input_ids, return_embeddings=True, layer_names=[layer_name]
                    )

                # Extract the embeddings tensor and ensure it's float32
                embedding_tensor = embeddings[layer_name].to(torch.float32)

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
