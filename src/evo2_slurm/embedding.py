import click
import os
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from datetime import datetime
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
    # Create a unique subfolder for the run
    output_dir = os.path.join(output_dir, f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create output directory for batch files
    batch_dir = os.path.join(output_dir, f"{prefix}_batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Initialize files
    headers_path = os.path.join(output_dir, f"{prefix}_headers.npy")
    failed_path = os.path.join(output_dir, f"{prefix}_failed.txt")
    final_embeddings_path = os.path.join(output_dir, f"{prefix}_embeddings.npy")

    # Create list to store all headers
    all_headers = []

    # Load model
    print(f"Loading {model_name} model...")
    evo2_model = Evo2(model_name)

    # Load sequences
    print(f"Loading sequences from {input}...")
    sequences = list(SeqIO.parse(input, "fasta"))
    print(f"Loaded {len(sequences)} sequences")

    # Process sequences in batches
    print("Generating embeddings...")

    # If batch size is 1, process sequences one by one
    if batch_size == 1:
        for idx, record in enumerate(tqdm(sequences)):
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

                # Save results
                batch_embeddings_path = os.path.join(batch_dir, f"batch_{idx:06d}.npy")
                np.save(batch_embeddings_path, avg_embedding)
                
                # Store header
                all_headers.append(header)

                # Clear GPU cache to free memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing sequence {header}: {e}")
                with open(failed_path, 'a') as f:
                    f.write(f"{header}\n")
                torch.cuda.empty_cache()
                continue
    else:
        # Process sequences in batches
        for batch_idx, i in enumerate(tqdm(range(0, len(sequences), batch_size))):
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

                # Save results
                batch_embeddings_path = os.path.join(batch_dir, f"batch_{batch_idx:06d}.npy")
                np.save(batch_embeddings_path, avg_embeddings)
                
                # Store headers
                all_headers.extend(batch_headers)

                # Clear GPU cache to free memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch starting at sequence {i}: {e}")
                with open(failed_path, 'a') as f:
                    for header in batch_headers:
                        f.write(f"{header}\n")
                torch.cuda.empty_cache()
                continue

    # Save headers at the end
    if all_headers:
        np.save(headers_path, np.array(all_headers))

    # Combine all batch files into final embeddings file
    print("Combining batch files into final embeddings file...")
    batch_files = sorted(os.listdir(batch_dir))
    if batch_files:
        embeddings_list = []
        for batch_file in tqdm(batch_files):
            batch_path = os.path.join(batch_dir, batch_file)
            embeddings_list.append(np.load(batch_path))
        final_embeddings = np.vstack(embeddings_list)
        np.save(final_embeddings_path, final_embeddings)
        
        # Optionally, remove batch directory after successful combination
        import shutil
        shutil.rmtree(batch_dir)
        
        print(f"Final results saved to:")
        print(f"- Embeddings: {final_embeddings_path}")
        print(f"- Headers: {headers_path}")
        if os.path.exists(failed_path):
            print(f"- Failed sequences: {failed_path}")


if __name__ == "__main__":
    generate()