import os
import pandas as pd
from Bio.Align.Applications import ClustalOmegaCommandline

# Define directories
base_dir = "data/raw"
csv_dir = base_dir
msa_output_dir = "data/processed/msa"

# Ensure output directory
os.makedirs(msa_output_dir, exist_ok=True)

# CSV to MSA configuration
csv_files = {
    "train": os.path.join(csv_dir, "train_sequences.csv"),
    "val": os.path.join(csv_dir, "validation_sequences.csv"),
    "test": os.path.join(csv_dir, "test_sequences.csv"),
}


def csv_to_fasta(csv_path, fasta_path):
    df = pd.read_csv(csv_path)
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['target_id']}\n")
            f.write(f"{row['sequence']}\n")


def run_clustalo(input_fasta, output_aln):
    clustalomega_cline = ClustalOmegaCommandline(
        infile=input_fasta,
        outfile=output_aln,
        verbose=True,
        auto=True,
        force=True,
        outfmt="fasta",  # Changed from a3m to fasta (supported format)
    )
    stdout, stderr = clustalomega_cline()
    print(stdout)
    print(stderr)


for tag, csv_file in csv_files.items():
    # Check if the source CSV file exists
    if not os.path.exists(csv_file):
        print(f"Warning: Input CSV file not found: {csv_file}. Skipping...")
        continue

    print(f"Processing {csv_file}...")
    # Temporary fasta file for clustalo input
    temp_fasta_file = os.path.join(msa_output_dir, f"{tag}_temp_sequences.fasta")
    # Final output alignment file, named by tag (train/val/test)
    # This assumes one MSA per split, which might not be intended.
    # If MSAs should be per-sequence, the logic needs significant change.
    # For now, aligning all sequences in a split together.
    aln_file = os.path.join(msa_output_dir, f"{tag}_sequences.aln")  # Using .aln extension with fasta format

    csv_to_fasta(csv_file, temp_fasta_file)
    run_clustalo(temp_fasta_file, aln_file)
    # Clean up temporary fasta file
    os.remove(temp_fasta_file)


print("✅ MSA generation completed.")
