"""_summary_
This script processes raw RNA sequence and structure data into a format suitable for training a neural network.
Source: https://www.kaggle.com/code/olaflundstrom/stanford-rna-3d-folding-competition-notebook#3.-Data-Preprocessing
"""

import pickle
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp

# Add RNA folding library for BPPM generation
try:
    import RNA  # ViennaRNA package
except ImportError:
    print("Warning: ViennaRNA package not installed. BPPM features will be disabled.")
    RNA = None

from src.utils import log_message, check_jax_device

# 🔹 Mapping RNA nucleotides to integers
nucleotide_map = {"A": 1, "C": 2, "G": 3, "U": 4}


def encode_sequence(seq):
    """Encodes an RNA sequence into a JAX array of integers."""
    return jnp.array([nucleotide_map.get(ch, 0) for ch in seq], dtype=jnp.int32)


def process_labels(labels_df):
    """
    Processes RNA structure labels into a dictionary mapping `target_id` to coordinates.
    Optimized using pandas groupby and vectorized NumPy operations.
    """
    # 🔹 Efficiently split "ID" column into "target_id" and "resid"
    ids_split = labels_df["ID"].str.rsplit("_", n=1, expand=True)
    labels_df["target_id"] = ids_split[0]
    labels_df["resid"] = ids_split[1].astype(int)  # Convert residue number to int

    # 🔹 Convert coordinates to NumPy array for batch processing
    coords = labels_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=jnp.float32)

    # 🔹 Group by `target_id` and stack residues
    label_dict = {}
    for target_id, group in labels_df.groupby("target_id"):
        # Sort by `resid` and stack the coordinates
        sorted_indices = jnp.argsort(group["resid"].to_numpy())
        sorted_coords = coords[group.index[sorted_indices]]  # Faster indexing

        # Convert to JAX array and store
        label_dict[target_id] = jnp.array(sorted_coords, dtype=jnp.float32)

    return label_dict


def create_dataset(sequences_df, labels_dict):
    """
    Creates a dataset: X (encoded RNA sequences), y (3D coordinates), target_ids.
    """
    X, y, target_ids = [], [], []

    for _, row in sequences_df.iterrows():
        tid = row["target_id"]
        if tid in labels_dict:
            X.append(encode_sequence(row["sequence"]))
            y.append(labels_dict[tid])
            target_ids.append(tid)

    return X, y, target_ids


def pad_sequences_jax(sequences, max_len):
    """
    Pads sequences using JAX operations.
    """
    padded_sequences = jnp.zeros((len(sequences), max_len), dtype=jnp.int32)

    for i, seq in enumerate(sequences):
        padded_sequences = padded_sequences.at[i, : len(seq)].set(seq)

    return padded_sequences


def pad_coordinates_jax(coord_array, max_len):
    """
    Pads coordinate arrays to the max sequence length using JAX.
    """
    L = coord_array.shape[0]
    if L < max_len:
        pad_width = ((0, max_len - L), (0, 0))  # Only pad along first axis
        return jnp.pad(coord_array, pad_width, mode="constant", constant_values=0)
    return coord_array


def load_msa_data(target_id, msa_dir="data/raw/msa"):
    """
    Loads MSA data for a given target ID.

    Args:
        target_id: The RNA target identifier
        msa_dir: Directory containing MSA files

    Returns:
        Dictionary with MSA features including conservation scores
    """
    msa_file = os.path.join(msa_dir, f"{target_id}.a2m")

    # Check if MSA file exists
    if not os.path.exists(msa_file):
        # Return empty features if no MSA data
        return None

    # Load MSA file - typically in A2M or FASTA format
    with open(msa_file, 'r') as f:
        msa_lines = f.readlines()

    # Parse MSA data (simplified version)
    sequences = []
    current_seq = ""

    for line in msa_lines:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
        else:
            current_seq += line

    if current_seq:
        sequences.append(current_seq)

    if not sequences:
        return None

    # Calculate conservation scores
    seq_len = len(sequences[0])
    conservation = np.zeros(seq_len, dtype=np.float32)

    for pos in range(seq_len):
        # Count frequencies of each nucleotide at this position
        counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0, '-': 0}
        for seq in sequences:
            if pos < len(seq):
                nt = seq[pos].upper()
                if nt in counts:
                    counts[nt] += 1

        # Calculate conservation as frequency of most common nucleotide
        total = sum(counts.values())
        if total > 0:
            max_freq = max(counts.values()) / total
            conservation[pos] = max_freq

    return {
        'conservation': jnp.array(conservation, dtype=jnp.float32),
        'num_sequences': len(sequences)
    }

def generate_bppm(sequence):
    """
    Generates Base Pair Probability Matrix for an RNA sequence using ViennaRNA.

    Args:
        sequence: RNA sequence

    Returns:
        BPPM as a square matrix
    """
    if RNA is None:
        # Return zeros if ViennaRNA is not available
        seq_len = len(sequence)
        return jnp.zeros((seq_len, seq_len), dtype=jnp.float32)

    # Create fold compound
    fc = RNA.fold_compound(sequence)

    # Compute partition function
    fc.pf()

    # Get base pair probabilities
    seq_len = len(sequence)
    bppm = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Access the base pair probability matrix directly
    probs = fc.bpp()  # Retrieve the probability matrix
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            prob = probs[i + 1][j + 1]  # Use 1-based indexing
            bppm[i, j] = prob
            bppm[j, i] = prob  # Make it symmetric

    return jnp.array(bppm, dtype=jnp.float32)

def process_sequences_with_features(sequences_df, labels_dict, msa_dir="data/raw/msa"):
    """
    Creates a dataset with additional features: X (encoded RNA sequences),
    MSA conservation, BPPMs, y (3D coordinates), target_ids.
    """
    X, msas, bppms, y, target_ids = [], [], [], [], []

    for _, row in sequences_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        if tid in labels_dict:
            # Basic sequence encoding
            X.append(encode_sequence(seq))

            # MSA features
            msa_features = load_msa_data(tid, msa_dir)
            if msa_features is None:
                # If no MSA data, use zeros
                msa_features = {
                    'conservation': jnp.zeros(len(seq), dtype=jnp.float32),
                    'num_sequences': 0
                }
            msas.append(msa_features)

            # BPPM features
            bppm = generate_bppm(seq)
            bppms.append(bppm)

            # Labels
            y.append(labels_dict[tid])
            target_ids.append(tid)

    return X, msas, bppms, y, target_ids

def pad_msa_features(msa_list, max_len):
    """
    Pads MSA conservation features to the max sequence length.
    """
    padded_features = []

    for msa in msa_list:
        conservation = msa['conservation']
        L = conservation.shape[0]

        if L < max_len:
            # Pad the conservation scores
            padded_conservation = jnp.pad(
                conservation,
                (0, max_len - L),
                mode="constant",
                constant_values=0
            )
        else:
            padded_conservation = conservation[:max_len]

        padded_features.append({
            'conservation': padded_conservation,
            'num_sequences': msa['num_sequences']
        })

    return padded_features

def pad_bppm(bppm, max_len):
    """
    Pads BPPM to the max sequence length.
    """
    L = bppm.shape[0]
    if L < max_len:
        # Pad the BPPM
        padded_bppm = jnp.pad(
            bppm,
            ((0, max_len - L), (0, max_len - L)),
            mode="constant",
            constant_values=0
        )
        return padded_bppm
    return bppm[:max_len, :max_len]


if __name__ == "__main__":
    log_message("🧬 Data processing started!")
    check_jax_device()
    # 🔹 Load data
    log_message("Loading raw data")
    train_sequences = pd.read_csv("data/raw/train_sequences.csv")
    valid_sequences = pd.read_csv("data/raw/validation_sequences.csv")
    train_labels = pd.read_csv("data/raw/train_labels.csv")
    valid_labels = pd.read_csv("data/raw/validation_labels.csv")

    # 🔹 Fill NaN values
    train_labels.fillna(0, inplace=True)
    valid_labels.fillna(0, inplace=True)

    # 🔹 Process labels
    log_message("Processing labels")
    train_labels_dict = process_labels(train_labels)
    valid_labels_dict = process_labels(valid_labels)

    # 🔹 Create datasets with enhanced features
    log_message("Creating datasets with MSA and BPPM features")
    X_train, msas_train, bppms_train, y_train, train_ids = process_sequences_with_features(
        train_sequences, train_labels_dict
    )
    X_valid, msas_valid, bppms_valid, y_valid, valid_ids = process_sequences_with_features(
        valid_sequences, valid_labels_dict
    )

    # 🔹 Determine max sequence length
    max_len = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_valid))

    # 🔹 Pad sequences using JAX
    log_message(f"Padding sequences to max length {max_len}")
    X_train_pad = pad_sequences_jax(X_train, max_len)
    X_valid_pad = pad_sequences_jax(X_valid, max_len)

    # 🔹 Pad MSA features
    log_message("Padding MSA features")
    msas_train_pad = pad_msa_features(msas_train, max_len)
    msas_valid_pad = pad_msa_features(msas_valid, max_len)

    # 🔹 Pad BPPM features
    log_message("Padding BPPM matrices")
    bppms_train_pad = jnp.array([pad_bppm(bppm, max_len) for bppm in bppms_train])
    bppms_valid_pad = jnp.array([pad_bppm(bppm, max_len) for bppm in bppms_valid])

    # 🔹 Pad labels using JAX
    log_message("Padding coordinates")
    y_train_pad = jnp.array([pad_coordinates_jax(arr, max_len) for arr in y_train])
    y_valid_pad = jnp.array([pad_coordinates_jax(arr, max_len) for arr in y_valid])

    # 🔹 Save processed data
    log_message("Saving processed data")
    with open("data/processed/processed_data_msa_bppm.pkl", "wb") as f:
        pickle.dump(
            (
                X_train_pad, y_train_pad, msas_train_pad, bppms_train_pad,
                X_valid_pad, y_valid_pad, msas_valid_pad, bppms_valid_pad,
                max_len
            ),
            f
        )

    log_message("✅ Data processing complete with MSA and BPPM features! \n")
