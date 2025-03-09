"""_summary_
This script processes raw RNA sequence and structure data into a format suitable for training a neural network.
Source: https://www.kaggle.com/code/olaflundstrom/stanford-rna-3d-folding-competition-notebook#3.-Data-Preprocessing
"""
# process.py
import pickle

import pandas as pd

import jax.numpy as jnp

from src.utils import log_message, check_jax_device

# 🔹 Mapping RNA nucleotides to integers
nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4}

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
        tid = row['target_id']
        if tid in labels_dict:
            X.append(encode_sequence(row['sequence']))
            y.append(labels_dict[tid])
            target_ids.append(tid)

    return X, y, target_ids

def pad_sequences_jax(sequences, max_len):
    """
    Pads sequences using JAX operations.
    """
    padded_sequences = jnp.zeros((len(sequences), max_len), dtype=jnp.int32)

    for i, seq in enumerate(sequences):
        padded_sequences = padded_sequences.at[i, :len(seq)].set(seq)

    return padded_sequences

def pad_coordinates_jax(coord_array, max_len):
    """
    Pads coordinate arrays to the max sequence length using JAX.
    """
    L = coord_array.shape[0]
    if L < max_len:
        pad_width = ((0, max_len - L), (0, 0))  # Only pad along first axis
        return jnp.pad(coord_array, pad_width, mode='constant', constant_values=0)
    return coord_array

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

    # 🔹 Create datasets
    log_message("Creating datasets")
    X_train, y_train, train_ids = create_dataset(train_sequences, train_labels_dict)
    X_valid, y_valid, valid_ids = create_dataset(valid_sequences, valid_labels_dict)

    # 🔹 Determine max sequence length
    max_len = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_valid))

    # 🔹 Pad sequences using JAX
    log_message("Determining max sequence length")
    X_train_pad = pad_sequences_jax(X_train, max_len)
    X_valid_pad = pad_sequences_jax(X_valid, max_len)

    # 🔹 Pad labels using JAX
    log_message("Padding coordinates")
    y_train_pad = jnp.array([pad_coordinates_jax(arr, max_len) for arr in y_train])
    y_valid_pad = jnp.array([pad_coordinates_jax(arr, max_len) for arr in y_valid])

    # 🔹 Save processed data
    log_message("Saving processed data")
    with open("data/processed/processed_data.pkl", "wb") as f:
        pickle.dump((X_train_pad, y_train_pad, X_valid_pad, y_valid_pad, max_len), f)

    log_message("✅ Data processing complete! \n")
