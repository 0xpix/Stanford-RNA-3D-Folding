import os
import shutil
import pickle
import subprocess
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

from src.utils.utils import log_message


def process_labels(labels_df: pd.DataFrame) -> Dict[str, jnp.ndarray]:
    """
    Processes RNA structure labels into a dictionary mapping `target_id` to coordinates.
    Optimized using pandas operations and JAX's efficient data conversion.

    Args:
        labels_df: DataFrame containing RNA structure labels

    Returns:
        Dictionary mapping target_id to JAX arrays of coordinates
    """
    log_message("Processing RNA structure labels")

    # Create a working copy and extract ID components efficiently
    df = labels_df.copy()
    id_parts = df["ID"].str.rsplit("_", n=1, expand=True)
    df["target_id"] = id_parts[0]
    df["resid"] = pd.to_numeric(id_parts[1], downcast="integer")  # Faster than astype

    # Sort once by both target_id and resid for efficiency
    df.sort_values(["target_id", "resid"], inplace=True)

    # Build dictionary using direct pandas groupby (more efficient)
    log_message("Building label dictionary")
    groups = df.groupby("target_id")

    # Pre-allocate dictionary for better memory efficiency
    target_ids = groups.groups.keys()
    log_message(f"Found {len(target_ids)} unique target IDs")

    # Process all groups at once using pandas operations
    label_dict = {}
    for target_id, group in groups:
        # Convert coordinates to JAX array
        coords = group[["x_1", "y_1", "z_1"]].values
        label_dict[target_id] = jnp.array(coords, dtype=jnp.float32)

    log_message(f"Completed processing {len(label_dict)} label entries")
    return label_dict


def compute_rna_secondary_structure(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RNA secondary structure using RNAfold from the ViennaRNA package.
    Provides helpful error messages if RNAfold is not installed.

    Args:
        sequences_df: DataFrame containing RNA sequences

    Returns:
        DataFrame with added 'structure' column
    """
    log_message("Starting RNA secondary structure computation")

    # Check if RNAfold is installed
    if not shutil.which("RNAfold"):
        log_message(
            "ERROR: RNAfold not found in PATH. Please install the ViennaRNA package:",
            "ERROR",
        )
        log_message("  - For Ubuntu/Debian: sudo apt-get install viennarna", "ERROR")
        log_message("  - For CentOS/RHEL: sudo yum install viennarna", "ERROR")
        log_message("  - For macOS: brew install viennarna", "ERROR")
        log_message("  - Or visit: https://www.tbi.univie.ac.at/RNA/", "ERROR")

        # Fallback: Return DataFrame with empty structures
        log_message("Returning DataFrame with empty structure column", "WARNING")
        if "structure" not in sequences_df.columns:
            sequences_df["structure"] = None
        return sequences_df

    # Ensure the DataFrame has a 'sequence' column
    if "sequence" not in sequences_df.columns:
        raise ValueError("The DataFrame must contain a 'sequence' column.")

    # Add a new column for the secondary structure
    sequences_df["structure"] = None

    log_message(
        f"Computing RNA secondary structures with RNAfold for {len(sequences_df)} sequences"
    )

    # Iterate over each row in the DataFrame
    success_count = 0
    error_count = 0
    for index, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):
        sequence = row["sequence"]

        # Call RNAfold to compute the secondary structure
        try:
            # Run RNAfold and capture the output
            result = subprocess.run(
                ["RNAfold", "--noPS"],  # --noPS prevents generating PostScript files
                input=sequence.encode(),  # Pass the sequence as input
                capture_output=True,
                check=True,
                timeout=60,  # Add timeout to prevent hanging
            )

            # Parse the output to extract the dot-bracket structure
            output = result.stdout.decode().strip()
            structure = output.split("\n")[1].split()[0]  # Extract structure

            # Store the structure in the DataFrame
            sequences_df.at[index, "structure"] = structure
            success_count += 1

        except subprocess.CalledProcessError as e:
            log_message(f"Error processing sequence at index {index}: {e}", "WARNING")
            sequences_df.at[index, "structure"] = None
            error_count += 1
        except subprocess.TimeoutExpired:
            log_message(f"Timeout for sequence at index {index}", "WARNING")
            sequences_df.at[index, "structure"] = None
            error_count += 1

    log_message(
        f"RNA structure computation complete: {success_count} successful, {error_count} failed"
    )
    return sequences_df


def pad_sequences_and_structures(
    sequences_df: pd.DataFrame, pad_char: str = "X"
) -> pd.DataFrame:
    """
    Pad RNA sequences and structures to uniform length.

    Args:
        sequences_df: DataFrame with 'sequence' and 'structure' columns
        pad_char: Character to use for padding

    Returns:
        DataFrame with added 'padded_sequence' and 'padded_structure' columns
    """
    log_message("Padding sequences and structures to uniform length")

    # Ensure the DataFrame has required columns
    if (
        "sequence" not in sequences_df.columns
        or "structure" not in sequences_df.columns
    ):
        raise ValueError(
            "The DataFrame must contain 'sequence' and 'structure' columns."
        )

    # Find the length of the longest sequence and structure
    max_seq_len = sequences_df["sequence"].str.len().max()
    max_struct_len = sequences_df["structure"].str.len().max()

    log_message(
        f"Maximum sequence length: {max_seq_len}, maximum structure length: {max_struct_len}"
    )

    # Padded sequences and structures (vectorized operations for efficiency)
    sequences_df["padded_sequence"] = sequences_df["sequence"].str.ljust(
        max_seq_len, pad_char
    )
    sequences_df["padded_structure"] = sequences_df["structure"].str.ljust(
        max_struct_len, pad_char
    )

    log_message("Padding complete")
    return sequences_df


@jax.jit
def encode_sequence(seq: str, mapping: Dict[str, int]) -> jnp.ndarray:
    """
    Encodes an RNA sequence into a JAX array of integers using character mapping.

    Args:
        seq: String sequence to encode
        mapping: Dictionary mapping characters to integers

    Returns:
        JAX array of encoded integers
    """
    # Default to 0 for unknown characters
    return jnp.array([mapping.get(ch, 0) for ch in seq], dtype=jnp.int32)


# Vectorized version for batch processing
def batch_encode_sequences(
    sequences: List[str], mapping: Dict[str, int]
) -> List[jnp.ndarray]:
    """
    Encode multiple sequences efficiently.

    Args:
        sequences: List of sequences to encode
        mapping: Dictionary mapping characters to integers

    Returns:
        List of encoded sequences as JAX arrays
    """
    log_message(f"Encoding {len(sequences)} sequences")
    encode_fn = partial(encode_sequence, mapping=mapping)
    return [encode_fn(seq) for seq in sequences]


def pad_coordinates_jax(coord_array: jnp.ndarray, max_len: int) -> jnp.ndarray:
    """
    Pad coordinate arrays to a uniform length using JAX operations.

    Args:
        coord_array: JAX array of coordinates
        max_len: Target length to pad to

    Returns:
        Padded JAX array
    """
    L = coord_array.shape[0]
    pad_width = ((0, max(0, max_len - L)), (0, 0))
    return jnp.pad(coord_array, pad_width, mode="constant", constant_values=0)


def create_dataset(
    train_df: pd.DataFrame, labels_dict: Dict[str, jnp.ndarray]
) -> Tuple[List, List, List]:
    """
    Create dataset from processed sequences and labels.

    Args:
        train_df: DataFrame with processed sequences and features
        labels_dict: Dictionary mapping target_ids to coordinate arrays

    Returns:
        Tuple of (features, labels, target_ids)
    """
    log_message("Creating dataset from processed sequences and labels")
    X, y, target_ids = [], [], []

    # Find the maximum length of padded sequences
    max_len = train_df["padded_sequence"].str.len().max()
    log_message(f"Maximum sequence length: {max_len}")

    valid_targets = 0
    skipped_targets = 0

    # Process in batches for better efficiency
    for _, row in tqdm(
        train_df.iterrows(), total=len(train_df), desc="Processing entries"
    ):
        tid = row["target_id"]
        if tid in labels_dict:
            # Use the precomputed combined_features
            X.append(row["combined_features"])

            # Get coordinates and pad (without using JIT in this loop)
            coord_array = labels_dict[tid]
            L = coord_array.shape[0]
            pad_width = ((0, max(0, max_len - L)), (0, 0))
            padded_coords = jnp.pad(
                coord_array, pad_width, mode="constant", constant_values=0
            )

            y.append(padded_coords)
            target_ids.append(tid)
            valid_targets += 1
        else:
            skipped_targets += 1

    log_message(
        f"Dataset creation complete: {valid_targets} valid entries, {skipped_targets} skipped"
    )
    return X, y, target_ids


def preprocess_data(
    sequences_df: pd.DataFrame, output_file: str = None
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for RNA sequences.

    Args:
        sequences_df: DataFrame with RNA sequences
        output_file: Optional file path to save preprocessed data

    Returns:
        Preprocessed DataFrame
    """
    log_message("Starting full data preprocessing pipeline")

    # Compute secondary structures
    sequences_df = compute_rna_secondary_structure(sequences_df)

    # Pad sequences and structures
    sequences_df = pad_sequences_and_structures(sequences_df)

    # Define mappings
    nucleotide_map = {"A": 1, "C": 2, "G": 3, "U": 4, "X": 0}
    structure_map = {".": 5, "(": 6, ")": 7, "X": 0}

    log_message("Encoding sequences and structures")

    # Encode sequences and structures (use vectorized operations for speed)
    encoded_seqs = batch_encode_sequences(
        sequences_df["padded_sequence"].tolist(), nucleotide_map
    )
    encoded_structs = batch_encode_sequences(
        sequences_df["padded_structure"].tolist(), structure_map
    )

    # Set the encoded values
    sequences_df["encoded_sequence"] = encoded_seqs
    sequences_df["encoded_structure"] = encoded_structs

    # Combine encoded sequences and structures
    log_message("Combining features")
    sequences_df["combined_features"] = [
        jnp.stack([seq, struct], axis=-1)
        for seq, struct in zip(encoded_seqs, encoded_structs)
    ]

    # Save if output file is specified
    if output_file:
        log_message(f"Saving preprocessed data to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(sequences_df, f)

    log_message("Preprocessing complete")
    return sequences_df


def load_data(features, y):
    # ---------- Data Preparation ---------
    # Extract combined_features and target 3D coordinates
    X = np.array(features).astype(np.float32)
    y = np.array(y, dtype=np.float32)

    # Check for NaN values in the data
    nan_indices = np.isnan(y).any(axis=(1, 2))
    if np.any(nan_indices):
        log_message(
            f"Warning: Found {np.sum(nan_indices)} entries with NaN values in coordinates",
            "WARNING",
        )
        # Remove NaN entries
        X = X[~nan_indices]
        y = y[~nan_indices]
        log_message(f"Removed NaN entries, {len(X)} samples remaining", "INFO")

    if len(X) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Split data into training and evaluation sets
    train_ratio = 0.8
    num_train = int(train_ratio * len(X))
    X_train, X_eval = X[:num_train], X[num_train:]
    y_train, y_eval = y[:num_train], y[num_train:]

    # Normalize 3D coordinates safely
    y_max_abs = np.max(np.abs(y_train))

    # Check if normalization factor is valid
    if np.isnan(y_max_abs) or y_max_abs == 0:
        log_message(
            "Cannot normalize coordinates properly: using default scale factor",
            "WARNING",
        )
        # Use a default scale to avoid division by zero/nan
        y_max_abs = 1.0

    y_train = y_train / y_max_abs
    y_eval = y_eval / y_max_abs
    print(f"Normalized coordinates with scale factor: {y_max_abs}")

    # Convert data to JAX arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_eval = jnp.array(X_eval)
    y_eval = jnp.array(y_eval)

    return X_train, y_train, X_eval, y_eval


if __name__ == "__main__":
    # Process labels
    log_message("Loading labels data")
    labels_df = pd.read_csv("data/raw/train_labels.csv")
    label_dict = process_labels(labels_df)

    # Process sequences with preprocessing pipeline
    preprocessed_data_file = "data/processed/preprocessed_data.pkl"

    if os.path.exists(preprocessed_data_file):
        log_message(f"Loading preprocessed data from {preprocessed_data_file}")
        with open(preprocessed_data_file, "rb") as f:
            sequences_df = pickle.load(f)
    else:
        log_message("Preprocessed data not found, starting preprocessing")
        sequences_df = pd.read_csv("data/raw/train_sequences.csv")
        sequences_df = preprocess_data(sequences_df, preprocessed_data_file)

    # Create dataset
    log_message("Creating final dataset")
    X, y, target_ids = create_dataset(sequences_df, label_dict)
    log_message(f"Final dataset created with {len(X)} samples")
    X = sequences_df["combined_features"].tolist()
    X_train, y_train, X_eval, y_eval = load_data(X, y)

    # save data set as pickle file
    with open("data/processed/preprocessed_data_final.pkl", "wb") as f:
        pickle.dump((X_train, y_train, X_eval, y_eval), f)
