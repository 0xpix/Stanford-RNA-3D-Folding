"""_summary_
This script processes raw RNA sequence and structure data into a format suitable for training a neural network.
Source: https://www.kaggle.com/code/olaflundstrom/stanford-rna-3d-folding-competition-notebook#3.-Data-Preprocessing
"""

import pickle
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
import h5py
from tqdm import tqdm
import multiprocessing
import jax
from jax import device_put
import gc
from jax import vmap, jit
import time

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
    coords = labels_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32)

    # 🔹 Group by `target_id` and stack residues
    label_dict = {}
    for target_id, group in labels_df.groupby("target_id"):
        # Sort by `resid` and stack the coordinates
        sorted_indices = np.argsort(group["resid"].to_numpy())
        sorted_coords = coords[group.index[sorted_indices]]  # Faster indexing

        # Keep as numpy initially to save memory
        label_dict[target_id] = sorted_coords

    return label_dict


def pad_sequences(sequences, max_len):
    """
    Pads sequences using NumPy operations.
    """
    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.int32)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded_sequences[i, :seq_len] = seq

    return padded_sequences


def pad_coordinates(coord_array, max_len):
    """
    Pads coordinate arrays to the max sequence length using NumPy.
    """
    L = coord_array.shape[0]
    if L < max_len:
        pad_width = ((0, max_len - L), (0, 0))  # Only pad along first axis
        return np.pad(coord_array, pad_width, mode="constant", constant_values=0)
    return coord_array[:max_len]


def load_msa_from_npz(target_id, msa_dir="data/processed/msa"):
    """
    Load MSA conservation data from a pre-generated NPZ file.

    Args:
        target_id: The RNA target identifier
        msa_dir: Directory containing MSA NPZ files

    Returns:
        Conservation scores as NumPy array or None if file not found
    """
    msa_file = os.path.join(msa_dir, f"{target_id}_features.npz")

    if not os.path.exists(msa_file):
        return None

    try:
        with np.load(msa_file, allow_pickle=True) as data:
            # Check if conservation data exists
            if "conservation" in data:
                conservation = np.array(data["conservation"], dtype=np.float32)
                return conservation
            else:
                log_message(f"Warning: No conservation data in {msa_file}")
                return None
    except Exception as e:
        log_message(f"Error loading MSA data for {target_id}: {str(e)}")
        return None


def load_bppm_from_npz(target_id, bppm_dir="data/processed/bppms_padded"):
    """
    Load BPPM data from a pre-generated NPZ file.

    Args:
        target_id: The RNA target identifier
        bppm_dir: Directory containing BPPM NPZ files

    Returns:
        BPPM matrix as NumPy array or None if file not found
    """
    bppm_file = os.path.join(bppm_dir, f"{target_id}_bppm.npz")

    if not os.path.exists(bppm_file):
        return None

    try:
        with np.load(bppm_file, allow_pickle=True) as data:
            # Check if BPPM data exists
            if "bppm" in data:
                # Keep as NumPy array to avoid GPU memory problems
                bppm = np.array(data["bppm"], dtype=np.float32)
                return bppm
            else:
                log_message(f"Warning: No BPPM data in {bppm_file}")
                return None
    except Exception as e:
        log_message(f"Error loading BPPM data for {target_id}: {str(e)}")
        return None


def process_single_sample(row, labels_dict, msa_dir, bppm_dir, max_len):
    """
    Process a single RNA sample and return all features as padded NumPy arrays.

    Args:
        row: DataFrame row containing target_id and sequence
        labels_dict: Dictionary mapping target_id to 3D coordinates
        msa_dir: Directory for MSA features
        bppm_dir: Directory for BPPM features
        max_len: Maximum length for padding

    Returns:
        Dictionary with processed features or None if sample should be skipped
    """
    tid = row["target_id"]
    seq = row["sequence"]

    # Skip if no 3D coordinates
    if tid not in labels_dict:
        return None, "Missing 3D coordinates"

    # Load MSA conservation data
    conservation = load_msa_from_npz(tid, msa_dir)
    if conservation is None:
        return None, "Missing MSA conservation data"

    # Load BPPM data
    bppm = load_bppm_from_npz(tid, bppm_dir)
    if bppm is None:
        return None, "Missing BPPM data"

    # If we made it here, we have all the required data
    # Encode and pad sequence
    encoded_seq = np.array([nucleotide_map.get(ch, 0) for ch in seq], dtype=np.int32)
    padded_seq = np.zeros(max_len, dtype=np.int32)
    padded_seq[: len(encoded_seq)] = encoded_seq

    # Pad MSA conservation
    padded_conservation = np.zeros(max_len, dtype=np.float32)
    padded_conservation[: min(len(conservation), max_len)] = conservation[
        : min(len(conservation), max_len)
    ]

    # Pad BPPM
    padded_bppm = pad_bppm(bppm, max_len)

    # Pad coordinates
    coords = labels_dict[tid]
    padded_coords = pad_coordinates(coords, max_len)

    # Return processed sample
    return {
        "X": padded_seq,
        "msa_conservation": padded_conservation,
        "bppm": padded_bppm,
        "y": padded_coords,
        "target_id": tid,
    }, None


def incremental_processing(
    sequences_df,
    labels_dict,
    max_len,
    msa_dir="data/processed/msa",
    bppm_dir="data/processed/bppms_padded",
    output_file="data/processed/processed_data_final.h5",
    batch_size=10,
    mode="train",
):
    """
    Process RNA samples incrementally, saving batches to disk to minimize memory usage.
    Uses HDF5 as the final output format.

    Args:
        sequences_df: DataFrame with target_id and sequence
        labels_dict: Dictionary mapping target_id to 3D coordinates
        max_len: Maximum sequence length for padding
        msa_dir: Directory containing MSA files
        bppm_dir: Directory containing BPPM files
        output_file: Path to save the processed data (HDF5 format)
        batch_size: Number of samples to process before saving
        mode: 'train' or 'valid'

    Returns:
        Dictionary with dataset metadata
    """
    log_message(
        f"Starting incremental processing for {mode} data ({len(sequences_df)} samples)"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create HDF5 file for saving data incrementally (this is now the final output)
    h5_file = output_file
    with h5py.File(h5_file, "w") as h5f:
        # Create datasets with extendable dimensions
        x_dset = h5f.create_dataset(
            "X",
            shape=(0, max_len),
            maxshape=(None, max_len),
            dtype=np.int32,
            chunks=(batch_size, max_len),
            compression="gzip",
            compression_opts=4,
        )
        msa_dset = h5f.create_dataset(
            "msa_conservation",
            shape=(0, max_len),
            maxshape=(None, max_len),
            dtype=np.float32,
            chunks=(batch_size, max_len),
            compression="gzip",
            compression_opts=4,
        )
        bppm_dset = h5f.create_dataset(
            "bppm",
            shape=(0, max_len, max_len),
            maxshape=(None, max_len, max_len),
            dtype=np.float32,
            chunks=(1, max_len, max_len),
            compression="gzip",
            compression_opts=1,  # Use lower compression for the large BPPMs for better speed
        )
        y_dset = h5f.create_dataset(
            "y",
            shape=(0, max_len, 3),
            maxshape=(None, max_len, 3),
            dtype=np.float32,
            chunks=(batch_size, max_len, 3),
            compression="gzip",
            compression_opts=4,
        )

        # Create a list to store target IDs
        target_ids = []
        skipped_targets = []

        # Process in batches
        batch_x = []
        batch_msa = []
        batch_bppm = []
        batch_y = []

        total_processed = 0

        # Process each sample
        for idx, row in tqdm(
            sequences_df.iterrows(),
            total=len(sequences_df),
            desc=f"Processing {mode} samples",
        ):
            # Process the sample
            sample, skip_reason = process_single_sample(
                row, labels_dict, msa_dir, bppm_dir, max_len
            )

            if sample is None:
                skipped_targets.append((row["target_id"], skip_reason))
                continue

            # Add to batch
            batch_x.append(sample["X"])
            batch_msa.append(sample["msa_conservation"])
            batch_bppm.append(sample["bppm"])
            batch_y.append(sample["y"])
            target_ids.append(sample["target_id"])

            # Save batch when it reaches batch_size
            if len(batch_x) >= batch_size:
                # Resize datasets
                current_size = x_dset.shape[0]
                new_size = current_size + len(batch_x)

                x_dset.resize(new_size, axis=0)
                msa_dset.resize(new_size, axis=0)
                bppm_dset.resize(new_size, axis=0)
                y_dset.resize(new_size, axis=0)

                # Write data
                x_dset[current_size:new_size] = np.array(batch_x)
                msa_dset[current_size:new_size] = np.array(batch_msa)
                bppm_dset[current_size:new_size] = np.array(batch_bppm)
                y_dset[current_size:new_size] = np.array(batch_y)

                # Update counter and clear batch
                total_processed += len(batch_x)

                batch_x = []
                batch_msa = []
                batch_bppm = []
                batch_y = []

                # Force garbage collection
                gc.collect()

        # Save any remaining samples
        if len(batch_x) > 0:
            current_size = x_dset.shape[0]
            new_size = current_size + len(batch_x)

            x_dset.resize(new_size, axis=0)
            msa_dset.resize(new_size, axis=0)
            bppm_dset.resize(new_size, axis=0)
            y_dset.resize(new_size, axis=0)

            x_dset[current_size:new_size] = np.array(batch_x)
            msa_dset[current_size:new_size] = np.array(batch_msa)
            bppm_dset[current_size:new_size] = np.array(batch_bppm)
            y_dset[current_size:new_size] = np.array(batch_y)

            total_processed += len(batch_x)

        # Save metadata as attributes
        h5f.attrs["max_len"] = max_len
        h5f.attrs["num_samples"] = total_processed
        h5f.attrs["date_created"] = str(pd.Timestamp.now())

        # Create a dataset for target IDs
        tid_dt = h5py.special_dtype(vlen=str)
        tid_dset = h5f.create_dataset(
            "target_ids", shape=(len(target_ids),), dtype=tid_dt
        )
        for i, tid in enumerate(target_ids):
            tid_dset[i] = tid

    log_message(
        f"Completed incremental processing for {mode}: {total_processed} samples included, {len(skipped_targets)} skipped"
    )
    log_message(f"Data saved to HDF5 file: {h5_file}")

    if skipped_targets:
        log_message(f"First 5 skipped targets (or fewer if less than 5):")
        for tid, reason in skipped_targets[:5]:
            log_message(f"  - {tid}: {reason}")

    # No conversion to .npz or .pkl needed - HDF5 is the final format

    return {
        "samples_processed": total_processed,
        "skipped": len(skipped_targets),
        "output_file": h5_file,
        "target_ids": target_ids,
        "max_len": max_len,
    }


def pad_bppm(bppm, max_len):
    """
    Pads BPPM to the max sequence length using NumPy to reduce GPU memory usage.
    """
    L = bppm.shape[0]
    if L < max_len:
        # Pad the BPPM using NumPy
        padded_bppm = np.pad(
            bppm,
            ((0, max_len - L), (0, max_len - L)),
            mode="constant",
            constant_values=0,
        )
        return padded_bppm
    return bppm[:max_len, :max_len]


def load_data_from_h5(h5_file, use_jax=True, batch_size=8):
    """
    Load RNA data from HDF5 file for model training.

    Args:
        h5_file: Path to HDF5 file containing processed data
        use_jax: If True, convert arrays to JAX arrays; otherwise, keep as NumPy
        batch_size: Number of samples to load at once to manage memory

    Returns:
        Dictionary containing the data ready for model training
    """
    log_message(f"Loading data from {h5_file}")

    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

    with h5py.File(h5_file, "r") as h5f:
        # Get metadata
        max_len = int(h5f.attrs["max_len"])
        num_samples = int(h5f.attrs["num_samples"])

        # Load target IDs
        target_ids = [
            tid.decode("utf-8") if isinstance(tid, bytes) else str(tid)
            for tid in h5f["target_ids"][:]
        ]

        # Load data in batches to manage memory
        if use_jax:
            # For JAX arrays
            X = jnp.zeros((num_samples, max_len), dtype=jnp.int32)
            msa_conservation = jnp.zeros((num_samples, max_len), dtype=jnp.float32)
            bppm = jnp.zeros((num_samples, max_len, max_len), dtype=jnp.float32)
            y = jnp.zeros((num_samples, max_len, 3), dtype=jnp.float32)

            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)

                # Load batch as numpy first
                X_batch = np.array(h5f["X"][i:end_idx])
                msa_batch = np.array(h5f["msa_conservation"][i:end_idx])
                bppm_batch = np.array(h5f["bppm"][i:end_idx])
                y_batch = np.array(h5f["y"][i:end_idx])

                # Convert to JAX and assign
                X = X.at[i:end_idx].set(jnp.array(X_batch))
                msa_conservation = msa_conservation.at[i:end_idx].set(
                    jnp.array(msa_batch)
                )
                bppm = bppm.at[i:end_idx].set(jnp.array(bppm_batch))
                y = y.at[i:end_idx].set(jnp.array(y_batch))

                # Force garbage collection
                del X_batch, msa_batch, bppm_batch, y_batch
                if (i // batch_size) % 5 == 0:
                    gc.collect()
        else:
            # For NumPy arrays (memory efficient)
            X = np.array(h5f["X"][:])
            msa_conservation = np.array(h5f["msa_conservation"][:])
            bppm = np.array(h5f["bppm"][:])
            y = np.array(h5f["y"][:])

    # Create the data dictionary
    data = {
        "X": X,
        "msa_conservation": msa_conservation,
        "bppm": bppm,
        "y": y,
        "target_ids": target_ids,
        "max_len": max_len,
    }

    log_message(f"Data loaded successfully: {num_samples} samples")
    log_message(
        f"Shapes: X={X.shape}, msa={msa_conservation.shape}, bppm={bppm.shape}, y={y.shape}"
    )

    return data


# Add an example usage function
def example_h5_loading():
    """Example of how to load data from HDF5 files for training"""
    # Load training data as JAX arrays
    train_data = load_data_from_h5(
        "data/processed/processed_data_final.h5", use_jax=True, batch_size=8
    )

    # Load validation data as NumPy arrays (more memory efficient)
    valid_data = load_data_from_h5(
        "data/processed/processed_data_valid_final.h5", use_jax=False, batch_size=8
    )

    # Access the data
    X_train = train_data["X"]  # JAX array
    y_train = train_data["y"]

    X_valid = valid_data["X"]  # NumPy array
    y_valid = valid_data["y"]

    # Example training loop would go here
    # ...

    return train_data, valid_data


if __name__ == "__main__":
    log_message("🧬 Data processing started!")
    start_time = time.time()
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

    # 🔹 Process labels - use NumPy instead of JAX for memory efficiency
    log_message("Processing labels")
    train_labels_dict = process_labels(train_labels)
    valid_labels_dict = process_labels(valid_labels)

    # 🔹 Estimate maximum sequence length from a sample of sequences
    # This avoids loading all sequences at once
    log_message("Estimating maximum sequence length")

    # Sample a small number of targets to estimate max length
    sample_train_rows = train_sequences.sample(min(50, len(train_sequences)))
    sample_valid_rows = valid_sequences.sample(min(50, len(valid_sequences)))

    # Get a preliminary estimate from sequences
    seq_lens = [len(row["sequence"]) for _, row in sample_train_rows.iterrows()]
    seq_lens += [len(row["sequence"]) for _, row in sample_valid_rows.iterrows()]

    # Load a few BPPMs to check their dimensions
    bppm_sample_size = 10
    bppm_train_sample = [
        load_bppm_from_npz(row["target_id"])
        for _, row in train_sequences.sample(bppm_sample_size).iterrows()
        if load_bppm_from_npz(row["target_id"]) is not None
    ]

    bppm_valid_sample = [
        load_bppm_from_npz(row["target_id"])
        for _, row in valid_sequences.sample(bppm_sample_size).iterrows()
        if load_bppm_from_npz(row["target_id"]) is not None
    ]

    bppm_lens = [
        b.shape[0] for b in bppm_train_sample + bppm_valid_sample if b is not None
    ]

    # Use the maximum dimension across sampled data
    max_len = max(max(seq_lens), max(bppm_lens) if bppm_lens else 0)
    # Add a small buffer to be safe
    max_len = int(max_len * 1.05)

    log_message(f"Estimated maximum dimension: {max_len}")

    # 🔹 Process data incrementally and save directly to HDF5 format
    train_result = incremental_processing(
        train_sequences,
        train_labels_dict,
        max_len,
        output_file="data/processed/processed_data_final.h5",
        batch_size=8,
        mode="train",
    )

    valid_result = incremental_processing(
        valid_sequences,
        valid_labels_dict,
        max_len,
        output_file="data/processed/processed_data_valid_final.h5",
        batch_size=8,
        mode="validation",
    )

    # Clean up any existing .npz or .pkl files that might confuse users
    for ext in [".npz", ".pkl"]:
        for prefix in ["processed_data_final", "processed_data_valid_final"]:
            file_path = f"data/processed/{prefix}{ext}"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    log_message(f"Removed obsolete file: {file_path}")
                except:
                    log_message(f"Note: Could not remove obsolete file: {file_path}")

    total_time = time.time() - start_time
    log_message(
        f"✅ Data processing complete with unified features in {total_time:.2f} seconds!"
    )
    log_message(f"Training samples: {train_result['samples_processed']}")
    log_message(f"Validation samples: {valid_result['samples_processed']}")
    log_message("Files saved to data/processed/ in HDF5 format\n")
