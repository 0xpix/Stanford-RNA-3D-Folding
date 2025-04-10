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

    # Calculate conservation scores using JAX vectorization
    seq_len = len(sequences[0])

    # Convert sequences to a matrix for vectorized operations
    seq_matrix = np.zeros((len(sequences), seq_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j, nt in enumerate(seq[:seq_len]):
            if nt.upper() == 'A':
                seq_matrix[i, j] = 1
            elif nt.upper() == 'C':
                seq_matrix[i, j] = 2
            elif nt.upper() == 'G':
                seq_matrix[i, j] = 3
            elif nt.upper() == 'U':
                seq_matrix[i, j] = 4
            # All other characters (gaps, etc.) are 0

    # JAX-friendly conservation calculation
    seq_matrix_jax = jnp.array(seq_matrix)

    def get_conservation(pos_data):
        # Count occurrences of each nucleotide
        counts = jnp.bincount(pos_data, length=5)  # 0-4 for gap, A, C, G, U
        total = jnp.sum(counts)
        # Avoid division by zero
        return jnp.where(total > 0, jnp.max(counts) / total, 0.0)

    # Apply conservation calculation to each position using vmap with jit
    get_conservation_batched = jit(vmap(get_conservation))
    conservation = get_conservation_batched(seq_matrix_jax.T)

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

    return bppm  # Return NumPy array instead of converting to JAX array here

def process_bppm_parallel(sequences, n_jobs=None):
    """
    Generates BPPMs for multiple sequences in parallel.

    Args:
        sequences: List of RNA sequences
        n_jobs: Number of CPU cores to use, None means all available

    Returns:
        List of BPPM matrices
    """
    if RNA is None:
        # Return zeros if ViennaRNA is not available
        return [np.zeros((len(seq), len(seq)), dtype=np.float32) for seq in sequences]

    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    log_message(f"Generating BPPMs using {n_jobs} CPU cores")

    # Use pool.imap for a progress bar
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(generate_bppm, sequences),
            total=len(sequences),
            desc="Generating BPPMs"
        ))

    return results

def process_sequences_with_features(sequences_df, labels_dict, msa_dir="data/raw/msa", n_jobs=None):
    """
    Creates a dataset with additional features: X (encoded RNA sequences),
    MSA conservation, BPPMs, y (3D coordinates), target_ids.
    """
    X, msas, y, target_ids, sequences = [], [], [], [], []

    log_message("Preprocessing sequences and MSA data")
    for _, row in tqdm(sequences_df.iterrows(), total=len(sequences_df), desc="Processing sequences"):
        tid = row["target_id"]
        seq = row["sequence"]

        if tid in labels_dict:
            # Basic sequence encoding
            X.append(encode_sequence(seq))

            # Store sequence for later BPPM batch processing
            sequences.append(seq)

            # MSA features
            msa_features = load_msa_data(tid, msa_dir)
            if msa_features is None:
                # If no MSA data, use zeros
                msa_features = {
                    'conservation': jnp.zeros(len(seq), dtype=jnp.float32),
                    'num_sequences': 0
                }
            msas.append(msa_features)

            # Labels
            y.append(labels_dict[tid])
            target_ids.append(tid)

    # Generate BPPMs in parallel
    log_message("Starting parallel BPPM generation")
    start_time = time.time()
    bppms = process_bppm_parallel(sequences, n_jobs=n_jobs)
    elapsed = time.time() - start_time
    log_message(f"BPPM generation completed in {elapsed:.2f} seconds")

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
    Pads BPPM to the max sequence length using NumPy to reduce GPU memory usage.
    """
    L = bppm.shape[0]
    if L < max_len:
        # Pad the BPPM using NumPy
        padded_bppm = np.pad(
            bppm,
            ((0, max_len - L), (0, max_len - L)),
            mode="constant",
            constant_values=0
        )
        return padded_bppm
    return bppm[:max_len, :max_len]

def process_bppms_in_batches(bppms, max_len, batch_size=10, temp_file=None):
    """
    Process and pad BPPMs in smaller batches to reduce memory usage.
    Uses HDF5 file-based storage to avoid large memory allocations.

    Args:
        bppms: List of BPPM matrices
        max_len: Maximum sequence length for padding
        batch_size: Size of batches to process at once
        temp_file: Optional file path for HDF5 storage

    Returns:
        Path to HDF5 file containing padded BPPMs or the padded array for small datasets
    """
    num_samples = len(bppms)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Generate a permanent filename in the data directory if not provided
    if temp_file is None:
        # Create directory if it doesn't exist
        os.makedirs("data/processed/bppms", exist_ok=True)
        temp_file = os.path.join("data/processed/bppms", f"bppms_{int(time.time())}.h5")

    log_message(f"Processing {num_samples} BPPMs in {num_batches} batches")
    log_message(f"Using file: {temp_file} to store BPPM matrices")

    # Use HDF5 to store the padded BPPMs without loading everything into memory
    with h5py.File(temp_file, 'w') as h5f:
        # Create a dataset with the right dimensions
        dset = h5f.create_dataset(
            "padded_bppms",
            shape=(num_samples, max_len, max_len),
            dtype=np.float32,
            chunks=(1, min(1024, max_len), min(1024, max_len)),  # Optimize chunk size
            compression="gzip",
            compression_opts=1  # Use minimal compression to speed up the process
        )

        # Use tqdm for progress tracking
        for i in tqdm(range(num_batches), desc="Padding BPPMs"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            log_message(f"Processing BPPM batch {i+1}/{num_batches} (samples {start_idx} to {end_idx-1})")

            # Process and write each BPPM individually
            for j in range(start_idx, end_idx):
                padded = pad_bppm(bppms[j], max_len)
                dset[j] = padded

                # Free memory immediately
                del padded

            # Force garbage collection every few batches
            if i % 5 == 0:
                import gc
                gc.collect()

    log_message(f"Completed padding all BPPMs to HDF5 file: {temp_file}")
    return temp_file

def load_bppms_from_h5(h5_file, indices=None, batch_size=4, use_cpu=False, output_file=None):
    """
    Load BPPMs from HDF5 file in batches, process them, and immediately save to a new HDF5 file.
    Avoids accumulating all data in memory.

    Args:
        h5_file: Path to HDF5 file with BPPMs
        indices: Optional list of indices to load (None loads all)
        batch_size: Number of matrices to load and process at once
        use_cpu: If True, process on CPU to reduce memory usage
        output_file: Path to save processed BPPMs (auto-generated if None)

    Returns:
        Path to the HDF5 file containing processed BPPM matrices
    """
    # Generate output filename if not provided
    if output_file is None:
        # Create directory if it doesn't exist
        os.makedirs("data/processed/bppms", exist_ok=True)
        output_file = os.path.join("data/processed/bppms", f"processed_bppms_{int(time.time())}.h5")

    log_message(f"Loading BPPMs from {h5_file} with batch_size={batch_size} and saving to {output_file}")

    # Get CPU device for initial processing
    cpu_device = jax.devices("cpu")[0]

    with h5py.File(h5_file, 'r') as h5f:
        dset = h5f["padded_bppms"]
        total_samples = dset.shape[0]
        sample_shape = dset[0].shape

        if indices is None:
            indices = list(range(total_samples))

        num_batches = (len(indices) + batch_size - 1) // batch_size

        # Create output file and dataset
        with h5py.File(output_file, 'w') as out_f:
            # Create dataset with the same shape and dtype
            out_dset = out_f.create_dataset(
                "padded_bppms",
                shape=(len(indices),) + sample_shape,
                dtype=dset.dtype,
                chunks=(min(batch_size, len(indices)),) + sample_shape,
                compression="gzip",
                compression_opts=1  # Use minimal compression for speed
            )

            for i in tqdm(range(num_batches), desc="Processing and saving BPPMs"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                # Load batch data
                batch_data = np.array([dset[j] for j in batch_indices])

                # Process batch (convert to JAX array if needed)
                if use_cpu:
                    # Process on CPU to reduce memory usage
                    batch_jax = device_put(jnp.array(batch_data), cpu_device)
                    # Convert back to numpy for saving
                    processed_batch = np.array(batch_jax)
                else:
                    processed_batch = batch_data

                # Save batch directly to output file
                out_dset[start_idx:end_idx] = processed_batch

                # Clear memory
                del batch_data
                del processed_batch
                if 'batch_jax' in locals():
                    del batch_jax

                # Force garbage collection every few batches
                if i % 3 == 0:
                    gc.collect()

    log_message(f"Successfully processed and saved all BPPMs to {output_file}")
    return output_file

def batch_process_arrays(arrays, process_fn, batch_size=32, description="Processing"):
    """
    Generic function to process arrays in batches to manage memory usage.

    Args:
        arrays: List of arrays to process
        process_fn: Function that takes an array and returns a processed array
        batch_size: Number of arrays to process in each batch
        description: Description for the progress bar

    Returns:
        List of processed arrays
    """
    results = []
    total = len(arrays)

    for i in tqdm(range(0, total, batch_size), desc=description):
        batch = arrays[i:min(i+batch_size, total)]
        processed_batch = [process_fn(arr) for arr in batch]
        results.extend(processed_batch)

        # Force garbage collection periodically
        if i % (batch_size * 5) == 0:
            import gc
            gc.collect()

    return results

def save_to_hdf5(filename, data_dict, compression="gzip", compression_opts=4):
    """
    Save data to HDF5 format with compression.
    Handles both direct data arrays and references to HDF5 files.
    """
    log_message(f"Saving data to {filename}")
    with h5py.File(filename, 'w') as f:
        # Create groups for organization
        train_group = f.create_group('train')
        valid_group = f.create_group('valid')

        # Save training data
        for key, value in tqdm(data_dict['train'].items(), desc="Saving training data"):
            if key == 'bppms_file':
                # Store the file path as an attribute
                train_group.attrs['bppms_file'] = np.string_(value)
            elif isinstance(value, list) and key == 'msas':
                # Handle MSA features specially
                msa_group = train_group.create_group('msas')
                for i, msa in enumerate(value):
                    msa_item = msa_group.create_group(f'item_{i}')
                    msa_item.create_dataset('conservation', data=msa['conservation'],
                                         compression=compression, compression_opts=compression_opts)
                    msa_item.attrs['num_sequences'] = msa['num_sequences']
            else:
                train_group.create_dataset(key, data=value,
                                        compression=compression, compression_opts=compression_opts)

        # Save validation data
        for key, value in tqdm(data_dict['valid'].items(), desc="Saving validation data"):
            if key == 'bppms_file':
                # Store the file path as an attribute
                valid_group.attrs['bppms_file'] = np.string_(value)
            elif isinstance(value, list) and key == 'msas':
                # Handle MSA features specially
                msa_group = valid_group.create_group('msas')
                for i, msa in enumerate(value):
                    msa_item = msa_group.create_group(f'item_{i}')
                    msa_item.create_dataset('conservation', data=msa['conservation'],
                                         compression=compression, compression_opts=compression_opts)
                    msa_item.attrs['num_sequences'] = msa['num_sequences']
            else:
                valid_group.create_dataset(key, data=value,
                                        compression=compression, compression_opts=compression_opts)

        # Save metadata
        f.attrs['max_len'] = data_dict['max_len']
        f.attrs['creation_date'] = np.string_(time.strftime("%Y-%m-%d %H:%M:%S"))

    log_message(f"Data successfully saved to {filename}")

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

    # 🔹 Process labels
    log_message("Processing labels")
    train_labels_dict = process_labels(train_labels)
    valid_labels_dict = process_labels(valid_labels)

    # 🔹 Create datasets with enhanced features
    log_message("Creating datasets with MSA and BPPM features")
    # Use available CPU cores minus one for BPPM generation
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    log_message(f"Using {n_jobs} CPU cores for parallel processing")

    X_train, msas_train, bppms_train, y_train, train_ids = process_sequences_with_features(
        train_sequences, train_labels_dict, n_jobs=n_jobs
    )
    X_valid, msas_valid, bppms_valid, y_valid, valid_ids = process_sequences_with_features(
        valid_sequences, valid_labels_dict, n_jobs=n_jobs
    )

    # 🔹 Determine max sequence length
    max_len = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_valid))
    log_message(f"Maximum sequence length: {max_len}")

    # 🔹 Pad sequences using JAX
    log_message(f"Padding sequences to max length {max_len}")
    X_train_pad = pad_sequences_jax(X_train, max_len)
    X_valid_pad = pad_sequences_jax(X_valid, max_len)

    # 🔹 Pad MSA features
    log_message("Padding MSA features")
    msas_train_pad = pad_msa_features(msas_train, max_len)
    msas_valid_pad = pad_msa_features(msas_valid, max_len)

    # 🔹 Pad BPPM features in smaller batches using disk-based storage
    log_message("Padding BPPM matrices in smaller batches using disk-based storage")
    # Use smaller batch size to reduce memory usage
    bppms_train_h5 = process_bppms_in_batches(bppms_train, max_len, batch_size=16)
    bppms_valid_h5 = process_bppms_in_batches(bppms_valid, max_len, batch_size=16)

    # 🔹 Pad labels using JAX in batches
    log_message("Padding coordinates in batches")
    def pad_coord_fn(arr):
        return pad_coordinates_jax(arr, max_len)
    y_train_batched = batch_process_arrays(y_train, pad_coord_fn, batch_size=64, description="Padding train coords")
    y_valid_batched = batch_process_arrays(y_valid, pad_coord_fn, batch_size=64, description="Padding valid coords")

    # Convert to JAX arrays
    log_message("Converting coordinates to JAX arrays")
    y_train_pad = jnp.array(y_train_batched)
    y_valid_pad = jnp.array(y_valid_batched)

    # Load BPPMs from HDF5 files in batches and convert to JAX arrays
    log_message("Loading BPPMs from HDF5 with batch-by-batch processing to avoid OOM")
    bppms_train_file = load_bppms_from_h5(bppms_train_h5, batch_size=4, use_cpu=True)
    bppms_valid_file = load_bppms_from_h5(bppms_valid_h5, batch_size=4, use_cpu=True)

    # Keep the intermediate HDF5 files for reference
    log_message("Keeping intermediate HDF5 files for training")

    # 🔹 Save processed data to HDF5 format
    log_message("Saving processed data to HDF5 format")
    data_dict = {
        'train': {
            'X': X_train_pad,
            'y': y_train_pad,
            'msas': msas_train_pad,
            'bppms_file': bppms_train_file,  # Store file path instead of array
            'target_ids': np.array(train_ids, dtype='S')  # Convert strings to fixed-length bytes
        },
        'valid': {
            'X': X_valid_pad,
            'y': y_valid_pad,
            'msas': msas_valid_pad,
            'bppms_file': bppms_valid_file,  # Store file path instead of array
            'target_ids': np.array(valid_ids, dtype='S')
        },
        'max_len': max_len
    }

    # Save to both formats for backward compatibility
    save_to_hdf5("data/processed/processed_data_msa_bppm.h5", data_dict)

    # Also save in pickle format for backward compatibility
    log_message("Also saving in pickle format for backward compatibility")
    with open("data/processed/processed_data_msa_bppm.pkl", "wb") as f:
        pickle.dump(
            (
                X_train_pad, y_train_pad, msas_train_pad, bppms_train_file,
                X_valid_pad, y_valid_pad, msas_valid_pad, bppms_valid_file,
                max_len
            ),
            f
        )

    total_time = time.time() - start_time
    log_message(f"✅ Data processing complete with MSA and BPPM features in {total_time:.2f} seconds!")
    log_message("Files saved to data/processed/ in both HDF5 and pickle formats\n")
