"""
Training script for RNA 3D Folding models.
This script handles data loading, training loop, evaluation, and checkpointing.
"""

import os
import time
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from flax.training import train_state
import optax
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.model.model1.model import RNAFoldingModel
from src.utils import log_message

# Configuration variables
DATA_PATH = 'data/processed/processed_data_msa_bppm.h5'
MODEL_DIM = 256
NUM_HEADS = 8
DEPTH = 6
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 100
WARMUP_STEPS = 1000
SEED = 42
OUTPUT_DIR = 'checkpoints'

# To override these defaults, modify the values above or use environment variables
import os
DATA_PATH = os.environ.get('RNA_DATA_PATH', DATA_PATH)
MODEL_DIM = int(os.environ.get('RNA_MODEL_DIM', MODEL_DIM))
NUM_HEADS = int(os.environ.get('RNA_NUM_HEADS', NUM_HEADS))
DEPTH = int(os.environ.get('RNA_DEPTH', DEPTH))
BATCH_SIZE = int(os.environ.get('RNA_BATCH_SIZE', BATCH_SIZE))
LEARNING_RATE = float(os.environ.get('RNA_LEARNING_RATE', LEARNING_RATE))
EPOCHS = int(os.environ.get('RNA_EPOCHS', EPOCHS))
WARMUP_STEPS = int(os.environ.get('RNA_WARMUP_STEPS', WARMUP_STEPS))
SEED = int(os.environ.get('RNA_SEED', SEED))
OUTPUT_DIR = os.environ.get('RNA_OUTPUT_DIR', OUTPUT_DIR)


def create_train_state(rng, model, learning_rate, warmup_steps, max_len=200):
    """Create initial training state with Flax."""
    # Cap the maximum sequence length for initialization to something reasonable
    # This avoids OOM during initialization but still allows training on longer sequences
    init_seq_len = min(max_len, 512)  # Cap to 512 for initialization

    log_message(f"Initializing model with sequence length {init_seq_len} (max training length: {max_len})")

    # Create dummy inputs with the initialization length
    dummy_tokens = jnp.ones((1, init_seq_len), dtype=jnp.int32)
    dummy_msa = jnp.ones((1, init_seq_len), dtype=jnp.float32)
    dummy_bppm = jnp.ones((1, init_seq_len, init_seq_len), dtype=jnp.float32) * 0.1

    # For large sequences, initialize with dummy bppm that has limited edges
    # This vastly reduces memory usage during initialization
    if init_seq_len > 256:
        # Create a sparse dummy BPPM with only diagonal and near-diagonal edges
        dummy_bppm = jnp.zeros((1, init_seq_len, init_seq_len), dtype=jnp.float32)
        # Add diagonal elements (self-loops)
        for i in range(init_seq_len):
            dummy_bppm = dummy_bppm.at[0, i, i].set(0.9)
            # Add some near-diagonal edges
            for j in range(1, 5):  # Add edges to 4 nearest neighbors
                if i + j < init_seq_len:
                    dummy_bppm = dummy_bppm.at[0, i, i+j].set(0.5)
                if i - j >= 0:
                    dummy_bppm = dummy_bppm.at[0, i, i-j].set(0.5)

    # Split initialization into stages to avoid OOM
    # First initialize only the feature processing part
    feature_params = model.init(
        rng,
        dummy_tokens,
        dummy_msa,
        dummy_bppm,
        training=False
    )

    # Learning rate schedule: linear warmup and then cosine decay
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=100000,
        end_value=learning_rate / 10.0
    )

    # Optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=schedule_fn)
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=feature_params,
        tx=tx
    )


def load_data(data_path):
    """Load preprocessed data from HDF5 file."""
    log_message(f"Loading data from {data_path}")

    data = {}
    with h5py.File(data_path, 'r') as f:
        # Get max_len from attributes
        data['max_len'] = f.attrs['max_len']

        # Load training data
        data['train'] = {
            'X': np.array(f['train/X']),
            'y': np.array(f['train/y']),
            'target_ids': np.array(f['train/target_ids'])
        }

        # Load MSA features
        msa_group = f['train/msas']
        msas_train = []
        for i in range(len(msa_group)):
            item = msa_group[f'item_{i}']
            msa_feat = {
                'conservation': np.array(item['conservation']),
                'num_sequences': item.attrs['num_sequences']
            }
            msas_train.append(msa_feat)
        data['train']['msas'] = msas_train

        # Store path to BPPM file
        data['train']['bppms_file'] = f['train'].attrs['bppms_file'].decode('utf-8')

        # Load validation data
        data['valid'] = {
            'X': np.array(f['valid/X']),
            'y': np.array(f['valid/y']),
            'target_ids': np.array(f['valid/target_ids'])
        }

        # Load MSA features for validation
        msa_group = f['valid/msas']
        msas_valid = []
        for i in range(len(msa_group)):
            item = msa_group[f'item_{i}']
            msa_feat = {
                'conservation': np.array(item['conservation']),
                'num_sequences': item.attrs['num_sequences']
            }
            msas_valid.append(msa_feat)
        data['valid']['msas'] = msas_valid

        # Store path to validation BPPM file
        data['valid']['bppms_file'] = f['valid'].attrs['bppms_file'].decode('utf-8')

    return data


def load_bppms_batch(bppms_file, indices):
    """Load a batch of BPPMs from HDF5 file."""
    with h5py.File(bppms_file, 'r') as f:
        bppms = np.array([f['padded_bppms'][i] for i in indices])
    return bppms


def compute_rmsd(pred_coords, true_coords, mask=None):
    """
    Compute Root Mean Square Deviation (RMSD) between predicted and true coordinates.
    If mask is provided, only compute RMSD for non-masked positions.
    """
    if mask is None:
        # Use all positions
        mask = (true_coords.sum(axis=-1) != 0).astype(jnp.float32)  # (B, L)

    # Center the structures to remove translation
    pred_mean = jnp.sum(pred_coords * mask[..., None], axis=1) / jnp.sum(mask, axis=1)[:, None]
    true_mean = jnp.sum(true_coords * mask[..., None], axis=1) / jnp.sum(mask, axis=1)[:, None]

    pred_centered = pred_coords - pred_mean[:, None, :]
    true_centered = true_coords - true_mean[:, None, :]

    # Compute RMSD
    sq_diff = jnp.sum((pred_centered - true_centered) ** 2, axis=-1)  # (B, L)
    rmsd = jnp.sqrt(jnp.sum(sq_diff * mask, axis=1) / jnp.sum(mask, axis=1))  # (B,)

    return rmsd


def compute_loss(params, apply_fn, batch):
    """Compute the training loss."""
    tokens, msa_conservation, bppms, true_coords = batch

    # Generate sequence mask based on tokens - padding token is 0
    mask = (tokens != 0).astype(jnp.float32)  # positions with padding token = 0

    # Check shapes before forward pass for debugging
    B, L = tokens.shape
    L_msa = msa_conservation.shape[1]
    L_bppms = bppms.shape[1]

    if L != L_msa or L != L_bppms:
        # Log shape mismatch for debugging but continue with computation
        print(f"WARNING: Shape mismatch - tokens: {tokens.shape}, MSA: {msa_conservation.shape}, BPPM: {bppms.shape}")

        # Handle potential shape mismatches - truncate or pad if needed
        if L_msa > L:
            msa_conservation = msa_conservation[:, :L]
        elif L_msa < L:
            # Pad with zeros
            padding = jnp.zeros((B, L - L_msa))
            msa_conservation = jnp.pad(msa_conservation, ((0, 0), (0, L - L_msa)), mode='constant')

        # Ensure BPPM matrix has correct dimensions
        if L_bppms != L:
            if L_bppms > L:
                bppms = bppms[:, :L, :L]
            else:
                # Pad with zeros
                bppms = jnp.pad(bppms, ((0, 0), (0, L - L_bppms), (0, L - L_bppms)), mode='constant')

    # Forward pass
    pred_coords = apply_fn(params, tokens, msa_conservation, bppms)

    # Compute masked MSE loss
    mask_3d = mask[..., None]  # (B, L, 1)

    # Safely handle potential shape mismatches in coordinates
    if pred_coords.shape != true_coords.shape:
        min_len = min(pred_coords.shape[1], true_coords.shape[1])
        pred_coords = pred_coords[:, :min_len]
        true_coords = true_coords[:, :min_len]
        mask_3d = mask_3d[:, :min_len]

    # Prevent division by zero
    total_mask = jnp.sum(mask_3d)
    mse_loss = jnp.sum(((pred_coords - true_coords) ** 2) * mask_3d) / (total_mask + 1e-8)

    # Compute RMSD for each sequence
    rmsds = compute_rmsd(pred_coords, true_coords, mask)
    mean_rmsd = jnp.mean(rmsds)

    return mse_loss, mean_rmsd


@jax.jit
def train_step(state, batch):
    """Execute one training step."""
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, rmsd), grads = grad_fn(state.params, state.apply_fn, batch)

    # Update parameters
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss, rmsd


@jax.jit
def eval_step(state, batch):
    """Evaluate model on a batch."""
    loss, rmsd = compute_loss(state.params, state.apply_fn, batch)
    return loss, rmsd


def create_batch_indices(num_samples, batch_size, rng_key):
    """Create shuffled batch indices."""
    indices = jnp.arange(num_samples)
    indices = jax.random.permutation(rng_key, indices)

    # Create batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_indices = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_indices.append(indices[start_idx:end_idx])

    return batch_indices


def extract_msa_conservation(msas_data, batch_indices):
    """Extract conservation scores from MSA data for a batch."""
    batch_conservation = []
    for idx in batch_indices:
        batch_conservation.append(msas_data[idx]['conservation'])
    return jnp.array(batch_conservation)


def get_max_sequence_len(batch_indices, data):
    """Calculate maximum sequence length within a batch to avoid unnecessary padding."""
    max_len_in_batch = 0
    for idx in batch_indices:
        seq = data['X'][idx]
        # Find where padding starts (padding token is 0)
        non_pad = np.nonzero(seq)[0]
        if len(non_pad) > 0:
            seq_len = non_pad[-1] + 1
            max_len_in_batch = max(max_len_in_batch, seq_len)
    return max_len_in_batch


def chunk_long_sequence(tokens, msa_conservation, bppms, true_coords, max_chunk_size=512):
    """
    Chunk very long sequences into manageable pieces that fit in memory.
    For sequences longer than max_chunk_size, we create overlapping chunks.
    """
    B, L = tokens.shape
    if L <= max_chunk_size:
        return [(tokens, msa_conservation, bppms, true_coords)]

    chunks = []
    overlap_size = 64  # Overlap between chunks to maintain context

    for start_idx in range(0, L, max_chunk_size - overlap_size):
        end_idx = min(start_idx + max_chunk_size, L)

        # Skip small final chunks
        if end_idx - start_idx < 128 and end_idx != L:
            continue

        chunk_tokens = tokens[:, start_idx:end_idx]
        chunk_msa = msa_conservation[:, start_idx:end_idx]
        chunk_bppms = bppms[:, start_idx:end_idx, start_idx:end_idx]
        chunk_coords = true_coords[:, start_idx:end_idx]

        chunks.append((chunk_tokens, chunk_msa, chunk_bppms, chunk_coords))

        # If we've reached the end, break
        if end_idx == L:
            break

    return chunks


def chunk_long_sequence_with_overlap(tokens, msa_conservation, bppms, true_coords, window_size=1024, overlap=256):
    """
    Process very long RNA sequences using a sliding window with overlap approach.
    This preserves context and allows capturing both local and long-range interactions.

    Args:
        tokens: RNA sequence tokens (B, L)
        msa_conservation: Conservation scores (B, L)
        bppms: Base pair probability matrices (B, L, L)
        true_coords: Target 3D coordinates (B, L, 3)
        window_size: Size of each window
        overlap: Overlap between consecutive windows

    Returns:
        List of chunks with overlapping regions to maintain context
    """
    B, L = tokens.shape
    if L <= window_size:
        return [(tokens, msa_conservation, bppms, true_coords)]

    chunks = []
    stride = window_size - overlap

    for start_idx in range(0, L, stride):
        end_idx = min(start_idx + window_size, L)

        # If this is a small final chunk, merge with previous
        if end_idx - start_idx < overlap and len(chunks) > 0:
            continue

        # Extract chunk with context
        chunk_tokens = tokens[:, start_idx:end_idx]
        chunk_msa = msa_conservation[:, start_idx:end_idx]
        chunk_bppms = bppms[:, start_idx:end_idx, start_idx:end_idx]
        chunk_coords = true_coords[:, start_idx:end_idx]

        # Get BPPM connections to regions outside the chunk
        # These represent long-range interactions that would be lost by simple chunking
        if start_idx > 0 or end_idx < L:
            # Store metadata about the chunk position for later stitching
            chunk_meta = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'full_length': L,
            }
            chunks.append((chunk_tokens, chunk_msa, chunk_bppms, chunk_coords, chunk_meta))
        else:
            chunks.append((chunk_tokens, chunk_msa, chunk_bppms, chunk_coords))

        # If we've reached the end, break
        if end_idx == L:
            break

    return chunks


def train_model():
    """Main training function."""
    # Set up logging and checkpointing
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = SummaryWriter(OUTPUT_DIR)

    # Initialize random keys
    rng = PRNGKey(SEED)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    # Load data
    data = load_data(DATA_PATH)
    max_len = data['max_len']
    log_message(f"Data loaded with max sequence length: {max_len}")

    # Cap the maximum length for memory efficiency
    max_model_len = min(max_len, 1024)  # Limit to 1024 for now to ensure it trains
    log_message(f"Setting maximum model sequence length to {max_model_len} (original max length: {max_len})")

    # Configure dynamic batch size based on sequence length for memory efficiency
    def get_dynamic_batch_size(seq_len):
        if seq_len <= 256:
            return BATCH_SIZE * 2  # Double batch size for short sequences
        elif seq_len <= 512:
            return BATCH_SIZE
        elif seq_len <= 1024:
            return max(BATCH_SIZE // 2, 4)  # Halve for medium sequences
        else:
            return max(BATCH_SIZE // 4, 2)  # Quarter for very long sequences

    # Initialize model
    model = RNAFoldingModel(
        dim=MODEL_DIM,
        heads=NUM_HEADS,
        depth=DEPTH
    )

    # Create training state with the capped max length
    state = create_train_state(
        init_rng,
        model,
        LEARNING_RATE,
        WARMUP_STEPS,
        max_model_len
    )

    # Training loop
    best_val_rmsd = float('inf')
    start_time = time.time()

    log_message("Starting training...")
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Training
        rng, subkey = jax.random.split(rng)
        train_batch_indices = create_batch_indices(
            len(data['train']['X']),
            BATCH_SIZE,
            subkey
        )

        train_losses = []
        train_rmsds = []

        for batch_idx, batch_indices in enumerate(tqdm(train_batch_indices, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            # Get maximum sequence length in this batch to minimize memory usage
            max_batch_seq_len = get_max_sequence_len(batch_indices, data['train'])
            max_batch_seq_len = min(max_batch_seq_len, max_model_len)

            # Adjust batch size dynamically based on sequence length
            dynamic_batch_size = get_dynamic_batch_size(max_batch_seq_len)

            # If current batch is too large, split it further
            if len(batch_indices) > dynamic_batch_size:
                sub_batches = [batch_indices[i:i+dynamic_batch_size]
                              for i in range(0, len(batch_indices), dynamic_batch_size)]
            else:
                sub_batches = [batch_indices]

            # Process each sub-batch
            for sub_batch_indices in sub_batches:
                # Prepare batch data with optimal padding for this specific batch
                tokens_batch = jnp.array(data['train']['X'][sub_batch_indices])[:, :max_batch_seq_len]
                msa_batch = extract_msa_conservation(data['train']['msas'], sub_batch_indices)[:, :max_batch_seq_len]
                bppms_batch = jnp.array(load_bppms_batch(data['train']['bppms_file'], sub_batch_indices))[:, :max_batch_seq_len, :max_batch_seq_len]
                coords_batch = jnp.array(data['train']['y'][sub_batch_indices])[:, :max_batch_seq_len]

                batch = (tokens_batch, msa_batch, bppms_batch, coords_batch)

                # Check if sequence is too long - if so, chunk it
                if max_batch_seq_len > 1024:
                    log_message(f"Long sequence detected ({max_batch_seq_len} > 1024), chunking for training")
                    chunks = chunk_long_sequence_with_overlap(*batch, window_size=1024, overlap=256)

                    # Process each chunk
                    chunk_losses = []
                    chunk_rmsds = []
                    for chunk_batch in chunks:
                        # Train step on chunk
                        state, chunk_loss, chunk_rmsd = train_step(state, chunk_batch)
                        chunk_losses.append(chunk_loss)
                        chunk_rmsds.append(chunk_rmsd)

                    # Average the losses and RMSDs from chunks
                    loss = sum(chunk_losses) / len(chunk_losses)
                    rmsd = sum(chunk_rmsds) / len(chunk_rmsds)
                else:
                    # Normal train step for regular-sized sequences
                    state, loss, rmsd = train_step(state, batch)

                train_losses.append(loss)
                train_rmsds.append(rmsd)

            # Log every 10 batches
            if batch_idx % 10 == 0:
                log_message(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss:.4f}, RMSD = {rmsd:.4f}, Max Seq Len = {max_batch_seq_len}")
                writer.add_scalar('train/batch_loss', loss, epoch * len(train_batch_indices) + batch_idx)
                writer.add_scalar('train/batch_rmsd', rmsd, epoch * len(train_batch_indices) + batch_idx)
                writer.add_scalar('train/max_seq_len', max_batch_seq_len, epoch * len(train_batch_indices) + batch_idx)

        # Training epoch summary
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_rmsd = sum(train_rmsds) / len(train_rmsds)

        # Validation
        rng, subkey = jax.random.split(rng)
        val_batch_indices = create_batch_indices(
            len(data['valid']['X']),
            BATCH_SIZE,
            subkey
        )

        val_losses = []
        val_rmsds = []

        for batch_indices in tqdm(val_batch_indices, desc="Validating"):
            # Get maximum sequence length in this batch
            max_batch_seq_len = get_max_sequence_len(batch_indices, data['valid'])
            max_batch_seq_len = min(max_batch_seq_len, max_model_len)

            # Adjust batch size dynamically based on sequence length
            dynamic_batch_size = get_dynamic_batch_size(max_batch_seq_len)

            # If current batch is too large, split it further
            if len(batch_indices) > dynamic_batch_size:
                sub_batches = [batch_indices[i:i+dynamic_batch_size]
                              for i in range(0, len(batch_indices), dynamic_batch_size)]
            else:
                sub_batches = [batch_indices]

            # Process each sub-batch
            for sub_batch_indices in sub_batches:
                # Prepare batch data with optimal padding for this specific batch
                tokens_batch = jnp.array(data['valid']['X'][sub_batch_indices])[:, :max_batch_seq_len]
                msa_batch = extract_msa_conservation(data['valid']['msas'], sub_batch_indices)[:, :max_batch_seq_len]
                bppms_batch = jnp.array(load_bppms_batch(data['valid']['bppms_file'], sub_batch_indices))[:, :max_batch_seq_len, :max_batch_seq_len]
                coords_batch = jnp.array(data['valid']['y'][sub_batch_indices])[:, :max_batch_seq_len]

                batch = (tokens_batch, msa_batch, bppms_batch, coords_batch)

                # Handle long sequences with chunking for validation
                if max_batch_seq_len > 1024:
                    chunks = chunk_long_sequence_with_overlap(*batch, window_size=1024, overlap=256)
                    chunk_losses = []
                    chunk_rmsds = []
                    for chunk_batch in chunks:
                        chunk_loss, chunk_rmsd = eval_step(state, chunk_batch)
                        chunk_losses.append(chunk_loss)
                        chunk_rmsds.append(chunk_rmsd)
                    loss = sum(chunk_losses) / len(chunk_losses)
                    rmsd = sum(chunk_rmsds) / len(chunk_rmsds)
                else:
                    loss, rmsd = eval_step(state, batch)

                val_losses.append(loss)
                val_rmsds.append(rmsd)

        # Validation epoch summary
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_rmsd = sum(val_rmsds) / len(val_rmsds)

        # Log epoch metrics
        epoch_time = time.time() - epoch_start
        log_message(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        log_message(f"Train Loss: {avg_train_loss:.4f}, Train RMSD: {avg_train_rmsd:.4f}")
        log_message(f"Valid Loss: {avg_val_loss:.4f}, Valid RMSD: {avg_val_rmsd:.4f}")

        writer.add_scalar('train/loss', avg_train_loss, epoch+1)
        writer.add_scalar('train/rmsd', avg_train_rmsd, epoch+1)
        writer.add_scalar('valid/loss', avg_val_loss, epoch+1)
        writer.add_scalar('valid/rmsd', avg_val_rmsd, epoch+1)

        # Save checkpoint
        if avg_val_rmsd < best_val_rmsd:
            best_val_rmsd = avg_val_rmsd
            checkpoint = {
                'params': state.params,
                'step': epoch,
                'rmsd': avg_val_rmsd
            }
            # Save best checkpoint to file
            import pickle
            with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
                pickle.dump(checkpoint, f)
            log_message(f"New best model saved with RMSD: {best_val_rmsd:.4f}")

        # Save the latest model
        checkpoint = {
            'params': state.params,
            'step': epoch,
            'rmsd': avg_val_rmsd
        }
        import pickle
        with open(os.path.join(OUTPUT_DIR, 'latest_model.pkl'), 'wb') as f:
            pickle.dump(checkpoint, f)

    # Training complete
    total_time = time.time() - start_time
    log_message(f"Training completed in {total_time:.2f}s")
    log_message(f"Best validation RMSD: {best_val_rmsd:.4f}")

    # Close writer
    writer.close()


if __name__ == "__main__":
    # Configure JAX
    jax.config.update('jax_platform_name', 'gpu')  # Use GPU if available

    # Print device info
    log_message(f"JAX devices: {jax.devices()}")

    train_model()
