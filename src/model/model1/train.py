"""
Training script for RNA 3D Folding models.
This script handles data loading, training loop, and checkpointing.
Uses streaming HDF5 data loading to efficiently handle large datasets.
"""

import os
import time
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.random import PRNGKey
from flax.training import train_state
import optax
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path
from functools import partial
import math
from typing import Dict, List, Tuple, Optional, Any, Callable

from src.model.model1.model import RNAFoldingModel
from src.utils import log_message

# Configuration variables (defaults)
DATA_PATH = "data/processed/processed_data_final.h5"
MODEL_DIM = 256
NUM_HEADS = 8
DEPTH = 6
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 100
WARMUP_STEPS = 1
SEED = 42
OUTPUT_DIR = "checkpoints"
SAVE_EVERY = 10  # Save checkpoint every N epochs
MAX_SEQ_LEN = None  # Set to an int to limit sequence length
USE_CHECKPOINT = True  # Whether to use gradient checkpointing


class HDF5BatchLoader:
    """
    Efficiently loads batches from HDF5 files without loading the entire dataset into memory.

    Attributes:
        h5_file: Path to HDF5 file
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        drop_last: Whether to drop the last batch if it's smaller than batch_size
        rng_key: JAX random key for shuffling
    """

    def __init__(
        self,
        h5_file: str,
        batch_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng_key = PRNGKey(seed)

        # Open the file and get metadata without loading all data
        with h5py.File(self.h5_file, "r") as f:
            self.num_samples = f["X"].shape[0]
            self.max_len = f.attrs.get("max_len", f["X"].shape[1])

        log_message(
            f"HDF5BatchLoader initialized with {self.num_samples} samples, max_len = {self.max_len}"
        )

    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def get_batch_indices(self) -> List[jnp.ndarray]:
        """Generate batch indices, optionally shuffled."""
        indices = jnp.arange(self.num_samples)

        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            indices = jax.random.permutation(subkey, indices)

        # Create batches
        batch_indices = []
        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)

            # Skip last batch if it's smaller and drop_last is True
            if self.drop_last and end_idx - i < self.batch_size:
                continue

            batch_indices.append(indices[i:end_idx])

        return batch_indices

    def load_batch(self, indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Load a single batch from HDF5 file by indices.

        Args:
            indices: Array of indices to load

        Returns:
            Dictionary with batch data as JAX arrays
        """
        # Convert JAX indices to numpy array for H5PY
        np_indices = np.array(indices)

        # H5PY requires indices to be sorted in increasing order
        sorted_indices = np.sort(np_indices)

        # Create a mapping from sorted indices to original positions
        # Convert each index to a native Python int so it can be used as a dict key
        idx_map = {int(sorted_indices[i]): i for i in range(len(sorted_indices))}

        with h5py.File(self.h5_file, "r") as f:
            # Load data for sorted indices
            tokens = np.array(f["X"][sorted_indices], dtype=np.int32)
            msa = np.array(f["msa_conservation"][sorted_indices], dtype=np.float32)
            bppm = np.array(f["bppm"][sorted_indices], dtype=np.float32)
            coords = np.array(f["y"][sorted_indices], dtype=np.float32)

            # Convert to JAX arrays directly (no reordering needed if we use sorted indices)
            tokens = jnp.array(tokens, dtype=jnp.int32)
            msa = jnp.array(msa, dtype=jnp.float32)
            bppm = jnp.array(bppm, dtype=jnp.float32)
            coords = jnp.array(coords, dtype=jnp.float32)

        return {
            "tokens": tokens,
            "msa_conservation": msa,
            "bppm": bppm,
            "coords": coords,
        }

    def get_dynamic_batch_size(self, seq_len: int) -> int:
        """Determine appropriate batch size based on sequence length to manage memory usage."""
        if seq_len <= 256:
            return self.batch_size * 2  # Double batch size for short sequences
        elif seq_len <= 512:
            return self.batch_size
        elif seq_len <= 1024:
            return max(self.batch_size // 2, 4)  # Halve for medium sequences
        else:
            return max(self.batch_size // 4, 2)  # Quarter for very long sequences

    def get_max_sequence_len(self, indices: jnp.ndarray) -> int:
        """Find the maximum actual sequence length in a batch to avoid unnecessary padding."""
        # Sort indices as required by h5py
        sorted_indices = np.sort(np.array(indices))

        with h5py.File(self.h5_file, "r") as f:
            batch_tokens = f["X"][sorted_indices]

            # Find max non-padding length (first non-zero token is 0)
            max_len = 0
            for seq in batch_tokens:
                # Find where non-zero tokens end
                non_pad = np.nonzero(seq)[0]
                if len(non_pad) > 0:
                    seq_len = non_pad[-1] + 1
                    max_len = max(max_len, seq_len)

            # If all sequences are empty, use a minimum length
            if max_len == 0:
                max_len = 1

        return max_len

    def stream_batches(self, max_model_len: Optional[int] = None) -> Tuple:
        """
        Stream batches from the HDF5 file with adaptive sequence length.

        Args:
            max_model_len: Maximum sequence length for the model

        Yields:
            Tuple of (tokens, msa, bppm, coords) as JAX arrays
        """
        batch_indices = self.get_batch_indices()

        for indices in batch_indices:
            # Find the maximum actual sequence length in this batch
            max_seq_len = self.get_max_sequence_len(indices)

            # Respect model maximum length constraint if provided
            if max_model_len is not None:
                max_seq_len = min(max_seq_len, max_model_len)

            # Adjust batch size based on sequence length
            current_batch_size = len(indices)
            dynamic_batch_size = self.get_dynamic_batch_size(max_seq_len)

            # If current batch is too large for this sequence length, split it
            if current_batch_size > dynamic_batch_size:
                # Process this batch in smaller chunks
                for i in range(0, current_batch_size, dynamic_batch_size):
                    sub_indices = indices[
                        i : min(i + dynamic_batch_size, current_batch_size)
                    ]

                    # Skip if empty
                    if len(sub_indices) == 0:
                        continue

                    # Load data for this sub-batch
                    batch_data = self.load_batch(sub_indices)

                    # Trim to actual sequence length needed
                    tokens = batch_data["tokens"][:, :max_seq_len]
                    msa = batch_data["msa_conservation"][:, :max_seq_len]
                    bppm = batch_data["bppm"][:, :max_seq_len, :max_seq_len]
                    coords = batch_data["coords"][:, :max_seq_len]

                    yield tokens, msa, bppm, coords
            else:
                # Process the whole batch at once
                batch_data = self.load_batch(indices)

                # Trim to actual sequence length needed
                tokens = batch_data["tokens"][:, :max_seq_len]
                msa = batch_data["msa_conservation"][:, :max_seq_len]
                bppm = batch_data["bppm"][:, :max_seq_len, :max_seq_len]
                coords = batch_data["coords"][:, :max_seq_len]

                yield tokens, msa, bppm, coords


def create_train_state(
    rng: jnp.ndarray,
    model: RNAFoldingModel,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> train_state.TrainState:
    """Create initial training state with model parameters and optimizer."""

    # Create input shapes for model initialization
    tokens_shape = (1, 128)
    msa_conservation_shape = (1, 128)
    bppm_shape = (1, 128, 128)

    # Split PRNG key for initialization and dropout
    rng, dropout_rng = jax.random.split(rng)

    # Use model's create_train_state method if available
    if hasattr(model, "create_train_state"):
        input_shape = [tokens_shape, msa_conservation_shape, bppm_shape]
        return model.create_train_state(rng, input_shape, learning_rate=learning_rate)

    # Otherwise initialize the model manually
    # Initialize with batch inputs for compatibility with both input formats
    tokens = jnp.ones(tokens_shape, dtype=jnp.int32)
    msa = jnp.ones(msa_conservation_shape, dtype=jnp.float32)
    bppm = jnp.ones(bppm_shape, dtype=jnp.float32)

    # Try the newer input format first
    try:
        params = model.init(rng, (tokens, msa, bppm), training=False)
    except:
        # Fall back to old-style separate arguments if the above fails
        try:
            params = model.init(rng, tokens, msa, bppm, training=False)
        except Exception as e:
            log_message(f"Error initializing model: {e}")
            raise

    # Ensure warmup_steps doesn't exceed total_steps
    actual_warmup_steps = min(warmup_steps, int(0.9 * total_steps))
    if actual_warmup_steps != warmup_steps:
        log_message(
            f"WARNING: Reducing warmup steps from {warmup_steps} to {actual_warmup_steps} to avoid negative decay_steps"
        )

    # Ensure there's at least one decay step
    actual_decay_steps = max(1, total_steps - actual_warmup_steps)

    # Create a learning rate schedule
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=actual_warmup_steps,
        decay_steps=actual_decay_steps,
        end_value=learning_rate * 0.1,
    )

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Prevent exploding gradients
        optax.adamw(learning_rate=schedule_fn, weight_decay=1e-4, b1=0.9, b2=0.999),
    )

    # Create TrainState with added dropout_rng field
    class TrainStateWithRNG(train_state.TrainState):
        dropout_rng: jnp.ndarray

    return TrainStateWithRNG.create(
        apply_fn=model.apply, params=params, tx=tx, dropout_rng=dropout_rng
    )


def compute_loss(
    params: Dict,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    model: RNAFoldingModel,
    training: bool = True,
    dropout_rng: Optional[jax.random.PRNGKey] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute loss and metrics for a batch.

    Args:
        params: Model parameters
        batch: Tuple of (tokens, msa, bppm, coords)
        model: The RNAFoldingModel to use
        training: Whether this is during training
        dropout_rng: PRNG key for dropout

    Returns:
        Tuple of (loss, metrics dict)
    """
    tokens, msa, bppm, true_coords = batch

    # Create RNG dict for dropout if provided
    rngs = {}
    if dropout_rng is not None and training:
        rngs["dropout"] = dropout_rng

    # Create the input tuple for the model
    inputs = (tokens, msa, bppm)

    # Apply the model to predict coordinates
    pred_coords = model.apply({"params": params}, inputs, training=training, rngs=rngs)

    # Calculate RMSD between predicted and true coordinates
    # Flatten sequences for efficient calculation
    true_coords_flat = true_coords.reshape(-1, 3)
    pred_coords_flat = pred_coords.reshape(-1, 3)

    # Create mask for actual tokens (exclude padding)
    mask = (tokens != 0).astype(jnp.float32).reshape(-1)

    # Calculate squared error (element-wise)
    squared_error = jnp.sum((pred_coords_flat - true_coords_flat) ** 2, axis=-1)

    # Apply mask to only consider non-padding positions
    masked_se = squared_error * mask

    # Sum up and normalize
    total_valid = jnp.maximum(jnp.sum(mask), 1.0)  # Avoid division by zero
    mse = jnp.sum(masked_se) / total_valid
    rmsd = jnp.sqrt(mse)  # Root Mean Square Deviation

    # Use RMSD as the loss
    loss = rmsd

    # Collect metrics
    metrics = {
        "loss": loss,
        "rmsd": rmsd,
        "mse": mse,
    }

    return loss, metrics


def train_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    model: RNAFoldingModel,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Perform a single training step.

    Args:
        state: Current TrainState (with dropout_rng)
        batch: Tuple of (tokens, msa, bppm, coords)
        model: The RNAFoldingModel to use

    Returns:
        Tuple of (new TrainState, metrics)
    """
    # Generate a new PRNG key for dropout based on current step
    dropout_rng = None
    if hasattr(state, "dropout_rng"):
        dropout_rng = jax.random.fold_in(state.dropout_rng, state.step)

    def loss_fn(params):
        return compute_loss(
            params, batch, model, training=True, dropout_rng=dropout_rng
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update parameters and state
    new_state = state.apply_gradients(grads=grads)

    # Update dropout_rng for next step if it exists in the state
    if hasattr(state, "dropout_rng") and dropout_rng is not None:
        new_state = new_state.replace(dropout_rng=dropout_rng)

    return new_state, metrics


def save_checkpoint(state: train_state.TrainState, path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def train_model(
    train_data_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    warmup_steps: int = WARMUP_STEPS,
    model_dim: int = MODEL_DIM,
    num_heads: int = NUM_HEADS,
    depth: int = DEPTH,
    seed: int = SEED,
    save_every: int = SAVE_EVERY,
    max_seq_len: Optional[int] = MAX_SEQ_LEN,
    use_checkpoint: bool = USE_CHECKPOINT,
):
    """
    Train RNA folding model using HDF5 streaming data.

    Args:
        train_data_path: Path to training data HDF5 file
        output_dir: Directory to save checkpoints and logs
        batch_size: Base batch size (will be adjusted for sequence length)
        learning_rate: Peak learning rate for warmup schedule
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        model_dim: Model hidden dimension size
        num_heads: Number of attention heads
        depth: Number of transformer layers
        seed: Random seed for reproducibility
        save_every: Save checkpoint every N epochs
        max_seq_len: Maximum sequence length to use (None = no limit)
        use_checkpoint: Whether to use gradient checkpointing
    """
    global USE_CHECKPOINT
    USE_CHECKPOINT = use_checkpoint

    # Set up output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up logging and tensorboard
    log_message(f"Starting RNA folding model training with data from {train_data_path}")
    tb_writer = SummaryWriter(logdir=str(output_dir))

    # Initialize random key
    rng_key = PRNGKey(seed)

    # Initialize data loader
    train_loader = HDF5BatchLoader(
        train_data_path, batch_size=batch_size, shuffle=True, seed=seed
    )

    # Initialize the model
    rng_key, model_key = jax.random.split(rng_key)
    model = RNAFoldingModel(
        dim=model_dim,
        heads=num_heads,
        depth=depth,
        dropout_rate=0.1,
        use_checkpoint=use_checkpoint,
    )

    # Estimate total steps for learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs

    # Create training state
    state = create_train_state(
        model_key, model, learning_rate, warmup_steps, total_steps
    )

    # Initialize metrics
    start_time = time.time()

    # Start training
    log_message(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training loop
        train_metrics = []
        for batch_idx, batch in enumerate(
            tqdm(
                train_loader.stream_batches(max_seq_len),
                desc=f"Epoch {epoch+1}/{epochs}",
                total=len(train_loader),
            )
        ):
            # Skip empty batches
            if any(x.shape[0] == 0 for x in batch):
                continue

            # Perform training step
            state, metrics = train_step(state, batch, model)
            train_metrics.append(metrics)

            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                current_loss = metrics["loss"].item()
                current_rmsd = metrics["rmsd"].item()
                batch_size = batch[0].shape[0]
                seq_len = batch[0].shape[1]

                log_message(
                    f"Batch {batch_idx+1}: loss={current_loss:.4f}, "
                    + f"RMSD={current_rmsd:.4f} Å, "
                    + f"batch_size={batch_size}, seq_len={seq_len}"
                )

                # Add to TensorBoard
                step = epoch * steps_per_epoch + batch_idx
                tb_writer.add_scalar("train/loss", current_loss, step)
                tb_writer.add_scalar("train/rmsd", current_rmsd, step)

        # Compute epoch metrics
        train_loss = np.mean([m["loss"] for m in train_metrics])
        train_rmsd = np.mean([m["rmsd"] for m in train_metrics])

        # Log epoch results
        epoch_time = time.time() - epoch_start
        log_message(
            f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
            + f"Train: loss={train_loss:.4f}, RMSD={train_rmsd:.4f} Å"
        )

        # Add to TensorBoard
        tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
        tb_writer.add_scalar("epoch/train_rmsd", train_rmsd, epoch)

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            save_checkpoint(state, str(output_dir / f"model1_epoch_{epoch+1}.pkl"))
            log_message(f"Saved checkpoint for epoch {epoch+1}")

    # Finalize training
    total_time = time.time() - start_time
    log_message(f"Training completed in {total_time:.2f}s")

    # Save final model
    save_checkpoint(state, str(output_dir / "model1_final.pkl"))
    log_message("Saved final model checkpoint")

    # Close tensorboard writer
    tb_writer.close()

    return {
        "final_state": state,
        "training_time": total_time,
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RNA 3D Folding Model Training")

    parser.add_argument(
        "--train_data",
        type=str,
        default=DATA_PATH,
        help="Path to training data HDF5 file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save checkpoints and logs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Base batch size (will be adjusted for sequence length)",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=WARMUP_STEPS,
        help="Number of warmup steps for learning rate scheduler",
    )

    parser.add_argument(
        "--model_dim", type=int, default=MODEL_DIM, help="Model hidden dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=NUM_HEADS, help="Number of attention heads"
    )
    parser.add_argument(
        "--depth", type=int, default=DEPTH, help="Number of transformer layers"
    )

    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=SAVE_EVERY,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length to use (None = no limit)",
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        default=USE_CHECKPOINT,
        help="Use gradient checkpointing to reduce memory usage",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Check for JAX devices
    devices = jax.devices()
    log_message(f"Available JAX devices: {devices}")

    # Train the model with the provided arguments
    train_model(
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        seed=args.seed,
        save_every=args.save_every,
        max_seq_len=args.max_seq_len,
        use_checkpoint=args.use_checkpoint,
    )
