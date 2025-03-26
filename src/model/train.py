"""
Training script for RNA 3D structure prediction model.
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model.model import RNAFoldingModel, ModelConfig
from src.utils.utils import log_message, check_jax_device
from flax.training import train_state  # Import train_state


# Data parameters
DATA_PATH = "data/processed/preprocessed_data_final.pkl"

# Model parameters
D_MODEL = 16
NUM_HEADS = 1
D_FF = 512
NUM_LAYERS = 4
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 1e-3
SEED = 42
EVAL_EVERY = 1

# Output parameters
OUTPUT_DIR = "models"
MODEL_NAME = None  # Will use timestamp if None


def load_data(data_path):
    """Load preprocessed RNA folding data."""
    log_message(f"Loading preprocessed data from {data_path}")

    try:
        with open(data_path, "rb") as f:
            X_train, y_train, X_eval, y_eval = pickle.load(f)

        log_message(
            f"Loaded data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, "
            f"X_eval: {X_eval.shape}, y_eval: {y_eval.shape}"
        )
        return X_train, y_train, X_eval, y_eval

    except Exception as e:
        log_message(f"Error loading data: {e}", level="ERROR")
        raise


def create_model_config():
    """Create model configuration from parameters."""
    config = ModelConfig()
    config.d_model = D_MODEL
    config.num_heads = NUM_HEADS
    config.d_ff = D_FF
    config.num_layers = NUM_LAYERS
    config.dropout_rate = DROPOUT
    config.learning_rate = LEARNING_RATE
    config.max_seq_len = 5000  # Increased to accommodate longer sequences

    return config


def save_training_history(history, output_path):
    """Save training history and create plots."""
    # Save history as JSON
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Create plot directory
    plot_dir = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history["epochs"], history["train_loss"], label="Training Loss")
    plt.plot(history["epochs"], history["eval_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("RMSD Loss")
    plt.title("RNA Structure Prediction Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / "training_loss.png", dpi=300)
    plt.close()

    log_message(f"Saved training history and plots to {output_path}")


def compute_loss(params, apply_fn, inputs, targets, rng=None, training=True):
    """Compute RMSD loss between predicted and actual coordinates."""
    # Apply mask to only consider non-padded positions
    # Assuming 0 coordinates are padding
    mask = (targets != 0).any(axis=-1)

    # Forward pass - include rng key for dropout
    if training and rng is not None:
        predictions = apply_fn(params, inputs, training=training, rngs={"dropout": rng})
    else:
        # For evaluation, no rng needed
        predictions = apply_fn(params, inputs, training=training)

    # Compute squared error with masking
    squared_error = jnp.sum((predictions - targets) ** 2, axis=-1) * mask

    # Compute RMSD for non-padded positions
    rmsd = jnp.sqrt(jnp.sum(squared_error) / jnp.sum(mask))

    return rmsd


def train_step(state, batch, rng):
    """Execute one training step."""
    inputs, targets = batch

    # Define gradient function
    def loss_fn(params):
        return compute_loss(
            params, state.apply_fn, inputs, targets, rng=rng, training=True
        )

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Update parameters
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss


def eval_step(state, batch):
    """Execute one evaluation step."""
    inputs, targets = batch
    loss = compute_loss(state.params, state.apply_fn, inputs, targets, training=False)
    return loss


def predict(state, sequences):
    """Generate 3D structure predictions for RNA sequences."""
    return state.apply_fn(state.params, sequences, training=False)


def save_model_params(state, output_path):
    """Save model parameters to a file (instead of pickling the entire state)."""
    with open(output_path, "wb") as f:
        pickle.dump(state.params, f)
    log_message(f"Saved model parameters to {output_path}")


def load_model_params(model, params_path, learning_rate=1e-3, rng=None):
    """Load model parameters and recreate the training state."""
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Recreate optimizer
    tx = optax.adam(learning_rate)

    # Recreate state
    return train_state.TrainState.create(
        apply_fn=model.model.apply, params=params, tx=tx
    )


def train_model():
    """Train RNA 3D folding model with given parameters."""
    # Check JAX device
    device = check_jax_device()
    log_message(f"Using device: {device}")

    # Set random seed
    rng = jax.random.PRNGKey(SEED)

    # Load data
    X_train, y_train, X_eval, y_eval = load_data(DATA_PATH)

    # Update the max_seq_len based on actual data
    max_seq_len = X_train.shape[1]
    log_message(f"Maximum sequence length in data: {max_seq_len}")

    # Create model configuration
    config = create_model_config()
    config.max_seq_len = max(config.max_seq_len, max_seq_len)
    log_message(f"Using max_seq_len: {config.max_seq_len}")

    # Initialize model
    model = RNAFoldingModel(config)
    log_message(
        f"Initialized model with {config.num_layers} transformer layers, "
        f"{config.d_model} dimensions, {config.num_heads} attention heads"
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = MODEL_NAME if MODEL_NAME else f"rna_transformer_{timestamp}"
    output_path = Path(OUTPUT_DIR) / model_name
    output_path.mkdir(exist_ok=True, parents=True)

    # Save configuration
    with open(output_path / "config.json", "w") as f:
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        json.dump(config_dict, f, indent=2)

    # Train model
    log_message(f"Starting training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    start_time = time.time()

    # Initialize training history
    history = {
        "epochs": [],
        "train_loss": [],
        "eval_loss": [],
        "learning_rate": [],
        "time_per_epoch": [],
    }

    try:
        # Initialize training state
        batch_size = min(
            BATCH_SIZE, len(X_train)
        )  # Ensure batch size isn't larger than dataset
        input_shape = (batch_size, X_train.shape[1], X_train.shape[2])
        rng, init_rng = jax.random.split(rng)
        state = model.create_train_state(
            init_rng, input_shape, learning_rate=config.learning_rate
        )

        # Training loop
        num_batches = len(X_train) // batch_size
        best_eval_loss = float("inf")

        for epoch in range(EPOCHS):
            epoch_start = time.time()

            # Shuffle training data
            rng, shuffle_rng = jax.random.split(rng)
            perm = jax.random.permutation(shuffle_rng, len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            # Training
            total_loss = 0.0
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS}"):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_X = X_train_shuffled[start_idx:end_idx]
                batch_y = y_train_shuffled[start_idx:end_idx]

                state, loss = train_step(state, (batch_X, batch_y), rng)
                total_loss += loss

            avg_train_loss = total_loss / num_batches

            # Evaluation
            if (epoch + 1) % EVAL_EVERY == 0:
                eval_loss = 0.0
                eval_batches = max(1, len(X_eval) // batch_size)

                for batch_idx in range(eval_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(X_eval))

                    # Handle last batch potentially being smaller
                    if end_idx - start_idx < batch_size:
                        continue

                    batch_X = X_eval[start_idx:end_idx]
                    batch_y = y_eval[start_idx:end_idx]

                    loss = eval_step(state, (batch_X, batch_y))
                    eval_loss += loss

                avg_eval_loss = eval_loss / eval_batches

                # Save best model
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    save_model_params(state, output_path / "best_model_params.pkl")
                    log_message(
                        f"Saved new best model with eval loss: {best_eval_loss:.4f}"
                    )
            else:
                # If not evaluating this epoch, use training loss
                avg_eval_loss = avg_train_loss

            epoch_time = time.time() - epoch_start
            log_message(
                f"Epoch {epoch+1}/{EPOCHS} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Eval Loss: {avg_eval_loss:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Update history
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(float(avg_train_loss))
            history["eval_loss"].append(float(avg_eval_loss))
            history["learning_rate"].append(float(config.learning_rate))
            history["time_per_epoch"].append(float(epoch_time))

            # Periodically save history and checkpoints
            if (epoch + 1) % 5 == 0:
                # Save checkpoint
                save_model_params(
                    state, output_path / f"checkpoint_epoch_{epoch+1}_params.pkl"
                )

                # Save current history
                save_training_history(history, output_path)

        # Save final model
        save_model_params(state, output_path / "final_model_params.pkl")

        # Save training history
        save_training_history(history, output_path)

        total_time = time.time() - start_time
        log_message(f"Training completed in {total_time:.2f} seconds")
        log_message(f"Best validation loss: {best_eval_loss:.4f}")
        log_message(f"All models and data saved to {output_path}")

        return {
            "model_path": str(output_path),
            "best_loss": float(best_eval_loss),
            "training_time": total_time,
            "final_state": state,
        }

    except Exception as e:
        log_message(f"Error during training: {e}", level="ERROR")
        # Try to save current progress if possible
        try:
            save_model_params(state, output_path / "interrupted_model_params.pkl")
            save_training_history(history, output_path)
            log_message(f"Saved interrupted model to {output_path}")
        except Exception as save_error:
            log_message(
                f"Could not save interrupted model: {save_error}", level="ERROR"
            )
        raise


def main():
    """Main function to run training."""
    # Train model directly using the defined variables
    train_model()


if __name__ == "__main__":
    main()
