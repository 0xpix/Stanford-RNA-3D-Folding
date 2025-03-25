import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import time
import os
from tqdm import tqdm
from flax.training import train_state
from flax.training import checkpoints

from src.model.model import ProteinTransformer
from src.utils import log_message

# ---------- Training Functions ---------


def mse_loss(pred, target):
    """Mean squared error loss function with better numerical stability."""
    # Mask NaN values if they exist
    pred = jnp.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extreme values
    pred = jnp.clip(pred, -1e6, 1e6)
    # Calculate loss with better epsilon
    loss = jnp.mean((pred - target) ** 2 + 1e-10)
    return loss


@jax.jit
def train_step(state, batch_x_combined, batch_y, rng):
    """Single training step with gradient checking."""

    def loss_fn(params):
        pred = state.apply_fn(
            {"params": params},
            batch_x_combined,
            rngs={"dropout": rng},
            deterministic=False,
        )
        return mse_loss(pred, batch_y)

    grads = jax.grad(loss_fn)(state.params)

    # Check for NaN or infinity in gradients and replace with zeros
    grads = jax.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
    )

    loss = loss_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss, grads


@jax.jit
def eval_step(state, batch_x_combined, batch_y):
    """Evaluation step."""
    pred = state.apply_fn(
        {"params": state.params}, batch_x_combined, deterministic=True
    )
    return mse_loss(pred, batch_y)


def log_gradient_stats(grads, prefix=""):
    """Log gradient statistics for debugging."""
    stats_message = ["Gradient Statistics:"]

    def collect_grad_stats(grads, prefix=""):
        for k, v in grads.items():
            if isinstance(v, dict):
                collect_grad_stats(v, prefix=f"{prefix}{k}.")
            else:
                mean_val = float(jnp.mean(v))
                std_val = float(jnp.std(v))
                min_val = float(jnp.min(v))
                max_val = float(jnp.max(v))
                stats_message.append(
                    f"{prefix}{k}: Mean={mean_val:.6f}, Std={std_val:.6f}, Min={min_val:.6f}, Max={max_val:.6f}"
                )

    collect_grad_stats(grads, prefix)
    log_message("\n".join(stats_message))


def save_model(state, save_dir, epoch=None):
    """Save model checkpoint with absolute path."""
    try:
        # Convert to absolute path
        abs_save_dir = os.path.abspath(save_dir)
        os.makedirs(abs_save_dir, exist_ok=True)

        if epoch is not None:
            checkpoints.save_checkpoint(abs_save_dir, state, epoch, keep=3)
            log_message(f"Saved model checkpoint for epoch {epoch} to {abs_save_dir}")
        else:
            checkpoints.save_checkpoint(abs_save_dir, state, 0, keep=1, overwrite=True)
            log_message(f"Saved final model checkpoint to {abs_save_dir}")
    except Exception as e:
        log_message(f"Error saving model: {str(e)}", level="ERROR")


def train_model(data_path, model_dir, config=None):
    """Main training function."""
    if config is None:
        config = {
            "batch_size": 32,
            "num_epochs": 4,
            "learning_rate": 5e-5,
            "eval_every": 1,
            "save_every": 1,
        }

    log_message("Starting training with configuration: " + str(config))

    # Data preparation
    data_file = "data/processed/preprocessed_data_final.pkl"
    if not os.path.exists(data_file):
        log_message("Loading data...")
        # _, y_train, _ = prepare_data(data_path)
        return None
    else:
        with open(data_file, "rb") as f:
            X_train, y_train, X_eval, y_eval = pickle.load(f)
    # ---------- Model Initialization ---------
    log_message("Initializing model...")
    model = ProteinTransformer(max_len=X_train.shape[1])
    rng = jax.random.PRNGKey(0)
    init_rng, dropout_rng = jax.random.split(rng)

    dummy_x_combined = X_train[:1]
    params = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_x_combined)[
        "params"
    ]

    # Set up optimizer with better stability
    learning_rate = config["learning_rate"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Stronger gradient clipping
        optax.scale_by_adam(
            b1=0.9, b2=0.999, eps=1e-8
        ),  # Adam optimizer with good defaults
        optax.scale(-learning_rate),  # Learning rate
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    log_message(f"Model initialized with {param_count:,} parameters")

    # ---------- Training Loop ---------
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    log_message(
        f"Starting training with batch size {batch_size} for {num_epochs} epochs..."
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()
        log_message(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Shuffle training data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]

        num_complete_batches = len(X_train) // batch_size
        epoch_loss = 0.0
        batch_losses = []

        with tqdm(total=num_complete_batches, desc=f"Training epoch {epoch+1}") as pbar:
            for i in range(0, len(X_train) - batch_size + 1, batch_size):
                batch_x_combined = X_train_shuffled[i : i + batch_size]
                batch_y = y_train_shuffled[i : i + batch_size]

                rng, dropout_rng = jax.random.split(rng)
                state, train_loss, batch_grads = train_step(
                    state, batch_x_combined, batch_y, dropout_rng
                )

                # Save batch gradients for analysis
                if i == 0:
                    epoch_grads = batch_grads

                batch_loss = float(train_loss)
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss
                avg_loss = epoch_loss / (pbar.n + 1)

                pbar.set_postfix({"batch_loss": batch_loss, "avg_loss": avg_loss})
                pbar.update(1)

        # Log gradient statistics
        log_message(
            f"Epoch {epoch+1} average training loss: {epoch_loss / num_complete_batches:.6f}"
        )
        log_gradient_stats(epoch_grads)

        # Check for NaN loss and break if found
        if jnp.isnan(epoch_loss):
            log_message("Detected NaN loss, stopping training early", level="WARNING")
            break

        # Evaluation
        if (epoch + 1) % config["eval_every"] == 0:
            log_message("Running evaluation...")
            eval_losses = []

            # Process evaluation data in batches
            eval_batch_size = min(batch_size, len(X_eval))
            for i in range(0, len(X_eval) - eval_batch_size + 1, eval_batch_size):
                eval_X_batch = X_eval[i : i + eval_batch_size]
                eval_y_batch = y_eval[i : i + eval_batch_size]
                eval_loss = float(eval_step(state, eval_X_batch, eval_y_batch))
                eval_losses.append(eval_loss)

            avg_eval_loss = np.mean(eval_losses) if eval_losses else 0.0
            log_message(f"Epoch {epoch+1} evaluation loss: {avg_eval_loss:.6f}")

        # Save model checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            save_model(state, model_dir, epoch + 1)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        log_message(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    # Save final model
    save_model(state, model_dir)
    log_message("Training completed!")

    return state


if __name__ == "__main__":
    # Set configuration variables directly
    data_path = "data/processed/preprocessed_data.pkl"  # Set this to your data path
    model_dir = "./models"  # Output directory for model

    config = {
        "batch_size": 32,
        "num_epochs": 4,
        "learning_rate": 5e-5,
        "eval_every": 1,
        "save_every": 1,
    }

    log_message("Starting training with direct configuration")
    train_model(data_path, model_dir, config)
