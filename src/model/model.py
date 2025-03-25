"""Transformer model for RNA 3D structure prediction."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Any, Callable
import optax
from flax.training import train_state

from src.utils.utils import log_message


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    d_model: int
    max_len: int = 5000  # Increased from 500 to accommodate longer sequences

    def setup(self):
        # Create constant positional encoding
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
        )

        # Calculate positional encoding
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Register as a parameter that doesn't require gradients
        self.pe = pe

    def __call__(self, x):
        seq_len = x.shape[1]
        # Ensure we have enough positional encodings for the sequence length
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        return x + self.pe[:seq_len]


class TransformerEncoderBlock(nn.Module):
    """A single transformer encoder block."""

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Multi-head attention
        # Pass deterministic parameter directly to MultiHeadAttention
        attn_output = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, x, x, deterministic=not training)  # Pass deterministic here

        # Residual connection and layer normalization
        x = x + nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.LayerNorm()(x)

        # Feed-forward network - Update to use nn.Sequential properly
        # Sequential in Flax doesn't handle deterministic parameter automatically
        ff = nn.Dense(self.d_ff)(x)
        ff = nn.relu(ff)
        ff = nn.Dropout(rate=self.dropout_rate)(ff, deterministic=not training)
        ff = nn.Dense(self.d_model)(ff)
        ff_output = nn.Dropout(rate=self.dropout_rate)(ff, deterministic=not training)

        # Residual connection and layer normalization
        x = x + ff_output
        x = nn.LayerNorm()(x)

        return x


class RNATransformerEncoder(nn.Module):
    """Transformer encoder for RNA 3D structure prediction."""

    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    num_features: int = 2  # Sequence and structure features
    max_seq_len: int = 5000  # Increased from 500 to accommodate longer sequences
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Create embeddings for sequence and structure
        # Input x shape: (batch_size, seq_len, 2)
        # 2 represents (sequence, structure) features

        # Embedding layer
        x = nn.Dense(self.d_model)(x)

        # Add positional encoding
        pos_encoding = PositionalEncoding(
            d_model=self.d_model, max_len=self.max_seq_len
        )
        x = pos_encoding(x)

        # Apply dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # Transformer encoder blocks
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
            )(x, training=training)

        # Output projection to 3D coordinates (x, y, z)
        output = nn.Dense(3)(x)

        return output


class RNAFoldingModel:
    """RNA 3D folding model with training and evaluation functionality."""

    def __init__(self, config):
        self.config = config
        self.model = RNATransformerEncoder(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_features=2,  # Sequence and structure
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
        )
        log_message(
            f"Initialized RNA folding model with {config.num_layers} transformer layers"
        )

    def create_train_state(self, rng, input_shape, learning_rate=1e-3):
        """Create initial training state with model parameters and optimizer."""
        params = self.model.init(rng, jnp.ones(input_shape), training=True)
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    @staticmethod
    def compute_loss(params, model, inputs, targets, training=True):
        """Compute RMSD loss between predicted and actual coordinates."""
        # Apply mask to only consider non-padded positions
        # Assuming 0 coordinates are padding
        mask = (targets != 0).any(axis=-1)

        # Forward pass
        predictions = model.apply(params, inputs, training=training)

        # Compute squared error with masking
        squared_error = jnp.sum((predictions - targets) ** 2, axis=-1) * mask

        # Compute RMSD for non-padded positions
        rmsd = jnp.sqrt(jnp.sum(squared_error) / jnp.sum(mask))

        return rmsd

    @staticmethod
    def train_step(state, batch):
        """Execute one training step."""
        inputs, targets = batch

        # Define gradient function
        def loss_fn(params):
            return RNAFoldingModel.compute_loss(
                params, state.apply_fn, inputs, targets, training=True
            )

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        return new_state, loss

    @staticmethod
    def eval_step(state, batch):
        """Execute one evaluation step."""
        inputs, targets = batch
        loss = RNAFoldingModel.compute_loss(
            state.params, state.apply_fn, inputs, targets, training=False
        )
        return loss

    def train(self, train_data, eval_data, num_epochs, batch_size, seed=42):
        """Train the model on RNA data."""
        rng = jax.random.PRNGKey(seed)

        # Extract training data
        X_train, y_train = train_data
        X_eval, y_eval = eval_data

        # Initialize training state
        input_shape = (batch_size, X_train.shape[1], X_train.shape[2])
        rng, init_rng = jax.random.split(rng)
        state = self.create_train_state(
            init_rng, input_shape, learning_rate=self.config.learning_rate
        )

        # Training loop
        num_batches = len(X_train) // batch_size
        log_message(
            f"Starting training for {num_epochs} epochs with {num_batches} batches per epoch"
        )

        for epoch in range(num_epochs):
            # Shuffle training data
            rng, shuffle_rng = jax.random.split(rng)
            perm = jax.random.permutation(shuffle_rng, len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            # Training
            total_loss = 0.0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_X = X_train_shuffled[start_idx:end_idx]
                batch_y = y_train_shuffled[start_idx:end_idx]

                state, loss = RNAFoldingModel.train_step(state, (batch_X, batch_y))
                total_loss += loss

            # Evaluation
            eval_loss = 0.0
            eval_batches = len(X_eval) // batch_size
            for batch_idx in range(eval_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_X = X_eval[start_idx:end_idx]
                batch_y = y_eval[start_idx:end_idx]

                loss = RNAFoldingModel.eval_step(state, (batch_X, batch_y))
                eval_loss += loss

            avg_train_loss = total_loss / num_batches
            avg_eval_loss = eval_loss / eval_batches

            log_message(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}"
            )

        return state

    def predict(self, state, sequences):
        """Generate 3D structure predictions for RNA sequences."""
        return state.apply_fn(state.params, sequences, training=False)


# Example model configuration
class ModelConfig:
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 4
    max_seq_len = 5000  # Increased from 500 to accommodate longer sequences
    dropout_rate = 0.1
    learning_rate = 1e-3


if __name__ == "__main__":
    import pickle
    from src.utils.utils import check_jax_device

    # Check JAX device
    check_jax_device()

    # Load preprocessed data
    log_message("Loading preprocessed data")
    with open("data/processed/preprocessed_data_final.pkl", "rb") as f:
        X_train, y_train, X_eval, y_eval = pickle.load(f)

    # Configure and initialize model
    config = ModelConfig()
    model = RNAFoldingModel(config)

    # Train model
    log_message("Starting model training")
    batch_size = 32
    num_epochs = 10

    # Train the model
    trained_state = model.train(
        train_data=(X_train, y_train),
        eval_data=(X_eval, y_eval),
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Save model state
    with open("models/rna_transformer_model.pkl", "wb") as f:
        pickle.dump(trained_state, f)

    log_message("Model training complete and saved")
