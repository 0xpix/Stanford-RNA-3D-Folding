"""Transformer model for RNA 3D structure prediction."""

import flax.linen as nn
import jax.numpy as jnp
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
        )(
            x, x, x, deterministic=not training
        )  # Pass deterministic here

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
    """RNA 3D folding model - architecture only."""

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


# Example model configuration
class ModelConfig:
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 4
    max_seq_len = 5000
    dropout_rate = 0.1
    learning_rate = 1e-3
