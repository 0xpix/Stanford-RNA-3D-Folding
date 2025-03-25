import jax.numpy as jnp
from flax import linen as nn

# Constants for RNA modeling
SEQ_VOCAB_SIZE = 5  # A,C,G,U + padding
STRUCT_VOCAB_SIZE = 8  # .(,) + padding/special tokens


# Memory-efficient transformer block
class TransformerBlock(nn.Module):
    """Memory-efficient transformer block with explicit remat."""

    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Attention block (with pre-norm)
        inputs = x
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = x + inputs

        # MLP block (with pre-norm)
        inputs = x
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = nn.Dense(features=self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = x + inputs

        return x


class TransformerDecoder(nn.Module):
    """Memory-efficient transformer decoder for RNA 3D structure prediction."""

    hidden_dim: int = 128  # Reduced from 256
    num_heads: int = 4  # Reduced from 8
    num_layers: int = 4  # Reduced from 6
    mlp_dim: int = 512  # Reduced from 1024
    dropout_rate: float = 0.1
    max_len: int = 512

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        # Extract sequence and structure tokens
        seq_tokens = inputs[:, :, 0].astype(jnp.int32)
        struct_tokens = inputs[:, :, 1].astype(jnp.int32)

        # More efficient embedding with fewer dimensions
        seq_embed = nn.Embed(
            num_embeddings=SEQ_VOCAB_SIZE,
            features=self.hidden_dim // 2,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(seq_tokens)

        struct_embed = nn.Embed(
            num_embeddings=STRUCT_VOCAB_SIZE,
            features=self.hidden_dim // 2,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(struct_tokens)

        # Combine embeddings
        x = jnp.concatenate([seq_embed, struct_embed], axis=-1)

        # Add positional embeddings
        positions = jnp.arange(inputs.shape[1])[None, :]
        pos_embed = nn.Embed(
            num_embeddings=self.max_len,
            features=self.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(positions)

        x = x + pos_embed
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        # Apply transformer blocks with explicit memory optimization
        # Use functional scan for better memory efficiency through layers
        # Each layer is wrapped in a separate remat to reduce memory
        def scan_layer(x, _):
            block = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )
            block_fn = nn.remat(block) if not deterministic else block
            return block_fn(x, deterministic=deterministic), None

        # Use scan to efficiently process layers
        x, _ = nn.scan(
            scan_layer,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.num_layers,
        )(x, None)

        # Final processing
        x = nn.LayerNorm(epsilon=1e-5)(x)

        # Project to 3D coordinates with smaller initialization
        coords = nn.Dense(
            features=3,
            kernel_init=nn.initializers.variance_scaling(
                scale=0.01,  # Smaller scale
                mode="fan_in",
                distribution="truncated_normal",
            ),
        )(x)

        return coords


# For backward compatibility
class ProteinTransformer(nn.Module):
    """Legacy name for the RNA structure transformer."""

    hidden_dim: int = 128  # Reduced from 256
    num_heads: int = 4  # Reduced from 8
    num_layers: int = 4  # Reduced from 6
    dropout_rate: float = 0.1
    max_len: int = 512

    @nn.compact
    def __call__(self, x_combined, deterministic=True):
        model = TransformerDecoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.hidden_dim * 4,
            dropout_rate=self.dropout_rate,
            max_len=self.max_len,
        )
        return model(x_combined, deterministic=deterministic)
