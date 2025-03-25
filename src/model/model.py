import jax.numpy as jnp
import flax.linen as nn
from flax.linen import remat

# Define hyperparameters
vocab_size = 5  # 4 RNA bases (A, U, C, G) + 1 for padding (0)
embedding_dim = 16
num_filters = 64
kernel_size = 3
drop_rate = 0.2


class TransformerBlock(nn.Module):
    """A single transformer block implemented as a Module."""

    d_model: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        # Self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(inputs, inputs)
        x = inputs + nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
            attn_output
        )
        x = nn.LayerNorm()(x)

        # Feed-forward
        ff_output = nn.Dense(features=self.d_model * 4)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
            ff_output
        )
        ff_output = nn.Dense(features=self.d_model * 2)(ff_output)
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
            ff_output
        )
        x = nn.LayerNorm()(x)
        return x


class ProteinTransformer(nn.Module):
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    max_len: int = 512

    @nn.compact
    def __call__(self, x_combined, deterministic=True):
        # Split combined input into sequences and structures
        x_seq = x_combined[:, :, 0].astype(jnp.int32)
        x_struct = x_combined[:, :, 1].astype(jnp.int32)

        x_seq_embed = nn.Embed(num_embeddings=5, features=self.d_model)(x_seq)
        x_struct_embed = nn.Embed(num_embeddings=3, features=self.d_model)(x_struct)
        x_combined_embed = jnp.concatenate([x_seq_embed, x_struct_embed], axis=-1)

        positions = jnp.arange(self.max_len)[None, :]
        pos_embed = nn.Embed(num_embeddings=self.max_len, features=self.d_model * 2)(
            positions
        )
        x = x_combined_embed + pos_embed
        x = nn.LayerNorm()(x)  # Initial normalization

        # Apply transformer blocks
        for i in range(self.num_layers):
            # Create a non-remat version first
            block = TransformerBlock(
                d_model=self.d_model * 2,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i}",
            )

            # Create a custom function that bakes in the deterministic parameter
            if deterministic:
                # For evaluation (deterministic=True)
                block_fn = lambda x: block(x, deterministic=True)
            else:
                # For training (deterministic=False)
                block_fn = lambda x: block(x, deterministic=False)

            # Apply remat to this function
            remat_block = remat(block_fn)
            x = remat_block(x)

        # Final projection
        x = nn.LayerNorm()(x)
        coords = nn.Dense(
            features=3,
            kernel_init=nn.initializers.variance_scaling(
                scale=0.02, mode="fan_in", distribution="truncated_normal"
            ),
        )(x)

        return coords
