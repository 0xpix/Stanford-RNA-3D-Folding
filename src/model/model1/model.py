import jax
import jax.numpy as jnp
from jax import random
from flash_attention_jax import flash_attention
from multimolecule.models import RibonanzaNetModel
from flax import linen as nn
from typing import Optional, Any, Dict

from src.utils import log_message

# Load the pretrained RibonanzaNet PyTorch model
log_message("Loading PyTorch model")
model = RibonanzaNetModel.from_pretrained("multimolecule/ribonanzanet")
model.eval()

# Check model structure
print("PyTorch Model Structure:", model)
print("\n")
log_message("Converting PyTorch model to JAX format")

# Convert PyTorch weights to NumPy
numpy_params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

# Convert NumPy to JAX arrays
jax_params = {k: jnp.array(v) for k, v in numpy_params.items()}

# Print parameter names
print("Converted parameters:", jax_params.keys())


class FlashSelfAttention(nn.Module):
    """Self-attention using Flash Attention 2 algorithm from flash_attention_jax."""
    num_heads: int
    dropout_rate: float = 0.0
    kernel_init: Any = nn.linear.default_kernel_init
    bias_init: Any = nn.initializers.zeros
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=None):
        """Applies FlashSelfAttention on the input data.

        Args:
            inputs: Input data with shape [batch, sequence, features].
            mask: Attention mask with shape [batch, sequence].
            deterministic: Run deterministically if True.

        Returns:
            Output with shape [batch, sequence, features].
        """
        batch, seq_len, features = inputs.shape
        head_dim = features // self.num_heads

        dense = lambda x: nn.Dense(
            features=x,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
        )

        # Create query, key, value projections
        query_dense = dense(features)
        key_dense = dense(features)
        value_dense = dense(features)

        query = query_dense(inputs)
        key = key_dense(inputs)
        value = value_dense(inputs)

        # Reshape to format expected by flash_attention: [batch, heads, seq, dim]
        query = query.reshape(batch, seq_len, self.num_heads, head_dim)
        query = jnp.transpose(query, (0, 2, 1, 3))

        key = key.reshape(batch, seq_len, self.num_heads, head_dim)
        key = jnp.transpose(key, (0, 2, 1, 3))

        value = value.reshape(batch, seq_len, self.num_heads, head_dim)
        value = jnp.transpose(value, (0, 2, 1, 3))

        # Apply flash attention
        attention_output, _ = flash_attention(query, key, value, mask)

        # Reshape back to [batch, seq, features]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch, seq_len, features)

        # Final projection
        return dense(features)(attention_output)

class RibonanzaNetJAX(nn.Module):
    """ JAX equivalent of RibonanzaNet """
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 9
    intermediate_size: int = 1024
    vocab_size: int = 16  # Check tokenizer for actual size
    dropout_rate: float = 0.1

    def setup(self):
        self.embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)
        self.encoder_layers = [
            FlashSelfAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        self.layer_norms = [nn.LayerNorm() for _ in range(self.num_layers)]
        self.intermediate_fc = nn.Dense(self.intermediate_size)
        self.output_fc = nn.Dense(self.hidden_size)

    def __call__(self, x, training=False):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Create attention mask (1 for tokens to attend to, 0 for padding)
        # This is a simple version - you might need to adjust based on your tokenization
        mask = jnp.ones((batch_size, seq_len))

        x = self.embed(x)

        # Pass through Transformer layers with residual connections and layer norm
        for i, (attention, layer_norm) in enumerate(zip(self.encoder_layers, self.layer_norms)):
            residual = x
            x = layer_norm(x)
            x = attention(x, mask=mask, deterministic=not training)
            x = x + residual  # Residual connection

        # Final layers
        x = self.intermediate_fc(x)
        x = nn.relu(x)
        x = self.output_fc(x)
        return x

# Initialize JAX model
log_message("Initializing JAX model")
rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 128))  # Adjust based on expected input length

jax_model = RibonanzaNetJAX()
params = jax_model.init(rng, dummy_input)

# Map PyTorch weights to JAX model
def load_weights(jax_params, torch_params):
    """ Maps PyTorch parameters to JAX model """
    jax_params = jax.tree_map(lambda p, j: jnp.array(p) if j in torch_params else j, jax_params, torch_params)
    return jax_params

jax_params = load_weights(params, jax_params)
