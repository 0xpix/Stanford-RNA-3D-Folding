import jax
import jax.numpy as jnp
from multimolecule.models import RibonanzaNetModel
from flax import linen as nn

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


class RibonanzaNetJAX(nn.Module):
    """ JAX equivalent of RibonanzaNet """
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 9
    intermediate_size: int = 1024
    vocab_size: int = 16  # Check tokenizer for actual size

    def setup(self):
        self.embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)
        self.encoder_layers = [
            nn.SelfAttention(num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ]
        self.intermediate_fc = nn.Dense(self.intermediate_size)
        self.output_fc = nn.Dense(self.hidden_size)

    def __call__(self, x):
        x = self.embed(x)

        # Pass through Transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

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

