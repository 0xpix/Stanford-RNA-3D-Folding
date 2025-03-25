import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from flax.training import train_state
from src.model.model1.model import RibonanzaNetJAX, jax_params
from src.utils import log_message

# 🔹 Load Processed Data
log_message("🧬 Loading processed data")
with open("data/processed/processed_data.pkl", "rb") as f:
    X_train, y_train, X_valid, y_valid, max_len = pickle.load(f)

# 🔹 Initialize JAX Model
log_message("🧠 Initializing model")
rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, max_len))  # Adjust to match input size

jax_model = RibonanzaNetJAX()
params = jax_model.init(rng, dummy_input)

# Load pre-trained PyTorch weights into JAX
params = jax.tree_map(lambda p, j: jnp.array(p) if j in jax_params else j, params, jax_params)

# 🔹 Define Loss Function (MSE for (X, Y, Z) coordinates)
def loss_fn(params, x, y):
    pred = jax_model.apply(params, x)
    return jnp.mean(jnp.square(pred - y))  # MSE Loss

# 🔹 Optimizer & Training State
tx = optax.adam(learning_rate=1e-3)
state = train_state.TrainState.create(
    apply_fn=jax_model.apply,
    params=params,
    tx=tx,
)

# 🔹 Training Step (JIT Compiled for Speed)
@jax.jit
def train_step(state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 🔹 Validation Step (No Dropout)
@jax.jit
def eval_step(params, x, y):
    pred = jax_model.apply(params, x)
    return jnp.mean(jnp.square(pred - y))  # MSE Loss

# 🔹 Training Loop
batch_size = 32
epochs = 50
num_batches = len(X_train) // batch_size

log_message("🚀 Starting training loop")
for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    epoch_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        state, loss = train_step(state, batch_x, batch_y)
        epoch_loss += loss

    epoch_loss /= num_batches

    # Compute validation loss
    val_loss = eval_step(state.params, X_valid, y_valid)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# 🔹 Save Trained Model
with open("model.pkl", "wb") as f:
    pickle.dump(state.params, f)

print("✅ Model training complete! Saved as 'model.pkl' 🎉")
