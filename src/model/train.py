# train.py
import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from flax.training import train_state
from src.model.model import CNNRNAFolding
from src.utils import log_message

# 🔹 Load processed data
log_message("🧬 Loading processed data")
with open("data/processed/processed_data.pkl", "rb") as f:
    X_train, y_train, X_valid, y_valid, max_len = pickle.load(f)

# 🔹 Model Initialization
log_message("🧠 Initializing model")
model = CNNRNAFolding(max_len=max_len)

rng = jax.random.PRNGKey(42)
variables = model.init(rng, X_train[0])  # Initialize model with dummy input
params = variables["params"]  # ✅ Extract params
batch_stats = variables["batch_stats"]  # ✅ Extract batch_stats for BatchNorm

# 🔹 Optimizer & Training State
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # ✅ Clip gradients to prevent explosion
    optax.adam(learning_rate=1e-3)   # ✅ Use Adam optimizer
)

log_message(f"Initializing optimizer and training state: {tx.__class__.__name__}")
class TrainState(train_state.TrainState):
    batch_stats: dict  # Store batch stats separately
    dropout_rng: jax.random.PRNGKey

# 🔹 Initialize training state
log_message("Initializing training state")
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    batch_stats=batch_stats,  # ✅ Store batch_stats
    dropout_rng=rng,
)

# 🔹 Loss function (Mean Squared Error for coordinate predictions)
def loss_fn(params, batch_stats, x, y, train, rng):
    variables = {"params": params, "batch_stats": batch_stats}  
    logits, updated_state = model.apply(variables, x, train=train, rngs={'dropout': rng}, mutable=["batch_stats"])
    loss = jnp.mean(jnp.square(logits - y) + 1e-6)  # ✅ Add small epsilon to prevent extreme loss values
    return loss, updated_state["batch_stats"]


# 🔹 Training step (JIT compiled for speed)
@jax.jit
def train_step(state, x, y):
    rng, new_rng = jax.random.split(state.dropout_rng)
    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.batch_stats, x, y, train=True, rng=rng
    )
    state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats, dropout_rng=new_rng)
    return state, loss

# 🔹 Validation step (No dropout, inference mode)
@jax.jit
def eval_step(params, batch_stats, x, y):
    variables = {"params": params, "batch_stats": batch_stats}  
    logits = model.apply(variables, x, train=False)  # ✅ Ensure stable inference
    return jnp.mean(jnp.square(logits - y))

# 🔹 Training loop
batch_size = 32
epochs = 200
num_batches = len(X_train) // batch_size

log_message("🚀 Starting training loop")
for epoch in range(epochs):
    # Shuffle dataset
    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    # Training loop per batch
    epoch_loss = 0
    for i in range(num_batches):
        batch_x = X_train[i * batch_size : (i + 1) * batch_size]
        batch_y = y_train[i * batch_size : (i + 1) * batch_size]
        state, loss = train_step(state, batch_x, batch_y)
        epoch_loss += loss

    epoch_loss /= num_batches

    # Compute validation loss
    val_loss = eval_step(state.params, state.batch_stats, X_valid, y_valid)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# 🔹 Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump({"params": state.params, "batch_stats": state.batch_stats}, f)
print("✅ Model saved as 'model.pkl'!")
