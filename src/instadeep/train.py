import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from typing import Optional  # ✅ Add this at the top of the file
from flax.training import train_state
from pretrained import get_pretrained_model


# 🔹 Load processed data
print("🧬 Loading processed data")
with open("../../data/processed/processed_data.pkl", "rb") as f:
    X_train, y_train, X_valid, y_valid, _ = pickle.load(f)
print(f"🧬 Raw X_train shape: {X_train.shape}")

# 🔹 Load Pretrained Model
print("🔬 Loading pre-trained BulkRNABert model")
params, forward_fn, tokenizer, config = get_pretrained_model("bulk_rna_bert_tcga")

# 🔹 Tokenization
EXPECTED_GENES = 19062  # ✅ Ensure this matches model config
current_genes = X_train.shape[1]

if y_train.shape[1] < EXPECTED_GENES:
    print(f"⚠️ Padding y_train from {y_train.shape[1]} → {EXPECTED_GENES}")
    y_padding = np.zeros((y_train.shape[0], EXPECTED_GENES - y_train.shape[1], y_train.shape[2]))
    y_train = np.hstack((y_train, y_padding))

if y_valid.shape[1] < EXPECTED_GENES:
    print(f"⚠️ Padding y_valid from {y_valid.shape[1]} → {EXPECTED_GENES}")
    y_padding = np.zeros((y_valid.shape[0], EXPECTED_GENES - y_valid.shape[1], y_valid.shape[2]))
    y_valid = np.hstack((y_valid, y_padding))

if current_genes < EXPECTED_GENES:
    print(f"⚠️ Padding X_train from {current_genes} → {EXPECTED_GENES}")
    padding = np.zeros((X_train.shape[0], EXPECTED_GENES - current_genes))
    X_train = np.hstack((X_train, padding))

if current_genes < EXPECTED_GENES:
    print(f"⚠️ Padding X_valid from {current_genes} → {EXPECTED_GENES}")
    padding = np.zeros((X_valid.shape[0], EXPECTED_GENES - current_genes))
    X_valid = np.hstack((X_valid, padding))

print(f"✅ Adjusted X_train shape: {X_train.shape}")
print(f"✅ Adjusted X_valid shape: {X_valid.shape}")

# Now tokenize
X_train_tokens = tokenizer.batch_tokenize(X_train)
X_valid_tokens = tokenizer.batch_tokenize(X_valid)

print("🔡 Tokenizing RNA sequences")
X_train_tokens = tokenizer.batch_tokenize(X_train)
X_valid_tokens = tokenizer.batch_tokenize(X_valid)


# 🔹 Optimizer & Training State
tx = optax.adam(learning_rate=1e-3)  # ✅ Adam optimizer

class TrainState(train_state.TrainState):
    batch_stats: Optional[dict] = None  # ✅ Make it optional


# Initialize training state
print("🛠 Initializing training state")
state = TrainState.create(
    apply_fn=forward_fn,
    params=params,
    tx=tx,
    batch_stats={},
)

# 🔹 Loss function
def loss_fn(params, x, y):
    logits = forward_fn.apply(params, None, x)["logits"]
    logits = logits[:, :y.shape[1], :y.shape[2]]  # Ensure shape matches y_train
    loss = jnp.mean(jnp.square(logits - y))  # Mean Squared Error
    return loss

# 🔹 Training step (JIT compiled for speed)
@jax.jit
def train_step(state, x, y):
    (loss, grads) = jax.value_and_grad(loss_fn)(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 🔹 Validation step
@jax.jit
def eval_step(params, x, y):
    logits = forward_fn.apply(params, None, x)["logits"]
    logits = logits[:, :y.shape[1], :y.shape[2]]  # Ensure shape matches y_train
    return jnp.mean(jnp.square(logits - y))

# 🔹 Training loop
batch_size = 4
epochs = 2
num_batches = len(X_train_tokens) // batch_size

print("🚀 Starting training loop")
for epoch in range(epochs):
    indices = np.random.permutation(len(X_train_tokens))
    X_train_tokens, y_train = X_train_tokens[indices], y_train[indices]

    epoch_loss = 0
    for i in range(num_batches):
        batch_x = X_train_tokens[i * batch_size : (i + 1) * batch_size]
        batch_y = y_train[i * batch_size : (i + 1) * batch_size]
        state, loss = train_step(state, batch_x, batch_y)
        epoch_loss += loss

    epoch_loss /= num_batches
    val_loss = eval_step(state.params, X_valid_tokens, y_valid)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# 🔹 Save trained model
with open("trained_bulk_rna_bert.pkl", "wb") as f:
    pickle.dump(state.params, f)
print("✅ Model saved as 'trained_bulk_rna_bert.pkl'!")
