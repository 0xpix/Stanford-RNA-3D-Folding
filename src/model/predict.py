# predict.py
import jax.numpy as jnp
import numpy as np
import pickle
import pandas as pd
from model import RNAClassifier

from src.preprocess.preprocess import encode_sequence

# 🔹 Load trained model
with open("model.pkl", "rb") as f:
    params = pickle.load(f)

# 🔹 Load test data
data = pd.read_csv("dataset.csv")
X_test = np.array([encode_sequence(seq) for seq in data["sequence"]])
X_test = jnp.array(X_test)

# 🔹 Initialize model
model = RNAClassifier(num_classes=2)

# 🔹 Make predictions
logits = model.apply(params, X_test)
predictions = jnp.argmax(logits, axis=1)

# 🔹 Output results
for seq, pred in zip(data["sequence"], predictions):
    print(f"Sequence: {seq} -> Predicted Class: {pred}")
