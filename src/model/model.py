import jax.numpy as jnp
import flax.linen as nn

# Define hyperparameters
vocab_size = 5  # 4 RNA bases (A, U, C, G) + 1 for padding (0)
embedding_dim = 16
num_filters = 64
kernel_size = 3
drop_rate = 0.2

class CNNRNAFolding(nn.Module):
    max_len: int  # Maximum sequence length

    def setup(self):
        # Embedding layer
        self.embedding = nn.Embed(num_embeddings=vocab_size, features=embedding_dim)

        # First convolutional block
        self.conv1 = nn.Conv(features=num_filters, kernel_size=(kernel_size,))
        self.bn1 = nn.BatchNorm()  # ✅ Removed `use_running_average`
        self.dropout1 = nn.Dropout(rate=drop_rate)

        # Second convolutional block
        self.conv2 = nn.Conv(features=num_filters, kernel_size=(kernel_size,))
        self.bn2 = nn.BatchNorm()  # ✅ Removed `use_running_average`
        self.dropout2 = nn.Dropout(rate=drop_rate)

        # Final output layer for 3D coordinates
        self.output_layer = nn.Conv(features=3, kernel_size=(1,))

    def __call__(self, x, train=True):
        use_running_average = not train  # ✅ Automatically infer

        x = self.embedding(x)

        # First conv block
        x = nn.relu(self.conv1(x))
        x = self.bn1(x, use_running_average=use_running_average)  # ✅ Now works correctly
        x = self.dropout1(x, deterministic=not train)

        # Second conv block
        x = nn.relu(self.conv2(x))
        x = self.bn2(x, use_running_average=use_running_average)  # ✅ Now works correctly
        x = self.dropout2(x, deterministic=not train)

        # Output (x, y, z) per residue
        x = self.output_layer(x)
        return x
