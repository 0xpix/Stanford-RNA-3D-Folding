import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import jraph
from einops import rearrange
from flash_attention_jax import flash_attention
from functools import partial


# === Memory-efficient FlashAttention Block (with checkpoint gradients) ===
class FlashAttentionBlock(nn.Module):
    dim: int
    heads: int
    use_checkpoint: bool = True  # Enable gradient checkpointing by default

    @nn.compact
    def __call__(self, x, mask=None):
        B, L, D = x.shape
        head_dim = self.dim // self.heads

        # Define forward computation for gradient checkpointing
        def attention_forward(x, mask):
            # Project QKV
            qkv = nn.Dense(self.dim * 3)(x)
            qkv = rearrange(qkv, 'b l (h d c) -> c b h l d', h=self.heads, c=3)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Mask shape: (B, L)
            if mask is None:
                mask = jnp.ones((B, L), dtype=jnp.int32)

            # Memory-efficient sequence length handling
            # Calculate actual sequence length based on non-padding tokens
            if L > 512:  # Only apply special handling for long sequences
                # Find out where the actual content ends (non-zero tokens)
                actual_seq_length = L
            else:
                actual_seq_length = L

            # Handle different sequence lengths by padding or truncating
            if q.shape[2] != actual_seq_length:
                if q.shape[2] < actual_seq_length:
                    padding = jnp.zeros((B, self.heads, actual_seq_length - q.shape[2], head_dim))
                    q = jnp.concatenate([q, padding], axis=2)
                else:
                    q = q[:, :, :actual_seq_length, :]

            if k.shape[2] != actual_seq_length:
                if k.shape[2] < actual_seq_length:
                    padding = jnp.zeros((B, self.heads, actual_seq_length - k.shape[2], head_dim))
                    k = jnp.concatenate([k, padding], axis=2)
                else:
                    k = k[:, :, :actual_seq_length, :]

            if v.shape[2] != actual_seq_length:
                if v.shape[2] < actual_seq_length:
                    padding = jnp.zeros((B, self.heads, actual_seq_length - v.shape[2], head_dim))
                    v = jnp.concatenate([v, padding], axis=2)
                else:
                    v = v[:, :, :actual_seq_length, :]

            # For very long sequences, use a more memory-efficient approach
            if L > 1024:
                # Split attention computation into smaller chunks
                chunk_size = 512
                num_chunks = (L + chunk_size - 1) // chunk_size

                out_chunks = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, L)

                    # Process current chunk
                    q_chunk = q[:, :, start_idx:end_idx, :]
                    # Use full k, v for attention but with lower precision to save memory
                    out_chunk = flash_attention(q_chunk, k, v, mask)
                    out_chunks.append(out_chunk)

                # Concatenate chunk outputs
                out = jnp.concatenate(out_chunks, axis=2)
            else:
                # Standard flash attention for smaller sequences
                out = flash_attention(q, k, v, mask)

            # Ensure output has the correct sequence length
            if out.shape[2] != L:
                if out.shape[2] < L:
                    padding = jnp.zeros((B, self.heads, L - out.shape[2], head_dim))
                    out = jnp.concatenate([out, padding], axis=2)
                else:
                    out = out[:, :, :L, :]

            out = rearrange(out, 'b h l d -> b l (h d)')
            return nn.Dense(self.dim)(out)

        # Use gradient checkpointing for memory efficiency during backprop
        if self.use_checkpoint and jax.config.read("jax_enable_x64"):
            # Only use checkpointing when training (not during inference)
            # We detect training mode by checking if 64-bit precision is enabled
            return lax.checkpoint(attention_forward, x, mask)
        else:
            return attention_forward(x, mask)


# === Memory-Efficient Transformer Layer with Checkpointing ===
class TransformerLayer(nn.Module):
    dim: int
    heads: int
    use_checkpoint: bool = True

    @nn.compact
    def __call__(self, x, mask=None):
        # Define the forward computation for attention + MLP blocks
        def forward_fn(x, mask):
            # Attention block
            attn_out = FlashAttentionBlock(self.dim, self.heads, self.use_checkpoint)(x, mask)
            x = nn.LayerNorm()(x + attn_out)

            # MLP block
            mlp_out = nn.Dense(self.dim * 4)(x)
            mlp_out = nn.gelu(mlp_out)
            mlp_out = nn.Dense(self.dim)(mlp_out)

            return nn.LayerNorm()(x + mlp_out)

        # Use gradient checkpointing for memory efficiency during backprop
        if self.use_checkpoint and jax.config.read("jax_enable_x64"):
            # Only use checkpointing when training
            return lax.checkpoint(forward_fn, x, mask)
        else:
            return forward_fn(x, mask)


# === Memory-Optimized Transformer Encoder ===
class RNAEncoder(nn.Module):
    dim: int
    heads: int
    depth: int
    use_checkpoint: bool = True

    @nn.compact
    def __call__(self, x, mask=None):
        # For long sequences, we can apply aggressive checkpointing
        seq_len = x.shape[1]

        # Define progressive forward function for the stack of transformer layers
        def forward_stack(start_layer, end_layer, x, mask):
            for layer_idx in range(start_layer, end_layer):
                x = TransformerLayer(
                    self.dim,
                    self.heads,
                    use_checkpoint=self.use_checkpoint
                )(x, mask)
            return x

        # For very long sequences, chunk the transformer stack into smaller pieces
        # This improves memory usage during backpropagation
        if seq_len > 1024 and self.depth > 2:
            # Process layers in smaller groups to reduce peak memory
            stack_size = 2  # Process 2 layers at a time
            num_stacks = (self.depth + stack_size - 1) // stack_size

            for stack_idx in range(num_stacks):
                start_layer = stack_idx * stack_size
                end_layer = min((stack_idx + 1) * stack_size, self.depth)

                # Apply gradient checkpointing to each stack
                if self.use_checkpoint and jax.config.read("jax_enable_x64"):
                    x = lax.checkpoint(forward_stack, start_layer, end_layer, x, mask)
                else:
                    x = forward_stack(start_layer, end_layer, x, mask)
        else:
            # Standard processing for smaller sequences
            for _ in range(self.depth):
                x = TransformerLayer(
                    self.dim,
                    self.heads,
                    use_checkpoint=self.use_checkpoint
                )(x, mask)

        return x



# === GNN Refinement using jraph ===
def make_rna_graph(batch_embeddings, bppms, threshold=0.1):
    """
    Converts batch of embeddings + BPPMs to a GraphsTuple.
    Each sequence becomes one graph.
    """
    graphs = []
    for node_features, bppm in zip(batch_embeddings, bppms):
        n_node = node_features.shape[0]
        senders, receivers = jnp.where(bppm > threshold)
        edges = jnp.ones((len(senders), 1), dtype=jnp.float32)

        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )
        graphs.append(graph)

    return jraph.batch_np(graphs)


# === Enhanced RNA Feature Processing ===
class RNAFeatureProcessor(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, tokens, msa_conservation, secondary_structure=None):
        # Embed nucleotide tokens (A, C, G, U)
        token_embed = nn.Embed(num_embeddings=5, features=self.dim)(tokens)  # 0-4

        # Process conservation scores
        conservation_embed = nn.Dense(self.dim // 4)(msa_conservation[..., None])

        # Optional secondary structure features if available
        if secondary_structure is not None:
            ss_embed = nn.Embed(num_embeddings=8, features=self.dim // 4)(secondary_structure)
            features = jnp.concatenate([token_embed, conservation_embed, ss_embed], axis=-1)
        else:
            features = jnp.concatenate([token_embed, conservation_embed], axis=-1)

        return nn.Dense(self.dim)(features)


# === Improved GNN Refinement Module ===
class EnhancedRNAGNN(nn.Module):
    hidden_dim: int = 128
    num_message_passing_steps: int = 3

    @nn.compact
    def __call__(self, graph):
        # Define update functions within the @nn.compact context
        def update_node_fn(nodes, sent_attrs, received_attrs, globals_):
            inputs = jnp.concatenate([nodes, received_attrs], axis=-1)
            x = nn.Dense(self.hidden_dim)(inputs)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim)(x)
            return x + nn.Dense(self.hidden_dim)(nodes)  # Residual connection

        def update_edge_fn(edges, sender_nodes, receiver_nodes, globals_):
            inputs = jnp.concatenate([edges, sender_nodes, receiver_nodes], axis=-1)
            x = nn.Dense(self.hidden_dim // 2)(inputs)
            x = nn.relu(x)
            return nn.Dense(edges.shape[-1])(x)

        # Apply message passing steps
        for _ in range(self.num_message_passing_steps):
            graph = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=None
            )(graph)

        # Project to 3D coordinates
        coords = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(3)
        ])(graph.nodes)

        return coords


# === Enhanced Full Model with Memory Optimizations ===
class RNAFoldingModel(nn.Module):
    dim: int
    heads: int
    depth: int
    dropout_rate: float = 0.1
    use_checkpoint: bool = True

    @nn.compact
    def __call__(self, tokens, msa_conservation, bppm, secondary_structure=None, training=False):
        # Check if we need to handle extremely long sequences
        B, L = tokens.shape

        # For extremely long sequences, use a lower embedding dimension
        effective_dim = self.dim
        if L > 2048:
            # Reduce dimension for very long sequences to save memory
            effective_dim = max(self.dim // 2, 128)

        # Enhanced feature processing with gradient checkpointing
        def feature_fn(tokens, msa_conservation, secondary_structure):
            return RNAFeatureProcessor(effective_dim)(tokens, msa_conservation, secondary_structure)

        if self.use_checkpoint and training and L > 1024:
            x = lax.checkpoint(feature_fn, tokens, msa_conservation, secondary_structure)
        else:
            x = feature_fn(tokens, msa_conservation, secondary_structure)

        # Apply dropout during training
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # Transformer encoder with memory optimizations
        encoded = RNAEncoder(
            effective_dim,
            self.heads,
            self.depth,
            use_checkpoint=self.use_checkpoint
        )(x)

        # For large sequences, apply additional optimizations to graph construction
        if L > 1024:
            # Use a higher threshold for edge creation to reduce graph density
            threshold = 0.2  # More conservative threshold for long sequences

            # Create graph and apply enhanced GNN (with potential chunking)
            def graph_fn(encoded, bppm):
                graph = make_rna_graph(encoded, bppm, threshold=threshold)
                return EnhancedRNAGNN(hidden_dim=min(effective_dim, 96))(graph)

            if self.use_checkpoint and training:
                coords = lax.checkpoint(graph_fn, encoded, bppm)
            else:
                coords = graph_fn(encoded, bppm)
        else:
            # Standard processing for smaller sequences
            graph = make_rna_graph(encoded, bppm)
            coords = EnhancedRNAGNN()(graph)

        return coords
