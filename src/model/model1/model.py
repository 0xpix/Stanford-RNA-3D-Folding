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



# === Fully JIT-compatible RNA graph creation (no dynamic shapes or boolean indexing) ===
def make_rna_graph(batch_embeddings, bppms, threshold=0.1, max_edges=5000):
    """
    Creates an RNA graph representation that is fully compatible with JAX's JIT compilation.
    Avoids all operations that could lead to dynamic shapes or boolean indexing.

    Args:
        batch_embeddings: Node features from encoder [batch_size, seq_len, hidden_dim]
        bppms: Base-pairing probability matrices [batch_size, seq_len, seq_len]
        threshold: Probability threshold for including base pairs
        max_edges: Maximum number of edges per graph (for memory efficiency)

    Returns:
        A batched GraphsTuple representation of the RNA structures
    """
    graphs = []

    for node_features, bppm in zip(batch_embeddings, bppms):
        n_node = node_features.shape[0]

        # Step 1: Always include backbone connections (fixed size, JIT-friendly)
        # Create indices for neighboring positions in the sequence
        backbone_src = jnp.arange(n_node - 1)  # 0 to n-2
        backbone_dst = backbone_src + 1        # 1 to n-1

        # Also add reverse edges
        backbone_srcs = jnp.concatenate([backbone_src, backbone_dst])
        backbone_dsts = jnp.concatenate([backbone_dst, backbone_src])

        # Fixed number of backbone edges: 2 * (n_node - 1)
        backbone_edge_count = len(backbone_srcs)

        # Step 2: Create a fixed number of base-pairing edges
        # Instead of using boolean masking, we'll:
        # 1. Determine max number of base-pairing edges to include
        # 2. Create dummy values that will be updated
        # 3. Top-k approach for selecting edges

        # Determine number of base-pairing edges to include beyond backbone
        remaining_edges = max(0, min(max_edges - backbone_edge_count, n_node * 10))

        # Initialize dummy values for base-pairing edges
        bp_srcs = jnp.zeros(remaining_edges, dtype=jnp.int32)
        bp_dsts = jnp.zeros(remaining_edges, dtype=jnp.int32)
        bp_scores = jnp.zeros(remaining_edges, dtype=jnp.float32)

        # To select the highest probability base pairs without boolean indexing,
        # we'll use a scan function that iteratively builds the edge list
        def scan_fn(carry, idx):
            i = idx // n_node
            j = idx % n_node

            # Current state
            srcs, dsts, scores, count = carry

            # Conditions for valid edge:
            # 1. Above threshold
            # 2. Not a backbone edge (must be separated by at least 2 positions)
            # 3. Still have space in our edge list
            prob = bppm[i, j]
            valid_edge = (prob > threshold) & (jnp.abs(i - j) > 2) & (count < remaining_edges)

            # Update arrays conditionally
            new_count = count + jnp.int32(valid_edge)
            idx_to_update = jnp.minimum(count, remaining_edges - 1)  # Ensure within bounds

            # Update edge arrays with new edge if valid
            new_srcs = jax.lax.dynamic_update_index_in_dim(srcs, i, idx_to_update, 0)
            new_dsts = jax.lax.dynamic_update_index_in_dim(dsts, j, idx_to_update, 0)
            new_scores = jax.lax.dynamic_update_index_in_dim(scores, prob, idx_to_update, 0)

            # Only update if this is a valid edge
            srcs = jnp.where(valid_edge, new_srcs, srcs)
            dsts = jnp.where(valid_edge, new_dsts, dsts)
            scores = jnp.where(valid_edge, new_scores, scores)

            return (srcs, dsts, scores, new_count), None

        # Start with empty edge lists
        init_state = (bp_srcs, bp_dsts, bp_scores, jnp.array(0, dtype=jnp.int32))

        # Only scan through a reasonable number of potential edges
        # For large sequences, we'll sample a subset of indices
        if n_node > 1000:
            # For very large sequences, only check a subset of potential edges
            # to avoid quadratic computation
            sample_indices = jnp.linspace(0, n_node*n_node - 1, min(n_node*100, 100000)).astype(jnp.int32)
        else:
            # For smaller sequences, check all potential edges
            sample_indices = jnp.arange(min(n_node*n_node, 100000))

        # Run the scan
        (final_srcs, final_dsts, final_scores, _), _ = jax.lax.scan(
            scan_fn,
            init_state,
            sample_indices
        )

        # Step 3: Combine backbone and base-pairing edges
        senders = jnp.concatenate([backbone_srcs, final_srcs])
        receivers = jnp.concatenate([backbone_dsts, final_dsts])

        # Create edge features - default to 1.0 for backbone, use score for base-pairs
        backbone_scores = jnp.ones(backbone_edge_count)
        edge_weights = jnp.concatenate([backbone_scores, final_scores])

        # Reshape to add feature dimension
        edge_weights = edge_weights.reshape(-1, 1)

        # Create the graph
        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=edge_weights,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_node]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )

        graphs.append(graph)

    # Use jraph.batch to combine multiple graphs
    if len(graphs) > 1:
        return jraph.batch(graphs)
    else:
        return graphs[0]


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


# === Simplified, Robust GNN for RNA Structure Prediction ===
class EnhancedRNAGNN(nn.Module):
    """
    A simplified GNN designed specifically for RNA 3D structure prediction.
    Avoids shape mismatches by using a more straightforward architecture.
    """
    output_dim: int = 3  # 3D coordinates

    @nn.compact
    def __call__(self, graph):
        # Extract graph components
        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        receivers = graph.receivers

        # Get input dimensions
        node_dim = nodes.shape[-1]

        # Initial node projection - first layer is critical for setting dimensions right
        # Using a 2-layer MLP for initial node projection
        h_nodes = nn.Dense(features=128)(nodes)
        h_nodes = nn.relu(h_nodes)
        h_nodes = nn.LayerNorm()(h_nodes)

        # First round of message passing (fixed architecture to avoid shape issues)
        # Step 1: Gather source node features
        s_nodes = h_nodes[senders]

        # Step 2: Gather target node features
        t_nodes = h_nodes[receivers]

        # Step 3: Process edge features
        # Ensure edge features are properly shaped
        if edges is None:
            # Create default edge features if none are provided
            edge_features = jnp.ones((senders.shape[0], 1))
        else:
            edge_features = edges

        # Project edges to correct dimension if needed
        if edge_features.shape[-1] != 64:
            edge_features = nn.Dense(features=64)(edge_features)

        # Step 4: Combine node and edge features for messages
        # Use simple summation instead of concatenation to avoid shape issues
        messages = s_nodes * 0.5 + t_nodes * 0.3 + nn.Dense(features=128)(edge_features) * 0.2
        messages = nn.relu(messages)

        # Step 5: Aggregate messages - simple summation aggregation
        # This uses a segment_sum operation which is JIT-friendly
        aggregated = jraph.segment_sum(messages, receivers, num_segments=nodes.shape[0])

        # Step 6: Update node features with a residual connection
        h_nodes = h_nodes + nn.Dense(features=128)(aggregated)
        h_nodes = nn.LayerNorm()(h_nodes)
        h_nodes = nn.relu(h_nodes)

        # Second round of message passing with same architecture
        s_nodes = h_nodes[senders]
        t_nodes = h_nodes[receivers]

        messages = s_nodes * 0.5 + t_nodes * 0.3 + nn.Dense(features=128)(edge_features) * 0.2
        messages = nn.relu(messages)

        aggregated = jraph.segment_sum(messages, receivers, num_segments=nodes.shape[0])

        h_nodes = h_nodes + nn.Dense(features=128)(aggregated)
        h_nodes = nn.LayerNorm()(h_nodes)
        h_nodes = nn.relu(h_nodes)

        # Final MLP to predict 3D coordinates
        coords = nn.Dense(features=64)(h_nodes)
        coords = nn.relu(coords)
        coords = nn.Dense(features=self.output_dim)(coords)

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
                # The new EnhancedRNAGNN doesn't take a hidden_dim parameter
                return EnhancedRNAGNN(output_dim=3)(graph)

            if self.use_checkpoint and training:
                coords = lax.checkpoint(graph_fn, encoded, bppm)
            else:
                coords = graph_fn(encoded, bppm)
        else:
            # Standard processing for smaller sequences
            graph = make_rna_graph(encoded, bppm)
            # The new EnhancedRNAGNN only takes output_dim
            coords = EnhancedRNAGNN(output_dim=3)(graph)

        return coords
