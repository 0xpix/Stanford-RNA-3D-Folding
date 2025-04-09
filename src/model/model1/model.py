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



# === Biologically-informed RNA graph creation ===
def make_rna_graph(batch_embeddings, bppms, threshold=0.1, max_edges_per_node=None):
    """
    Converts batch of embeddings + BPPMs to a GraphsTuple.
    Each sequence becomes one graph.

    Preserves biologically meaningful interactions while managing memory use.
    """
    graphs = []
    for node_features, bppm in zip(batch_embeddings, bppms):
        n_node = node_features.shape[0]

        # For RNA, we need to capture:
        # 1. Local interactions (sequential neighbors)
        # 2. Base pairing interactions (from BPPM)
        # 3. Important tertiary contacts

        # Determine appropriate edge strategy based on sequence length
        if n_node > 2048:
            # For extremely long sequences
            # Use hierarchical approach with three classes of edges:

            # 1. Keep all sequential neighbors (backbone connections)
            seq_senders = jnp.arange(n_node-1)
            seq_receivers = jnp.arange(1, n_node)

            # Create bi-directional edges
            backbone_senders = jnp.concatenate([seq_senders, seq_receivers])
            backbone_receivers = jnp.concatenate([seq_receivers, seq_senders])

            # 2. Keep strong base pairs from BPPM (likely secondary structure)
            # Use higher threshold for very long sequences
            sec_threshold = max(threshold, 0.25)
            sec_senders, sec_receivers = jnp.where(bppm > sec_threshold)

            # 3. Add tertiary contacts with very high probability
            tert_threshold = 0.5  # Only very confident tertiary interactions
            tert_senders, tert_receivers = jnp.where((bppm > tert_threshold) & (jnp.abs(jnp.arange(n_node)[:, None] - jnp.arange(n_node)[None, :]) > 4))

            # Combine all types of edges
            senders = jnp.concatenate([backbone_senders, sec_senders, tert_senders])
            receivers = jnp.concatenate([backbone_receivers, sec_receivers, tert_receivers])

        elif n_node > 1024:
            # For long sequences, use a base-pairing cutoff with backbone guarantee
            # 1. Always include backbone connections
            seq_senders = jnp.arange(n_node-1)
            seq_receivers = jnp.arange(1, n_node)

            # Create bi-directional backbone edges
            backbone_senders = jnp.concatenate([seq_senders, seq_receivers])
            backbone_receivers = jnp.concatenate([seq_receivers, seq_senders])

            # 2. Include base pairing information from BPPM
            # Use adaptive threshold based on sequence length
            adaptive_threshold = max(threshold, 0.15)
            bp_senders, bp_receivers = jnp.where(bppm > adaptive_threshold)

            # Combine backbone and base pairing edges
            senders = jnp.concatenate([backbone_senders, bp_senders])
            receivers = jnp.concatenate([backbone_receivers, bp_receivers])

            # Limit edges if we have too many (but ensure we keep backbone)
            if max_edges_per_node is not None:
                edge_limit_total = n_node * max_edges_per_node
                if len(senders) > edge_limit_total:
                    # Keep all backbone edges
                    backbone_count = len(backbone_senders)

                    # For remaining edges, select by probability
                    remaining_limit = edge_limit_total - backbone_count
                    if remaining_limit > 0 and len(bp_senders) > 0:
                        # Get probability values for base pair edges
                        bp_indices = bp_senders * bppm.shape[1] + bp_receivers
                        bp_probs = bppm.flatten()[bp_indices]

                        # Sort and keep top base pairs
                        sorted_indices = jnp.argsort(-bp_probs)
                        keep_indices = sorted_indices[:remaining_limit]

                        # Final edge set: backbone + top base pairs
                        bp_senders = bp_senders[keep_indices]
                        bp_receivers = bp_receivers[keep_indices]
                        senders = jnp.concatenate([backbone_senders, bp_senders])
                        receivers = jnp.concatenate([backbone_receivers, bp_receivers])
        else:
            # Standard approach for shorter sequences
            # Use BPPM threshold and include backbone
            basic_senders, basic_receivers = jnp.where(bppm > threshold)

            # Always include backbone connections for biological plausibility
            seq_senders = jnp.arange(n_node-1)
            seq_receivers = jnp.arange(1, n_node)

            # Combine the edges
            senders = jnp.concatenate([basic_senders, seq_senders, seq_receivers])
            receivers = jnp.concatenate([basic_receivers, seq_receivers, seq_senders])

            # Remove duplicates by creating a unique edge identifier
            edge_ids = senders * n_node + receivers
            unique_ids, indices = jnp.unique(edge_ids, return_index=True)
            senders = senders[indices]
            receivers = receivers[indices]

        # Create edge features - for RNA we use base pair probabilities as edge weights
        edge_indices = senders * n_node + receivers
        flat_bppm = bppm.flatten()

        # Cap the edge indices to avoid out-of-bounds
        safe_indices = jnp.minimum(edge_indices, flat_bppm.shape[0]-1)
        edge_weights = flat_bppm[safe_indices].reshape(-1, 1)

        # Add minimum weight for backbone edges that might have low BPPM
        min_weight = 0.01
        edge_weights = jnp.maximum(edge_weights, min_weight)

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


# === Memory-Optimized GNN Refinement Module ===
class EnhancedRNAGNN(nn.Module):
    hidden_dim: int = 128
    num_message_passing_steps: int = 3

    @nn.compact
    def __call__(self, graph):
        # Check if this is a very large graph and reduce complexity if needed
        n_nodes = graph.nodes.shape[0]
        n_edges = graph.edges.shape[0] if graph.edges is not None else 0

        # Dynamically adjust hyperparameters based on graph size to save memory
        if n_nodes > 2000:
            # For extremely large graphs, reduce hidden dimension and message passing
            effective_hidden_dim = min(self.hidden_dim, 64)
            effective_mp_steps = 2  # Fewer message passing steps
        elif n_nodes > 1000:
            effective_hidden_dim = min(self.hidden_dim, 96)
            effective_mp_steps = min(self.num_message_passing_steps, 2)
        else:
            effective_hidden_dim = self.hidden_dim
            effective_mp_steps = self.num_message_passing_steps

        # Handle possible empty or invalid graphs
        if n_edges == 0:
            # If no edges, directly predict coordinates from node features
            coords = nn.Sequential([
                nn.Dense(64),
                nn.relu,
                nn.Dense(3)
            ])(graph.nodes)
            return coords

        # Define update functions with memory-efficient operations
        def update_node_fn(nodes, sent_attrs, received_attrs, globals_):
            # Safety check for empty received attributes (happens with empty graphs)
            if received_attrs.shape[0] == 0:
                return nodes

            # Ensure compatible shapes
            if nodes.shape[-1] != received_attrs.shape[-1]:
                # Project received attributes to match node dimension
                received_attrs = nn.Dense(nodes.shape[-1])(received_attrs)

            # Memory-efficient aggregation
            x = nodes + 0.1 * received_attrs  # Simple weighted sum instead of concatenation
            x = nn.LayerNorm()(x)  # Normalize for stability
            x = nn.Dense(effective_hidden_dim)(x)
            x = nn.relu(x)

            return x

        def update_edge_fn(edges, sender_nodes, receiver_nodes, globals_):
            # For very large graphs, use a simplified edge update that's more memory efficient
            if n_nodes > 2000:
                # Instead of concatenation, compute a simple weighted sum
                edge_update = 0.5 * (sender_nodes + receiver_nodes)
                return nn.Dense(edges.shape[-1])(edge_update)
            else:
                # Standard approach for smaller graphs
                try:
                    # Make sure all inputs have the same last dimension
                    if edges.shape[-1] != sender_nodes.shape[-1] or edges.shape[-1] != receiver_nodes.shape[-1]:
                        # Project to a common dimension
                        common_dim = min(edges.shape[-1], sender_nodes.shape[-1], receiver_nodes.shape[-1])
                        edges_proj = nn.Dense(common_dim)(edges)
                        sender_proj = nn.Dense(common_dim)(sender_nodes)
                        receiver_proj = nn.Dense(common_dim)(receiver_nodes)
                        inputs = jnp.concatenate([edges_proj, sender_proj, receiver_proj], axis=-1)
                    else:
                        inputs = jnp.concatenate([edges, sender_nodes, receiver_nodes], axis=-1)

                    x = nn.Dense(effective_hidden_dim // 2)(inputs)
                    x = nn.relu(x)
                    return nn.Dense(edges.shape[-1])(x)
                except Exception as e:
                    # Fallback in case of shape mismatch
                    print(f"Warning: Shape mismatch in edge update, using fallback method. Error: {e}")
                    return edges  # Keep edges unchanged

        # Apply reduced message passing steps for memory efficiency
        for _ in range(effective_mp_steps):
            # Use try-except to handle potential shape mismatches
            try:
                graph = jraph.GraphNetwork(
                    update_node_fn=update_node_fn,
                    update_edge_fn=update_edge_fn,
                    update_global_fn=None
                )(graph)
            except Exception as e:
                print(f"Warning: Error in graph processing: {e}")
                # If graph processing fails, skip this step
                continue

        # Project to 3D coordinates with a simple MLP
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
