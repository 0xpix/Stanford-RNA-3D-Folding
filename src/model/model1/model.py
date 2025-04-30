import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import jraph
from einops import rearrange
from flash_attention_jax import flash_attention
from functools import partial
from flax.training import train_state
import optax


class FlashMHA(nn.Module):
    """
    Clean implementation of Flash Multi-Head Attention using flash_attention_jax.
    Designed to avoid shape issues and be compatible with JAX transformations.
    """

    dim: int
    heads: int = 8
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        batch_size, seq_len, _ = x.shape
        head_dim = self.dim // self.heads

        # Project input to query, key, value
        qkv = nn.Dense(features=3 * self.dim)(x)
        qkv = rearrange(qkv, "b l (t h d) -> t b h l d", t=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [batch, heads, seq_len, head_dim]

        # Default mask that doesn't mask anything if None provided
        if mask is None:
            mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Flash attention - core computation
        attn_output = flash_attention(q, k, v, mask)

        # Merge heads and project
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")

        # Output projection
        attn_output = nn.Dense(features=self.dim)(attn_output)

        # Apply dropout during training
        if self.dropout_rate > 0.0 and not deterministic:
            attn_output = nn.Dropout(rate=self.dropout_rate)(
                attn_output, deterministic=deterministic
            )

        return attn_output


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block with Pre-LayerNorm and Flash Attention.
    """

    dim: int
    heads: int
    mlp_dim: int = None
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Attention with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = FlashMHA(dim=self.dim, heads=self.heads, dropout_rate=self.dropout_rate)(
            x, mask=mask, deterministic=deterministic
        )
        x = x + residual

        # MLP with pre-norm
        residual = x
        x = nn.LayerNorm()(x)

        # Set MLP dimension if not specified
        if self.mlp_dim is None:
            mlp_dim = self.dim * 4
        else:
            mlp_dim = self.mlp_dim

        # MLP layers
        x = nn.Dense(features=mlp_dim)(x)
        x = nn.gelu(x)

        if self.dropout_rate > 0.0 and not deterministic:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        x = nn.Dense(features=self.dim)(x)

        if self.dropout_rate > 0.0 and not deterministic:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not deterministic)

        # Residual connection
        x = x + residual

        return x


class RNATransformer(nn.Module):
    """
    Transformer Encoder stack for RNA processing.
    """

    dim: int
    depth: int
    heads: int
    mlp_dim: int = None
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Process through transformer blocks
        for _ in range(self.depth):
            x = TransformerBlock(
                dim=self.dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )(x, mask=mask, deterministic=deterministic)

        # Final layer norm
        x = nn.LayerNorm()(x)

        return x


class RNAFeatureProcessor(nn.Module):
    """
    Process RNA input features into embeddings.
    """

    dim: int
    vocab_size: int = 5  # RNA token vocabulary size (0-4)

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        # Process tuple inputs (tokens, msa_conservation)
        if isinstance(inputs, (tuple, list)):
            tokens, msa_conservation = inputs

            # Token embeddings
            token_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.dim)(
                tokens
            )

            # Conservation score processing
            msa_embed = nn.Dense(features=self.dim // 4)(msa_conservation[..., None])

            # Combine features
            x = jnp.concatenate([token_embed, msa_embed], axis=-1)

        else:
            # Handle legacy format with combined features
            x = inputs

        # Final projection to desired dimension
        x = nn.Dense(features=self.dim)(x)

        return x


class SimpleGNN(nn.Module):
    """
    Graph Neural Network for RNA structure prediction.
    """

    hidden_dim: int = 128
    output_dim: int = 3  # 3D coordinates
    num_layers: int = 2

    @nn.compact
    def __call__(self, graph):
        # Extract graph components
        nodes = graph.nodes
        edges = (
            graph.edges
            if graph.edges is not None
            else jnp.ones((graph.senders.shape[0], 1))
        )
        senders = graph.senders
        receivers = graph.receivers

        # Initial node embeddings
        h_nodes = nn.Dense(features=self.hidden_dim)(nodes)
        h_nodes = nn.relu(h_nodes)
        h_nodes = nn.LayerNorm()(h_nodes)

        # Edge processing
        h_edges = nn.Dense(features=self.hidden_dim // 2)(edges)
        h_edges = nn.relu(h_edges)

        # Message passing layers
        for _ in range(self.num_layers):
            # Gather source and destination node features
            h_src = h_nodes[senders]
            h_dst = h_nodes[receivers]

            # Create messages
            messages = jnp.concatenate([h_src, h_dst, h_edges], axis=-1)
            messages = nn.Dense(features=self.hidden_dim)(messages)
            messages = nn.relu(messages)

            # Aggregate messages at nodes
            aggregated = jraph.segment_sum(
                messages, receivers, num_segments=nodes.shape[0]
            )

            # Update node features
            h_nodes_update = nn.Dense(features=self.hidden_dim)(aggregated)
            h_nodes = h_nodes + h_nodes_update
            h_nodes = nn.LayerNorm()(h_nodes)
            h_nodes = nn.relu(h_nodes)

        # Output projection to 3D coordinates
        coords = nn.Dense(features=64)(h_nodes)
        coords = nn.relu(coords)
        coords = nn.Dense(features=self.output_dim)(coords)

        return coords


def make_rna_graph(node_features, bppms, threshold=0.1, max_edges=1000):
    """
    Create RNA structure graphs from node features and base pair probability matrices.
    Returns batched GraphsTuple for processing with GNN.
    """
    batch_size = node_features.shape[0]
    graphs = []

    for i in range(batch_size):
        features = node_features[i]
        bppm = bppms[i]
        seq_len = features.shape[0]

        # Create backbone connections (sequential neighbors)
        indices = jnp.arange(seq_len)
        backbone_src = indices[:-1]  # 0 to n-2
        backbone_dst = indices[1:]  # 1 to n-1

        # Add reverse edges too (bidirectional)
        backbone_srcs = jnp.concatenate([backbone_src, backbone_dst])
        backbone_dsts = jnp.concatenate([backbone_dst, backbone_src])

        # Edge features for backbone (all ones)
        backbone_features = jnp.ones((len(backbone_srcs), 1))

        # Add base pair edges from BPPM (avoid diagonal)
        bp_srcs = []
        bp_dsts = []
        bp_features = []

        # Use a simpler approach that's JIT-compatible
        # Get the top-k values from the flattened BPPM (excluding diagonal)
        flat_bppm = bppm.flatten()
        k = min(max_edges, seq_len * seq_len // 10)
        vals, indices = lax.top_k(flat_bppm, k)

        # Convert flat indices to 2D coordinates
        srcs = indices // seq_len
        dsts = indices % seq_len

        # Filter by threshold
        valid = vals > threshold
        valid_srcs = jnp.where(valid, srcs, 0)
        valid_dsts = jnp.where(valid, dsts, 0)
        valid_vals = jnp.where(valid, vals, 0.0)

        # Combine backbone and base pair edges
        all_srcs = jnp.concatenate([backbone_srcs, valid_srcs])
        all_dsts = jnp.concatenate([backbone_dsts, valid_dsts])
        all_features = jnp.concatenate([backbone_features, valid_vals[:, None]])

        # Create graph
        graph = jraph.GraphsTuple(
            nodes=features,
            edges=all_features,
            senders=all_srcs,
            receivers=all_dsts,
            n_node=jnp.array([seq_len]),
            n_edge=jnp.array([len(all_srcs)]),
            globals=None,
        )

        graphs.append(graph)

    # Batch graphs together
    if len(graphs) > 1:
        return jraph.batch(graphs)
    else:
        return graphs[0]


class RNAFoldingModel(nn.Module):
    """
    Complete RNA 3D structure prediction model.
    Processes RNA sequences and predicts 3D coordinates.
    """

    dim: int
    depth: int
    heads: int
    dropout_rate: float = 0.1
    use_checkpoint: bool = True  # Flag only, not actually used in this implementation

    def setup(self):
        self.feature_processor = RNAFeatureProcessor(self.dim)
        self.transformer = RNATransformer(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout_rate=self.dropout_rate,
        )
        self.gnn = SimpleGNN(hidden_dim=self.dim)

    def __call__(self, inputs, training=False):
        # Process inputs
        if isinstance(inputs, (tuple, list)):
            tokens, msa_conservation, bppm = inputs

            # Process features
            x = self.feature_processor(
                (tokens, msa_conservation), deterministic=not training
            )

            # Create mask for non-padding tokens
            mask = tokens > 0

        else:
            # Legacy format with combined features
            x = self.feature_processor(inputs, deterministic=not training)
            bppm = None
            mask = None

        # Apply transformer encoder
        encoded = self.transformer(x, mask=mask, deterministic=not training)

        # Use GNN if we have base pair information
        if bppm is not None:
            # Create RNA structure graph
            graph = make_rna_graph(encoded, bppm)
            coords = self.gnn(graph)
        else:
            # Fallback to direct prediction if no graph info
            coords = nn.Dense(3)(encoded)

        return coords

    def create_train_state(self, rng, input_shape, learning_rate=1e-3):
        """Create initial training state with parameters and optimizer."""
        # Split RNG keys for parameters and dropout
        params_rng, dropout_rng = jax.random.split(rng)

        # Create mock inputs for initialization
        if isinstance(input_shape[0], (tuple, list)):
            # Newer format with separate inputs
            tokens_shape, msa_shape, bppm_shape = input_shape

            mock_tokens = jnp.ones(tokens_shape, dtype=jnp.int32)
            mock_msa = jnp.ones(msa_shape)
            mock_bppm = jnp.ones(bppm_shape)
            mock_input = (mock_tokens, mock_msa, mock_bppm)

        else:
            # Legacy format with combined features
            mock_input = jnp.ones(input_shape)

        # Initialize model parameters
        variables = self.init(
            {"params": params_rng, "dropout": dropout_rng}, mock_input, training=True
        )

        # Create optimizer
        tx = optax.adam(learning_rate)

        # Create train state with dropout RNG
        class TrainStateWithRNG(train_state.TrainState):
            dropout_rng: jnp.ndarray

        return TrainStateWithRNG.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=tx,
            dropout_rng=dropout_rng,
        )
