"""LightGCN model: embedding init, forward pass, BPR loss."""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp


def init_params(n_users: int, n_items: int, embed_dim: int, key: jax.Array) -> dict:
    """Initialize user and item embeddings (single matrix)."""
    n_nodes = n_users + n_items
    embedding = jax.random.normal(key, (n_nodes, embed_dim)) * 0.1
    return {"embedding": embedding}


def lightgcn_forward(
    params: dict,
    adj_norm: jsparse.BCOO,
    n_layers: int,
    embed_dropout: float = 0.0,
    key: jax.Array | None = None,
    training: bool = False,
) -> jnp.ndarray:
    """LightGCN propagation: mean of layer-0..K embeddings."""
    ego_embed = params["embedding"]

    # Embedding dropout during training
    if training and embed_dropout > 0.0 and key is not None:
        mask = jax.random.bernoulli(key, 1.0 - embed_dropout, ego_embed.shape)
        ego_embed = ego_embed * mask / (1.0 - embed_dropout)

    all_embeds = [ego_embed]
    x = ego_embed
    for _ in range(n_layers):
        x = adj_norm @ x
        all_embeds.append(x)
    return jnp.mean(jnp.stack(all_embeds, axis=0), axis=0)


def bpr_loss(
    params: dict,
    adj_norm: jsparse.BCOO,
    n_layers: int,
    n_users: int,
    users: jnp.ndarray,
    pos_items: jnp.ndarray,
    neg_items: jnp.ndarray,
    reg_weight: float,
    embed_dropout: float = 0.0,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """BPR loss with L2 regularization on ego embeddings."""
    all_embed = lightgcn_forward(
        params, adj_norm, n_layers,
        embed_dropout=embed_dropout, key=key, training=key is not None,
    )

    user_embed = all_embed[users]
    pos_embed = all_embed[n_users + pos_items]
    neg_embed = all_embed[n_users + neg_items]

    pos_scores = jnp.sum(user_embed * pos_embed, axis=1)
    neg_scores = jnp.sum(user_embed * neg_embed, axis=1)
    bpr = -jnp.mean(jax.nn.log_sigmoid(pos_scores - neg_scores))

    # L2 reg on ego (layer-0) embeddings of batch entities only
    ego = params["embedding"]
    user_ego = ego[users]
    pos_ego = ego[n_users + pos_items]
    neg_ego = ego[n_users + neg_items]
    reg = reg_weight * (
        jnp.mean(jnp.sum(user_ego**2, axis=1))
        + jnp.mean(jnp.sum(pos_ego**2, axis=1))
        + jnp.mean(jnp.sum(neg_ego**2, axis=1))
    )

    return bpr + reg
