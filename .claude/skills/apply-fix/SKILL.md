---
name: apply-fix
description: Implement the top recommendation from /analyze-run. Applies one focused change at a time — config tweak or code-level improvement — then prepares the training run.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---

# Apply Fix

Implement exactly ONE improvement from the latest /analyze-run diagnosis. This skill applies changes directly to the codebase, following the one-variable-at-a-time methodology.

## Principles

- **One change per invocation** — never change multiple things at once
- **Edit in-place** — modify `configs/experiment/lgcn_gowalla_full.yaml` and source files directly
- **Backward compatible** — use `getattr(cfg, "new_param", default)` for new config keys
- **Always explain** — update config comment and provide before/after summary

## Step 1: Read Current State

Read these files to understand what we're working with:

```
configs/experiment/lgcn_gowalla_full.yaml
src/model.py
src/train.py
src/data.py
```

## Step 2: Identify Which Fix to Apply

The user will tell you which recommendation to implement, or you should apply the top priority one from the latest /analyze-run. The fixes below are ordered by typical priority.

## Available Fixes

### Fix A: Config Tweak (reg_weight)

When: Analysis says "low loss but metrics not proportionally better" or "may be overfitting BPR objective".

**Edit** `configs/experiment/lgcn_gowalla_full.yaml`:
- Change `reg_weight` to the recommended value
- Update the comment at the top

No code changes needed.

---

### Fix B: Multiple Negative Samples

When: Analysis says "need more gradient signal per epoch" or "converged but below benchmarks".

This multiplies effective gradient steps without adding epochs. Currently 1 negative per positive — change to K negatives.

**Edit** `src/data.py` — modify `sample_negatives()`:

```python
def sample_negatives(
    train_dict: dict[int, list[int]],
    n_items: int,
    rng: np.random.Generator,
    n_negatives: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate n_negatives negative samples per positive interaction."""
    users, pos_items, neg_items = [], [], []
    for user, items in train_dict.items():
        item_set = set(items)
        for item in items:
            for _ in range(n_negatives):
                users.append(user)
                pos_items.append(item)
                neg = rng.integers(0, n_items)
                while neg in item_set:
                    neg = rng.integers(0, n_items)
                neg_items.append(neg)
    return (
        np.array(users, dtype=np.int32),
        np.array(pos_items, dtype=np.int32),
        np.array(neg_items, dtype=np.int32),
    )
```

**Edit** `src/train.py` — pass n_negatives from config:

```python
n_negatives = getattr(cfg.train, "n_negatives", 1)

# In the training loop, change the sample_negatives call:
users, pos_items, neg_items = sample_negatives(
    dataset.train_dict, dataset.n_items, rng, n_negatives
)
```

**Edit** `configs/experiment/lgcn_gowalla_full.yaml` — add:
```yaml
train:
  n_negatives: 3  # 3 negatives per positive
```

---

### Fix C: Layer Attention Weights

When: Analysis says "capacity limit" or "4 layers may over-smooth" or "need architectural change".

Replace uniform mean over layers with learnable per-layer weights. Layers closer to ego may matter more than deep layers.

**Edit** `src/model.py` — modify `init_params()`:

```python
def init_params(n_users: int, n_items: int, embed_dim: int, n_layers: int, key: jax.Array) -> dict:
    """Initialize user/item embeddings and layer attention weights."""
    n_nodes = n_users + n_items
    embedding = jax.random.normal(key, (n_nodes, embed_dim)) * (1.0 / jnp.sqrt(embed_dim))
    # Learnable layer weights (n_layers + 1 for ego layer), initialized uniform
    layer_weights = jnp.ones(n_layers + 1) / (n_layers + 1)
    return {"embedding": embedding, "layer_weights": layer_weights}
```

**Edit** `src/model.py` — modify `lightgcn_forward()`:

```python
def lightgcn_forward(
    params: dict,
    adj_norm: jsparse.BCOO,
    n_layers: int,
    embed_dropout: float = 0.0,
    key: jax.Array | None = None,
    training: bool = False,
) -> jnp.ndarray:
    """LightGCN propagation: learned weighted sum of layer-0..K embeddings."""
    ego_embed = params["embedding"]

    if training and embed_dropout > 0.0 and key is not None:
        mask = jax.random.bernoulli(key, 1.0 - embed_dropout, ego_embed.shape)
        ego_embed = ego_embed * mask / (1.0 - embed_dropout)

    all_embeds = [ego_embed]
    x = ego_embed
    for _ in range(n_layers):
        x = adj_norm @ x
        all_embeds.append(x)

    # Softmax over layer weights for learned attention
    weights = jax.nn.softmax(params.get("layer_weights", jnp.ones(n_layers + 1)))
    stacked = jnp.stack(all_embeds, axis=0)  # (n_layers+1, n_nodes, embed_dim)
    return jnp.sum(stacked * weights[:, None, None], axis=0)
```

**Edit** `src/train.py` — pass n_layers to init_params:

```python
params = init_params(dataset.n_users, dataset.n_items, cfg.model.embed_dim, cfg.model.n_layers, key)
```

**Edit** `src/main.py` — same change in init_params call if present.

Note: `bpr_loss()` doesn't need changes — it calls `lightgcn_forward()` which reads `params["layer_weights"]` internally. The L2 reg only applies to embeddings, not layer weights (they're tiny).

---

### Fix D: Edge Dropout

When: Analysis says "overfitting" or "need regularization without embedding dropout".

Randomly drop edges from the adjacency matrix during training. This is different from embedding dropout — it forces the model to learn from different subgraphs.

**Edit** `src/model.py` — add edge dropout to forward:

```python
def lightgcn_forward(
    params: dict,
    adj_norm: jsparse.BCOO,
    n_layers: int,
    embed_dropout: float = 0.0,
    edge_dropout: float = 0.0,
    key: jax.Array | None = None,
    training: bool = False,
) -> jnp.ndarray:
    ego_embed = params["embedding"]

    if training and embed_dropout > 0.0 and key is not None:
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, 1.0 - embed_dropout, ego_embed.shape)
        ego_embed = ego_embed * mask / (1.0 - embed_dropout)

    all_embeds = [ego_embed]
    x = ego_embed
    for _ in range(n_layers):
        if training and edge_dropout > 0.0 and key is not None:
            key, subkey = jax.random.split(key)
            edge_mask = jax.random.bernoulli(subkey, 1.0 - edge_dropout, (adj_norm.data.shape[0],))
            dropped_adj = jsparse.BCOO((adj_norm.data * edge_mask / (1.0 - edge_dropout), adj_norm.indices), shape=adj_norm.shape)
            x = dropped_adj @ x
        else:
            x = adj_norm @ x
        all_embeds.append(x)
    return jnp.mean(jnp.stack(all_embeds, axis=0), axis=0)
```

Thread `edge_dropout` through `bpr_loss()` and `train_step()` from config:
```python
edge_dropout = getattr(cfg.model, "edge_dropout", 0.0)
```

**Edit** config:
```yaml
model:
  edge_dropout: 0.1
```

---

### Fix E: Popularity-Biased Negative Sampling

When: Analysis says "diversity too low" or "popularity bias" or "no cold items recommended".

Sample negatives proportional to item popularity instead of uniform. Popular items are harder negatives.

**Edit** `src/data.py` — add popularity-weighted sampling:

```python
def sample_negatives(
    train_dict: dict[int, list[int]],
    n_items: int,
    rng: np.random.Generator,
    n_negatives: int = 1,
    popularity_bias: float = 0.0,
    item_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate negative samples. popularity_bias=0 is uniform, 0.75 is standard bias."""
    # Build sampling weights if biased
    if popularity_bias > 0.0 and item_counts is not None:
        weights = np.power(item_counts.astype(np.float64) + 1, popularity_bias)
        weights /= weights.sum()
    else:
        weights = None

    users, pos_items, neg_items = [], [], []
    for user, items in train_dict.items():
        item_set = set(items)
        for item in items:
            for _ in range(n_negatives):
                users.append(user)
                pos_items.append(item)
                if weights is not None:
                    neg = rng.choice(n_items, p=weights)
                else:
                    neg = rng.integers(0, n_items)
                while neg in item_set:
                    if weights is not None:
                        neg = rng.choice(n_items, p=weights)
                    else:
                        neg = rng.integers(0, n_items)
                neg_items.append(neg)
    return (
        np.array(users, dtype=np.int32),
        np.array(pos_items, dtype=np.int32),
        np.array(neg_items, dtype=np.int32),
    )
```

Requires precomputing item_counts in train.py and passing to sample_negatives.

---

## Step 3: Apply the Fix

1. Read current files
2. Make the edits described above for the chosen fix
3. Update the config comment to explain what changed
4. Verify no syntax errors by running: `uv run python -c "from model import *; from data import *; from train import *; print('OK')"`

## Step 4: Output

Provide:
1. **Fix applied**: which fix (A/B/C/D/E) and why
2. **Files changed**: list with brief description
3. **Run command**: `uv run python src/main.py experiment=lgcn_gowalla_full`
4. **What to look for**: which metric should improve and by how much
5. **Rollback**: how to undo if it doesn't work (git checkout the files)
