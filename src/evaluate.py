"""Evaluation: Recall@K and NDCG@K via batched full-ranking."""

import jax.numpy as jnp
import numpy as np


def evaluate(
    all_embed: jnp.ndarray,
    n_users: int,
    n_items: int,
    train_dict: dict[int, list[int]],
    test_dict: dict[int, list[int]],
    topk: int = 20,
    user_batch_size: int = 1024,
) -> dict[str, float]:
    """Compute Recall@K and NDCG@K over all test users."""
    user_embed = all_embed[:n_users]
    item_embed = all_embed[n_users : n_users + n_items]

    test_users = sorted(test_dict.keys())
    if not test_users:
        return {"recall": 0.0, "ndcg": 0.0}

    recall_sum = 0.0
    ndcg_sum = 0.0
    count = 0

    # Precompute IDCG denominators
    idcg_table = np.zeros(topk + 1)
    for i in range(1, topk + 1):
        idcg_table[i] = idcg_table[i - 1] + 1.0 / np.log2(i + 1)

    for start in range(0, len(test_users), user_batch_size):
        batch_users = test_users[start : start + user_batch_size]
        batch_user_ids = jnp.array(batch_users, dtype=jnp.int32)

        # Compute scores: (batch, n_items)
        scores = user_embed[batch_user_ids] @ item_embed.T

        # Mask training positives to -inf
        scores_np = np.array(scores)
        for i, uid in enumerate(batch_users):
            if uid in train_dict:
                train_items = train_dict[uid]
                scores_np[i, train_items] = -np.inf

        # Top-K
        top_indices = np.argpartition(-scores_np, topk, axis=1)[:, :topk]
        # Sort within top-K
        for i in range(len(batch_users)):
            order = np.argsort(-scores_np[i, top_indices[i]])
            top_indices[i] = top_indices[i, order]

        # Metrics per user
        for i, uid in enumerate(batch_users):
            test_items = set(test_dict[uid])
            if not test_items:
                continue
            hits = [
                1.0 if top_indices[i, j] in test_items else 0.0 for j in range(topk)
            ]
            n_test = len(test_items)

            # Recall@K
            recall_sum += sum(hits) / min(n_test, topk)

            # NDCG@K
            dcg = sum(hits[j] / np.log2(j + 2) for j in range(topk))
            idcg = idcg_table[min(n_test, topk)]
            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

            count += 1

    return {
        "recall": recall_sum / count if count > 0 else 0.0,
        "ndcg": ndcg_sum / count if count > 0 else 0.0,
    }
