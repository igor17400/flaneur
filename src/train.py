"""Training loop with JIT-compiled train step, periodic eval, and W&B logging."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

import wandb
from data import Dataset, sample_negatives
from evaluate import evaluate
from model import bpr_loss, init_params, lightgcn_forward


def train(cfg, dataset: Dataset):
    """Main training loop."""
    key = jax.random.PRNGKey(cfg.train.seed)
    params = init_params(dataset.n_users, dataset.n_items, cfg.model.embed_dim, key)

    optimizer = optax.adam(cfg.train.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, users, pos_items, neg_items):
        loss, grads = jax.value_and_grad(bpr_loss)(
            params,
            dataset.adj_norm,
            cfg.model.n_layers,
            dataset.n_users,
            users,
            pos_items,
            neg_items,
            cfg.train.reg_weight,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    rng = np.random.default_rng(cfg.train.seed)
    best_recall = 0.0
    best_ndcg = 0.0

    for epoch in trange(1, cfg.train.epochs + 1, desc="Training"):
        t0 = time.time()

        # Sample negatives
        users, pos_items, neg_items = sample_negatives(
            dataset.train_dict, dataset.n_items, rng
        )

        # Shuffle
        perm = rng.permutation(len(users))
        users, pos_items, neg_items = users[perm], pos_items[perm], neg_items[perm]

        # Mini-batch training
        epoch_loss = 0.0
        n_batches = 0
        batch_size = cfg.train.batch_size

        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            batch_u = jnp.array(users[start:end])
            batch_p = jnp.array(pos_items[start:end])
            batch_n = jnp.array(neg_items[start:end])

            params, opt_state, loss = train_step(
                params, opt_state, batch_u, batch_p, batch_n
            )
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - t0

        log_dict = {"train/bpr_loss": avg_loss, "train/epoch_time_sec": epoch_time}

        # Periodic evaluation
        if epoch % cfg.train.eval_every == 0:
            all_embed = lightgcn_forward(params, dataset.adj_norm, cfg.model.n_layers)
            metrics = evaluate(
                all_embed,
                dataset.n_users,
                dataset.n_items,
                dataset.train_dict,
                dataset.test_dict,
                topk=cfg.train.topk,
            )
            recall = metrics["recall"]
            ndcg = metrics["ndcg"]
            log_dict["eval/recall@20"] = recall
            log_dict["eval/ndcg@20"] = ndcg

            if recall > best_recall:
                best_recall = recall
                best_ndcg = ndcg
                if cfg.wandb.enabled:
                    wandb.summary["best_recall@20"] = best_recall
                    wandb.summary["best_ndcg@20"] = best_ndcg

            print(
                f"\nEpoch {epoch}: loss={avg_loss:.4f} | "
                f"Recall@20={recall:.4f} | NDCG@20={ndcg:.4f} | "
                f"Best Recall@20={best_recall:.4f}"
            )

        if cfg.wandb.enabled:
            wandb.log(log_dict, step=epoch)

    return params, best_recall, best_ndcg
