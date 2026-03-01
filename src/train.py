"""Training loop with JIT-compiled train step, periodic eval, and W&B logging."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

import wandb
from data import Dataset, sample_negatives
from evaluate import evaluate
from model import bpr_loss, init_params, lightgcn_forward

console = Console()


def train(cfg, dataset: Dataset):
    """Main training loop."""
    key = jax.random.PRNGKey(cfg.train.seed)
    params = init_params(dataset.n_users, dataset.n_items, cfg.model.embed_dim, key)

    # Learning rate schedule: cosine decay
    total_steps = cfg.train.epochs * (dataset.n_train // cfg.train.batch_size + 1)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=cfg.train.lr,
        decay_steps=total_steps,
        alpha=0.01,  # final lr = 1% of initial
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    embed_dropout = getattr(cfg.model, "embed_dropout", 0.0)

    @jax.jit
    def train_step(params, opt_state, users, pos_items, neg_items, dropout_key):
        loss, grads = jax.value_and_grad(bpr_loss)(
            params,
            dataset.adj_norm,
            cfg.model.n_layers,
            dataset.n_users,
            users,
            pos_items,
            neg_items,
            cfg.train.reg_weight,
            embed_dropout,
            dropout_key,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    n_negatives = getattr(cfg.train, "n_negatives", 1)

    rng = np.random.default_rng(cfg.train.seed)
    best_recall = 0.0
    best_ndcg = 0.0
    patience = getattr(cfg.train, "patience", 0)  # 0 = disabled
    epochs_without_improvement = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("• loss={task.fields[loss]:.4f}"),
        console=console,
    )

    with progress:
        task = progress.add_task("Training", total=cfg.train.epochs, loss=0.0)

        for epoch in range(1, cfg.train.epochs + 1):
            t0 = time.time()

            # Sample negatives
            users, pos_items, neg_items = sample_negatives(
                dataset.train_dict, dataset.n_items, rng, n_negatives
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

                key, dropout_key = jax.random.split(key)
                params, opt_state, loss = train_step(
                    params, opt_state, batch_u, batch_p, batch_n, dropout_key
                )
                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            epoch_time = time.time() - t0

            progress.update(task, advance=1, loss=avg_loss)

            log_dict = {"train/bpr_loss": avg_loss, "train/epoch_time_sec": epoch_time}

            # Periodic evaluation on validation set
            if epoch % cfg.train.eval_every == 0:
                all_embed = lightgcn_forward(params, dataset.adj_norm, cfg.model.n_layers)
                val_metrics = evaluate(
                    all_embed,
                    dataset.n_users,
                    dataset.n_items,
                    dataset.train_dict,
                    dataset.val_dict,
                    topk=cfg.train.topk,
                )
                recall = val_metrics["recall"]
                ndcg = val_metrics["ndcg"]
                log_dict["val/recall@20"] = recall
                log_dict["val/ndcg@20"] = ndcg

                if recall > best_recall:
                    best_recall = recall
                    best_ndcg = ndcg
                    epochs_without_improvement = 0
                    if cfg.wandb.enabled:
                        wandb.summary["best_val_recall@20"] = best_recall
                        wandb.summary["best_val_ndcg@20"] = best_ndcg
                else:
                    epochs_without_improvement += cfg.train.eval_every

                console.print(
                    f"  [green]Epoch {epoch}[/green]: loss={avg_loss:.4f} | "
                    f"Val Recall@20={recall:.4f} | Val NDCG@20={ndcg:.4f} | "
                    f"Best Val Recall@20=[bold]{best_recall:.4f}[/bold]"
                )

                # Early stopping
                if patience > 0 and epochs_without_improvement >= patience:
                    console.print(
                        f"  [yellow]Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)[/yellow]"
                    )
                    if cfg.wandb.enabled:
                        wandb.log(log_dict, step=epoch)
                    break

            if cfg.wandb.enabled:
                wandb.log(log_dict, step=epoch)

    return params, best_recall, best_ndcg
