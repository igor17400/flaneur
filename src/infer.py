"""Run LightGCN inference and save top-K predictions for Derive visualization.

Usage:
    cd flaneur_main_hackatho
    uv run python src/infer.py                          # auto-pick 3 users
    uv run python src/infer.py --users 0 100 1000       # specific users
    uv run python src/infer.py --pick 5 --top_k 30      # pick 5, top-30
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data import load_dataset
from model import init_params, lightgcn_forward


def get_top_k(all_embed, user_id, n_users, n_items, train_items, k=20):
    """Get top-K item predictions for a user, excluding training items."""
    user_embed = all_embed[user_id]
    item_embed = all_embed[n_users : n_users + n_items]
    scores = jnp.dot(item_embed, user_embed)

    scores_np = np.array(scores)
    for item in train_items:
        scores_np[item] = -np.inf

    top_k_idx = np.argsort(-scores_np)[:k]
    return top_k_idx.tolist(), scores_np[top_k_idx].tolist()


def pick_users(dataset, n=3, seed=42):
    """Pick users with both train and test data and enough interactions."""
    rng = np.random.default_rng(seed)
    candidates = [
        uid
        for uid in dataset.train_dict
        if uid in dataset.test_dict
        and len(dataset.train_dict[uid]) >= 5
        and len(dataset.test_dict[uid]) >= 3
    ]
    chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    return sorted(chosen.tolist())


def main():
    parser = argparse.ArgumentParser(description="LightGCN inference for Derive")
    parser.add_argument("--users", type=int, nargs="+", help="Specific user IDs")
    parser.add_argument("--pick", type=int, default=3, help="Auto-pick N users")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = str(Path(__file__).resolve().parent.parent / "data" / "gowalla")

    print("Loading dataset...")
    dataset = load_dataset(data_dir)

    if args.users:
        user_ids = args.users
    else:
        user_ids = pick_users(dataset, n=args.pick, seed=args.seed)

    print(f"Users: {user_ids}")

    key = jax.random.PRNGKey(args.seed)

    if args.checkpoint:
        raise NotImplementedError("Checkpoint loading not yet implemented")
    else:
        print("Initializing untrained model (random embeddings)...")
        params = init_params(dataset.n_users, dataset.n_items, args.embed_dim, key)
        model_name = "untrained"

    print("Running forward pass...")
    all_embed = lightgcn_forward(params, dataset.adj_norm, args.n_layers)

    predictions = {}
    for uid in user_ids:
        train_items = dataset.train_dict.get(uid, [])
        test_set = set(dataset.test_dict.get(uid, []))
        top_items, top_scores = get_top_k(
            all_embed, uid, dataset.n_users, dataset.n_items, train_items, args.top_k
        )
        hits = [it for it in top_items if it in test_set]
        predictions[str(uid)] = {
            "items": top_items,
            "scores": [round(s, 6) for s in top_scores],
            "n_train": len(train_items),
            "n_test": len(test_set),
        }
        print(
            f"  User {uid}: {len(train_items)} train, {len(test_set)} test, "
            f"{len(hits)}/{args.top_k} hits"
        )

    output = {
        "model": model_name,
        "embed_dim": args.embed_dim,
        "n_layers": args.n_layers,
        "top_k": args.top_k,
        "users": predictions,
    }

    out_path = Path(__file__).resolve().parent.parent / "data" / "predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    print(f"Users: {user_ids} — load these in Derive to see predictions.")


if __name__ == "__main__":
    main()
