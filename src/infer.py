"""Run inference on ALL users and export predictions for the Derive platform.

Writes data/predictions.json with top-K recommendations per user.

Usage:
    uv run python src/infer.py --run dim256_layers4_lr0.001_reg1e-05_neg3_drop0.0
    uv run python src/infer.py --list
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import track

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions.json"
TOPK = 20

console = Console()


def list_checkpoints():
    from evaluate_weave import list_checkpoints as _list
    _list()


def run_inference(checkpoint_name: str):
    ckpt = CHECKPOINT_DIR / checkpoint_name
    if not ckpt.exists():
        console.print(f"[red]Checkpoint '{checkpoint_name}' not found.[/red]")
        raise SystemExit(1)

    # Load embeddings and metadata
    data = np.load(ckpt / "embeddings.npz")
    all_embed = data["all_embed"]
    n_users = int(data["n_users"])
    n_items = int(data["n_items"])

    with open(ckpt / "metadata.json") as f:
        meta = json.load(f)

    user_embed = all_embed[:n_users]
    item_embed = all_embed[n_users : n_users + n_items]

    train_dict = {int(k): v for k, v in meta["train_dict"].items()}
    cfg = meta.get("config", {})

    console.print(
        f"[bold]Loaded:[/bold] {checkpoint_name}\n"
        f"  {n_users} users, {n_items} items, embed_dim={user_embed.shape[1]}"
    )

    # Run inference for all users
    predictions = {}
    for uid in track(range(n_users), description="Running inference"):
        scores = user_embed[uid] @ item_embed.T

        # Mask training items
        train_items = train_dict.get(uid, [])
        scores[train_items] = -np.inf

        top_k_idx = np.argpartition(-scores, TOPK)[:TOPK]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        predictions[str(uid)] = {
            "items": top_k_idx.tolist(),
            "scores": [round(float(scores[i]), 4) for i in top_k_idx],
        }

    # Build output
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    output = {
        "model": checkpoint_name,
        "embed_dim": model_cfg.get("embed_dim"),
        "n_layers": model_cfg.get("n_layers"),
        "lr": train_cfg.get("lr"),
        "reg_weight": train_cfg.get("reg_weight"),
        "epochs": train_cfg.get("epochs"),
        "n_negatives": train_cfg.get("n_negatives"),
        "val_recall_at_20": meta.get("best_val_recall@20"),
        "val_ndcg_at_20": meta.get("best_val_ndcg@20"),
        "n_users": n_users,
        "n_items": n_items,
        "topk": TOPK,
        "users": predictions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    console.print(f"\n[bold green]Predictions saved to {OUTPUT_PATH} ({size_mb:.1f} MB)[/bold green]")
    console.print(f"  {len(predictions)} users x {TOPK} items each")
    console.print(f"\n[bold]Start the platform:[/bold] uv run python derive/server.py")


def main():
    parser = argparse.ArgumentParser(description="Run inference for Derive platform")
    parser.add_argument("--run", type=str, help="Checkpoint name to use")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    args = parser.parse_args()

    if args.list:
        list_checkpoints()
        return

    if not args.run:
        console.print("[red]Please specify --run <checkpoint_name> or use --list[/red]")
        raise SystemExit(1)

    run_inference(args.run)


if __name__ == "__main__":
    main()
