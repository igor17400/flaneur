"""Run inference on ALL users and export predictions for the Derive platform.

Writes predictions/{checkpoint_name}.json with top-K recommendations per user.

Usage:
    uv run python src/infer.py --run dim256_layers4_lr0.001_reg1e-05_neg3_drop0.0
    uv run python src/infer.py --raw          # raw (untrained) baseline
    uv run python src/infer.py --all          # run all checkpoints + raw
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
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
DATA_DIR = PROJECT_ROOT / "data"
TOPK = 20

console = Console()


def list_checkpoints():
    from evaluate_weave import list_checkpoints as _list
    _list()


def _score_and_rank(user_embed, item_embed, train_dict, n_users):
    """Run top-K inference for all users, masking training items."""
    predictions = {}
    for uid in track(range(n_users), description="Running inference"):
        scores = user_embed[uid] @ item_embed.T

        train_items = train_dict.get(uid, [])
        scores[train_items] = -np.inf

        top_k_idx = np.argpartition(-scores, TOPK)[:TOPK]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        predictions[str(uid)] = {
            "items": top_k_idx.tolist(),
            "scores": [round(float(scores[i]), 4) for i in top_k_idx],
        }
    return predictions


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

    predictions = _score_and_rank(user_embed, item_embed, train_dict, n_users)

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

    _save_predictions(output, checkpoint_name, len(predictions))


def run_raw_inference():
    """Generate predictions from a raw (untrained) LightGCN model as baseline."""
    import jax
    import scipy.sparse as sp

    # We need train_dict from any checkpoint to mask training items
    ckpt_dirs = sorted(CHECKPOINT_DIR.iterdir())
    if not ckpt_dirs:
        console.print("[red]No checkpoints found — need at least one for train_dict.[/red]")
        raise SystemExit(1)

    ref_ckpt = ckpt_dirs[0]
    with open(ref_ckpt / "metadata.json") as f:
        meta = json.load(f)

    train_dict = {int(k): v for k, v in meta["train_dict"].items()}
    ref_data = np.load(ref_ckpt / "embeddings.npz")
    n_users = int(ref_data["n_users"])
    n_items = int(ref_data["n_items"])

    console.print(
        f"[bold]Raw baseline:[/bold] {n_users} users, {n_items} items\n"
        f"  Using train_dict from {ref_ckpt.name}"
    )

    # Build normalized adjacency (same as src/data.py _build_adj_norm)
    rows, cols = [], []
    for user, items in train_dict.items():
        for item in items:
            rows.append(user)
            cols.append(n_users + item)
    rows, cols = np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)

    n_nodes = n_users + n_items
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    data = np.ones(len(sym_rows), dtype=np.float32)

    adj = sp.coo_matrix((data, (sym_rows, sym_cols)), shape=(n_nodes, n_nodes)).tocsr()
    degrees = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0.0)
    adj_norm_sp = sp.diags(d_inv_sqrt) @ adj @ sp.diags(d_inv_sqrt)
    adj_norm_sp = adj_norm_sp.tocsr()

    # Xavier-init embeddings (same as model.py init_params)
    embed_dim = 64
    n_layers = 4
    key = jax.random.PRNGKey(42)
    embedding = np.array(
        jax.random.normal(key, (n_nodes, embed_dim)) * (1.0 / np.sqrt(embed_dim))
    )

    # LightGCN propagation using scipy sparse (avoid JAX overhead for one-off)
    console.print(f"  Propagating through {n_layers} GCN layers (embed_dim={embed_dim})...")
    all_embeds = [embedding]
    x = embedding
    for layer in range(n_layers):
        x = adj_norm_sp @ x
        all_embeds.append(x)
    all_embed = np.mean(np.stack(all_embeds, axis=0), axis=0)

    user_embed = all_embed[:n_users]
    item_embed = all_embed[n_users : n_users + n_items]

    predictions = _score_and_rank(user_embed, item_embed, train_dict, n_users)

    output = {
        "model": "raw_untrained",
        "embed_dim": embed_dim,
        "n_layers": n_layers,
        "lr": None,
        "reg_weight": None,
        "epochs": 0,
        "n_negatives": None,
        "val_recall_at_20": None,
        "val_ndcg_at_20": None,
        "n_users": n_users,
        "n_items": n_items,
        "topk": TOPK,
        "users": predictions,
    }

    _save_predictions(output, "raw_untrained", len(predictions))


def _save_predictions(output: dict, name: str, n_preds: int):
    """Write predictions JSON to disk."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / f"{name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    console.print(f"\n[bold green]Predictions saved to {output_path} ({size_mb:.1f} MB)[/bold green]")
    console.print(f"  {n_preds} users x {TOPK} items each")


def run_all():
    """Run inference for all checkpoints + raw baseline."""
    ckpt_dirs = sorted(d.name for d in CHECKPOINT_DIR.iterdir() if d.is_dir())
    console.print(f"[bold]Running inference for {len(ckpt_dirs)} checkpoints + raw baseline...[/bold]\n")

    for name in ckpt_dirs:
        pred_path = PREDICTIONS_DIR / f"{name}.json"
        if pred_path.exists():
            console.print(f"  [dim]Skipping {name} (already exists)[/dim]")
            continue
        console.print(f"\n[bold cyan]>>> {name}[/bold cyan]")
        run_inference(name)

    raw_path = PREDICTIONS_DIR / "raw_untrained.json"
    if raw_path.exists():
        console.print(f"\n  [dim]Skipping raw_untrained (already exists)[/dim]")
    else:
        console.print(f"\n[bold cyan]>>> raw_untrained[/bold cyan]")
        run_raw_inference()

    console.print(f"\n[bold green]All done![/bold green]")
    console.print(f"[bold]Start the platform:[/bold] uv run python derive/server.py")


def main():
    parser = argparse.ArgumentParser(description="Run inference for Derive platform")
    parser.add_argument("--run", type=str, help="Checkpoint name to use")
    parser.add_argument("--raw", action="store_true", help="Generate raw (untrained) baseline predictions")
    parser.add_argument("--all", action="store_true", help="Run all checkpoints + raw baseline")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    args = parser.parse_args()

    if args.list:
        list_checkpoints()
        return

    if args.all:
        run_all()
        return

    if args.raw:
        run_raw_inference()
        return

    if not args.run:
        console.print("[red]Please specify --run <name>, --raw, --all, or --list[/red]")
        raise SystemExit(1)

    run_inference(args.run)


if __name__ == "__main__":
    main()
