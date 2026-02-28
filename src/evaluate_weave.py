"""Weave-traced online simulation: recommend → score → Mistral reflects."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import weave
from dotenv import load_dotenv
from mistralai import Mistral
from rich.console import Console
from rich.table import Table

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
console = Console()

# ---------------------------------------------------------------------------
# Global state (set by load_checkpoint)
# ---------------------------------------------------------------------------
user_embed: np.ndarray = None
item_embed: np.ndarray = None
train_dict: dict[int, list[int]] = {}
test_dict: dict[int, list[int]] = {}
item_popularity: np.ndarray = None
run_config: dict = {}
TOPK = 20


def list_checkpoints() -> list[Path]:
    """List all available checkpoints with their metadata."""
    if not CHECKPOINT_DIR.exists():
        console.print("[red]No checkpoints directory found.[/red]")
        return []

    runs = sorted(
        [d for d in CHECKPOINT_DIR.iterdir() if d.is_dir() and (d / "metadata.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        console.print("[red]No checkpoints found.[/red]")
        return []

    table = Table(title="Available Checkpoints")
    table.add_column("#", style="bold")
    table.add_column("Run Name", style="cyan")
    table.add_column("Val Recall@20", style="green")
    table.add_column("Val NDCG@20", style="green")
    table.add_column("Embed Dim")
    table.add_column("Epochs")

    for i, run_dir in enumerate(runs):
        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)
        cfg = meta.get("config", {})
        table.add_row(
            str(i),
            meta.get("run_name", run_dir.name),
            f"{meta.get('best_val_recall@20', '?'):.4f}" if isinstance(meta.get('best_val_recall@20'), float) else "?",
            f"{meta.get('best_val_ndcg@20', '?'):.4f}" if isinstance(meta.get('best_val_ndcg@20'), float) else "?",
            str(cfg.get("model", {}).get("embed_dim", "?")),
            str(cfg.get("train", {}).get("epochs", "?")),
        )

    console.print(table)
    return runs


def load_checkpoint(checkpoint_dir: str):
    """Load saved embeddings and metadata from training."""
    global user_embed, item_embed, train_dict, test_dict, run_config, item_popularity

    ckpt = Path(checkpoint_dir)
    data = np.load(ckpt / "embeddings.npz")
    all_embed = data["all_embed"]
    n_users = int(data["n_users"])
    n_items = int(data["n_items"])

    user_embed = all_embed[:n_users]
    item_embed = all_embed[n_users : n_users + n_items]

    with open(ckpt / "metadata.json") as f:
        meta = json.load(f)

    train_dict = {int(k): v for k, v in meta["train_dict"].items()}
    test_dict = {int(k): v for k, v in meta["test_dict"].items()}
    run_config = meta.get("config", {})

    # Pre-compute item popularity (avoids O(n_users) per scorer call)
    item_popularity = np.zeros(n_items, dtype=np.int32)
    for items in train_dict.values():
        for item in items:
            if item < n_items:
                item_popularity[item] += 1

    console.print(
        f"[bold]Loaded:[/bold] {meta.get('run_name', ckpt.name)} — "
        f"{n_users} users, {n_items} items, embed_dim={user_embed.shape[1]}"
    )


# ---------------------------------------------------------------------------
# Weave ops: recommendation + scorers
# ---------------------------------------------------------------------------


@weave.op
def recommend(user_id: int) -> dict:
    """Generate top-K recommendations for a single test user."""
    scores = user_embed[user_id] @ item_embed.T

    # Mask items seen during training
    train_items = train_dict.get(user_id, [])
    scores[train_items] = -np.inf

    top_k_idx = np.argpartition(-scores, TOPK)[:TOPK]
    top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

    return {
        "user_id": user_id,
        "recommendations": top_k_idx.tolist(),
        "scores": scores[top_k_idx].tolist(),
    }


@weave.op
def recall_scorer(user_id: int, model_output: dict) -> dict:
    """Recall@K: fraction of test items found in top-K."""
    recs = set(model_output["recommendations"])
    ground_truth = test_dict.get(user_id, [])
    if not ground_truth:
        return {"recall": 0.0, "hits": 0, "n_test_items": 0}
    hits = sum(1 for item in ground_truth if item in recs)
    return {
        "recall": hits / min(len(ground_truth), TOPK),
        "hits": hits,
        "n_test_items": len(ground_truth),
    }


@weave.op
def ndcg_scorer(user_id: int, model_output: dict) -> dict:
    """NDCG@K: normalized discounted cumulative gain."""
    recs = model_output["recommendations"]
    ground_truth = set(test_dict.get(user_id, []))
    if not ground_truth:
        return {"ndcg": 0.0}

    dcg = sum(
        (1.0 if recs[i] in ground_truth else 0.0) / np.log2(i + 2)
        for i in range(len(recs))
    )
    n_relevant = min(len(ground_truth), len(recs))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return {"ndcg": dcg / idcg if idcg > 0 else 0.0}


@weave.op
def diversity_scorer(model_output: dict) -> dict:
    """Intra-list diversity: 1 - avg pairwise cosine similarity."""
    recs = model_output["recommendations"]
    embeds = item_embed[recs]
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    embeds_norm = embeds / np.maximum(norms, 1e-8)
    sim = embeds_norm @ embeds_norm.T
    n = len(recs)
    if n < 2:
        return {"diversity": 0.0}
    avg_sim = (sim.sum() - n) / (n * (n - 1))
    return {"diversity": 1.0 - float(avg_sim)}


@weave.op
def coverage_scorer(model_output: dict) -> dict:
    """Per-user item popularity stats (cold items vs popular)."""
    recs = model_output["recommendations"]
    pops = item_popularity[recs]
    return {
        "avg_item_popularity": float(np.mean(pops)),
        "min_item_popularity": int(np.min(pops)),
        "cold_items_recommended": int(np.sum(pops < 5)),
    }


# ---------------------------------------------------------------------------
# Mistral reflection (called after evaluation)
# ---------------------------------------------------------------------------


@weave.op
def mistral_reflect(eval_summary: dict, config: dict) -> dict:
    """Mistral analyzes aggregate evaluation results and suggests improvements."""
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    template = (PROMPTS_DIR / "reflection.txt").read_text()
    prompt = template.format(
        config=json.dumps(config, indent=2),
        eval_summary=json.dumps(eval_summary, indent=2, default=str),
    )

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content

    # Try to parse JSON from response
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        analysis = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        analysis = {"raw_analysis": content}

    return analysis


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


async def run_evaluation(checkpoint_path: Path, n_users: int = 200):
    """Run the full Weave evaluation pipeline."""
    weave.init("flaneur")
    load_checkpoint(str(checkpoint_path))

    # Sample test users (those with ground truth)
    test_users = [uid for uid in sorted(test_dict.keys()) if test_dict[uid]]
    if n_users < len(test_users):
        rng = np.random.default_rng(42)
        sampled = rng.choice(len(test_users), size=n_users, replace=False)
        test_users = [test_users[i] for i in sorted(sampled)]

    dataset = [{"user_id": uid} for uid in test_users]
    console.print(f"Running evaluation on [bold]{len(dataset)}[/bold] test users...")

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[recall_scorer, ndcg_scorer, diversity_scorer, coverage_scorer],
        name=f"eval_{checkpoint_path.name}",
    )

    results = await evaluation.evaluate(recommend)
    console.print("\n[bold green]--- Evaluation Results ---[/bold green]")
    console.print_json(json.dumps(results, default=str))

    # Mistral reflection on aggregate results
    console.print("\n[bold yellow]--- Mistral Reflection ---[/bold yellow]")
    analysis = mistral_reflect(results, run_config)
    console.print_json(json.dumps(analysis, default=str))

    return results, analysis


def select_checkpoint(run_name: str | None = None) -> Path:
    """Select a checkpoint: by name, by index, or interactively."""
    runs = list_checkpoints()
    if not runs:
        raise SystemExit(1)

    if run_name:
        # Match by name
        for run_dir in runs:
            if run_dir.name == run_name:
                return run_dir
        console.print(f"[red]Checkpoint '{run_name}' not found.[/red]")
        raise SystemExit(1)

    # Interactive selection
    if len(runs) == 1:
        console.print(f"Auto-selecting only checkpoint: [cyan]{runs[0].name}[/cyan]")
        return runs[0]

    choice = console.input("\nSelect checkpoint [bold][#][/bold] or [bold][name][/bold]: ").strip()
    if choice.isdigit() and int(choice) < len(runs):
        return runs[int(choice)]
    for run_dir in runs:
        if run_dir.name == choice:
            return run_dir

    console.print("[red]Invalid selection.[/red]")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Weave evaluation for LightGCN")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Checkpoint run name (omit to list and select interactively)",
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=200,
        help="Number of test users to evaluate",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_checkpoints()
        return

    checkpoint_path = select_checkpoint(args.run)
    asyncio.run(run_evaluation(checkpoint_path, args.n_users))


if __name__ == "__main__":
    main()
