"""Weave-traced online simulation: recommend → score → Mistral reflects."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import weave
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
console = Console()

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


# ---------------------------------------------------------------------------
# Weave Model: LightGCN recommender with config tracking
# ---------------------------------------------------------------------------


class LightGCNModel(weave.Model):
    """LightGCN recommender model — Weave tracks all attributes in the leaderboard."""

    # Model config
    embed_dim: int
    n_layers: int
    embed_dropout: float

    # Training config
    lr: float
    reg_weight: float
    batch_size: int
    epochs: int
    n_negatives: int

    # Validation metrics (from training)
    val_recall_at_20: float
    val_ndcg_at_20: float

    # Runtime state (excluded from Weave versioning)
    _user_embed: np.ndarray = None
    _item_embed: np.ndarray = None
    _train_dict: dict = {}
    _test_dict: dict = {}
    _item_popularity: np.ndarray = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "LightGCNModel":
        """Load a model from a training checkpoint."""
        ckpt = Path(checkpoint_dir)
        data = np.load(ckpt / "embeddings.npz")
        all_embed = data["all_embed"]
        n_users = int(data["n_users"])
        n_items = int(data["n_items"])

        with open(ckpt / "metadata.json") as f:
            meta = json.load(f)

        cfg = meta.get("config", {})
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("train", {})

        model = cls(
            embed_dim=model_cfg.get("embed_dim", 64),
            n_layers=model_cfg.get("n_layers", 3),
            embed_dropout=model_cfg.get("embed_dropout", 0.0),
            lr=train_cfg.get("lr", 1e-3),
            reg_weight=train_cfg.get("reg_weight", 1e-5),
            batch_size=train_cfg.get("batch_size", 2048),
            epochs=train_cfg.get("epochs", 100),
            n_negatives=train_cfg.get("n_negatives", 1),
            val_recall_at_20=meta.get("best_val_recall@20", 0.0),
            val_ndcg_at_20=meta.get("best_val_ndcg@20", 0.0),
        )

        model._user_embed = all_embed[:n_users]
        model._item_embed = all_embed[n_users : n_users + n_items]
        model._train_dict = {int(k): v for k, v in meta["train_dict"].items()}
        model._test_dict = {int(k): v for k, v in meta["test_dict"].items()}
        model._config = cfg

        # Pre-compute item popularity
        item_popularity = np.zeros(n_items, dtype=np.int32)
        for items in model._train_dict.values():
            for item in items:
                if item < n_items:
                    item_popularity[item] += 1
        model._item_popularity = item_popularity

        console.print(
            f"[bold]Loaded:[/bold] {meta.get('run_name', ckpt.name)} — "
            f"{n_users} users, {n_items} items, embed_dim={model.embed_dim}, "
            f"reg={model.reg_weight}, lr={model.lr}, layers={model.n_layers}"
        )
        return model

    @weave.op
    def predict(self, user_id: int) -> dict:
        """Generate top-K recommendations for a single test user."""
        scores = self._user_embed[user_id] @ self._item_embed.T

        # Mask items seen during training
        train_items = self._train_dict.get(user_id, [])
        scores[train_items] = -np.inf

        top_k_idx = np.argpartition(-scores, TOPK)[:TOPK]
        top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

        return {
            "user_id": user_id,
            "recommendations": top_k_idx.tolist(),
            "scores": scores[top_k_idx].tolist(),
        }


# ---------------------------------------------------------------------------
# Weave Model: PredictionModel — loads pre-computed predictions from JSON
# ---------------------------------------------------------------------------


class PredictionModel(weave.Model):
    """Pre-computed predictions model — looks up results from infer.py JSON files."""

    # Model config (mirrors LightGCNModel for Weave leaderboard comparison)
    embed_dim: int
    n_layers: int
    embed_dropout: float

    # Training config
    lr: float
    reg_weight: float
    batch_size: int
    epochs: int
    n_negatives: int

    # Validation metrics
    val_recall_at_20: float
    val_ndcg_at_20: float

    # Source
    prediction_source: str

    # Runtime state
    _predictions: dict = {}
    _train_dict: dict = {}
    _test_dict: dict = {}
    _item_popularity: np.ndarray = None
    _item_embed: np.ndarray = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_prediction_file(cls, pred_path: Path, checkpoint_dir: Path | None = None) -> "PredictionModel":
        """Load a model from a predictions JSON file.

        Args:
            pred_path: Path to predictions/*.json
            checkpoint_dir: Optional checkpoint dir for test_dict/train_dict/item_embed.
                           If None, searches for any available checkpoint.
        """
        with open(pred_path) as f:
            data = json.load(f)

        pred_name = pred_path.stem

        # Build predictions lookup: {int(uid) → {items, scores}}
        predictions = {}
        for uid_str, entry in data["users"].items():
            predictions[int(uid_str)] = {
                "items": entry["items"],
                "scores": entry["scores"],
            }

        model = cls(
            embed_dim=data.get("embed_dim") or 64,
            n_layers=data.get("n_layers") or 4,
            embed_dropout=0.0,
            lr=data.get("lr") or 0.0,
            reg_weight=data.get("reg_weight") or 0.0,
            batch_size=0,
            epochs=data.get("epochs") or 0,
            n_negatives=data.get("n_negatives") or 0,
            val_recall_at_20=data.get("val_recall_at_20") or 0.0,
            val_ndcg_at_20=data.get("val_ndcg_at_20") or 0.0,
            prediction_source=pred_name,
        )
        model._predictions = predictions

        # Load supplementary data from a checkpoint
        ckpt_dir = checkpoint_dir
        if ckpt_dir is None:
            # Try matching checkpoint first, then fall back to any available
            matching = CHECKPOINT_DIR / pred_name
            if matching.exists() and (matching / "metadata.json").exists():
                ckpt_dir = matching
            else:
                available = [
                    d for d in sorted(CHECKPOINT_DIR.iterdir())
                    if d.is_dir() and (d / "metadata.json").exists()
                ]
                if available:
                    ckpt_dir = available[0]

        if ckpt_dir and (ckpt_dir / "metadata.json").exists():
            with open(ckpt_dir / "metadata.json") as f:
                meta = json.load(f)
            model._train_dict = {int(k): v for k, v in meta["train_dict"].items()}
            model._test_dict = {int(k): v for k, v in meta["test_dict"].items()}

            # Item popularity from train_dict
            n_items = data.get("n_items", 0)
            if n_items > 0:
                item_popularity = np.zeros(n_items, dtype=np.int32)
                for items in model._train_dict.values():
                    for item in items:
                        if item < n_items:
                            item_popularity[item] += 1
                model._item_popularity = item_popularity

            # Item embeddings: only from matching checkpoint
            matching_ckpt = CHECKPOINT_DIR / pred_name
            if matching_ckpt.exists() and (matching_ckpt / "embeddings.npz").exists():
                emb_data = np.load(matching_ckpt / "embeddings.npz")
                n_users = int(emb_data["n_users"])
                n_items_emb = int(emb_data["n_items"])
                model._item_embed = emb_data["all_embed"][n_users : n_users + n_items_emb]
            else:
                model._item_embed = None

            console.print(
                f"[bold]Loaded predictions:[/bold] {pred_name} — "
                f"{len(predictions)} users, embed_dim={model.embed_dim}, "
                f"supplementary data from {ckpt_dir.name}"
                + (" (no item embeddings — diversity will be skipped)" if model._item_embed is None else "")
            )
        else:
            console.print(
                f"[bold]Loaded predictions:[/bold] {pred_name} — "
                f"{len(predictions)} users, embed_dim={model.embed_dim} "
                f"[yellow](no checkpoint found — limited scorers)[/yellow]"
            )

        return model

    @weave.op
    def predict(self, user_id: int) -> dict:
        """Look up pre-computed recommendations for a user."""
        entry = self._predictions.get(user_id)
        if entry is None:
            return {"user_id": user_id, "recommendations": [], "scores": []}
        return {
            "user_id": user_id,
            "recommendations": entry["items"],
            "scores": entry["scores"],
        }


@weave.op
def recall_scorer(user_id: int, output: dict) -> dict:
    """Recall@K: fraction of test items found in top-K."""
    recs = set(output["recommendations"])
    ground_truth = _active_model._test_dict.get(user_id, [])
    if not ground_truth:
        return {"recall": 0.0, "hits": 0, "n_test_items": 0}
    hits = sum(1 for item in ground_truth if item in recs)
    return {
        "recall": hits / min(len(ground_truth), TOPK),
        "hits": hits,
        "n_test_items": len(ground_truth),
    }


@weave.op
def ndcg_scorer(user_id: int, output: dict) -> dict:
    """NDCG@K: normalized discounted cumulative gain."""
    recs = output["recommendations"]
    ground_truth = set(_active_model._test_dict.get(user_id, []))
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
def diversity_scorer(output: dict) -> dict:
    """Intra-list diversity: 1 - avg pairwise cosine similarity."""
    if _active_model._item_embed is None:
        return {"diversity": None}
    recs = output["recommendations"]
    if not recs:
        return {"diversity": None}
    embeds = _active_model._item_embed[recs]
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    embeds_norm = embeds / np.maximum(norms, 1e-8)
    sim = embeds_norm @ embeds_norm.T
    n = len(recs)
    if n < 2:
        return {"diversity": 0.0}
    avg_sim = (sim.sum() - n) / (n * (n - 1))
    return {"diversity": 1.0 - float(avg_sim)}


@weave.op
def coverage_scorer(output: dict) -> dict:
    """Per-user item popularity stats (cold items vs popular)."""
    if _active_model._item_popularity is None:
        return {"avg_item_popularity": None, "min_item_popularity": None, "cold_items_recommended": None}
    recs = output["recommendations"]
    if not recs:
        return {"avg_item_popularity": None, "min_item_popularity": None, "cold_items_recommended": None}
    pops = _active_model._item_popularity[recs]
    return {
        "avg_item_popularity": float(np.mean(pops)),
        "min_item_popularity": int(np.min(pops)),
        "cold_items_recommended": int(np.sum(pops < 5)),
    }


# Global reference to the active model (set before evaluation)
_active_model: LightGCNModel | PredictionModel = None


# ---------------------------------------------------------------------------
# Mistral reflection — disabled for now, to be used in the platform layer.
# The function is preserved for future integration where Mistral can:
# - Compare traces across A/B groups
# - Analyze per-user failure patterns
# - Generate stakeholder summaries
# - Track its own reasoning quality via Weave traces
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def _human_model_name(model) -> str:
    """Build a short, readable name like 'LightGCN-d256-L4-reg1e-05' or 'raw-untrained'."""
    if isinstance(model, PredictionModel):
        src = model.prediction_source
        if "raw" in src or "untrained" in src:
            return "raw-untrained"
    # Format reg_weight compactly: 1e-05, 0.001, etc.
    reg = getattr(model, "reg_weight", 0)
    if reg and reg > 0:
        reg_str = f"{reg:.0e}".replace("+", "").replace("0", "").rstrip("e")
        # Clean up: "1e-05" → "1e-5", "1e-04" → "1e-4"
        reg_str = f"-reg{reg:.0e}"
    else:
        reg_str = ""
    return f"LightGCN-d{model.embed_dim}-L{model.n_layers}{reg_str}"


async def run_evaluation(
    checkpoint_path: Path | None = None,
    n_users: int = 200,
    user_ids: list[int] | None = None,
    model: weave.Model | None = None,
    eval_label: str | None = None,
):
    """Run the full Weave evaluation pipeline.

    Provide either checkpoint_path (loads LightGCNModel) or a pre-built model.
    eval_label overrides the auto-generated evaluation name.
    """
    global _active_model

    weave.init("flaneur")
    if model is None:
        model = LightGCNModel.from_checkpoint(str(checkpoint_path))
    _active_model = model

    if user_ids is not None:
        # Use pre-assigned user IDs (e.g., from A/B test split)
        test_users = user_ids
    else:
        # Sample test users (those with ground truth)
        test_users = [uid for uid in sorted(model._test_dict.keys()) if model._test_dict[uid]]
        if n_users < len(test_users):
            rng = np.random.default_rng(42)
            sampled = rng.choice(len(test_users), size=n_users, replace=False)
            test_users = [test_users[i] for i in sorted(sampled)]

    dataset = [{"user_id": uid} for uid in test_users]
    console.print(f"Running evaluation on [bold]{len(dataset)}[/bold] test users...")

    if eval_label:
        eval_name = eval_label
    else:
        model_name = _human_model_name(model)
        n = len(dataset)
        eval_name = f"{model_name} ({n} users)"

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[recall_scorer, ndcg_scorer, diversity_scorer, coverage_scorer],
        name=eval_name,
        evaluation_name=eval_name,
    )

    results = await evaluation.evaluate(model)
    console.print("\n[bold green]--- Evaluation Results ---[/bold green]")
    console.print_json(json.dumps(results, default=str))

    # Publish leaderboard for side-by-side model comparison in Weave UI
    try:
        from weave.flow import leaderboard as lb
        from weave.trace.ref_util import get_ref

        eval_ref = get_ref(evaluation)
        if eval_ref:
            spec = lb.Leaderboard(
                name="LightGCN Gowalla",
                description="Compare LightGCN checkpoints on held-out test users",
                columns=[
                    lb.LeaderboardColumn(
                        evaluation_object_ref=eval_ref.uri(),
                        scorer_name="recall_scorer",
                        summary_metric_path="recall.mean",
                    ),
                    lb.LeaderboardColumn(
                        evaluation_object_ref=eval_ref.uri(),
                        scorer_name="ndcg_scorer",
                        summary_metric_path="ndcg.mean",
                    ),
                    lb.LeaderboardColumn(
                        evaluation_object_ref=eval_ref.uri(),
                        scorer_name="diversity_scorer",
                        summary_metric_path="diversity.mean",
                    ),
                ],
            )
            weave.publish(spec)
            console.print("[bold green]Leaderboard published to Weave UI (Leaders tab)[/bold green]")
    except Exception as e:
        console.print(f"[yellow]Leaderboard publish skipped: {e}[/yellow]")

    return results


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
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Compare multiple checkpoints on SAME users: --compare ckpt1 ckpt2",
    )
    parser.add_argument(
        "--ab-test",
        type=str,
        nargs=2,
        default=None,
        help="A/B test with non-overlapping user groups: --ab-test model_a model_b",
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=200,
        help="Number of test users to evaluate",
    )
    parser.add_argument(
        "--from-predictions",
        type=str,
        nargs="*",
        default=None,
        metavar="NAME",
        help="Evaluate pre-computed prediction files from predictions/*.json. "
             "Use as bare flag with --ab-test to load predictions by ab-test names.",
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

    # --from-predictions mode (list is non-None when flag is present, even if empty)
    if args.from_predictions is not None:
        pred_names = args.from_predictions

        if args.ab_test:
            # A/B test with prediction files — use ab-test names if --from-predictions is bare
            if not pred_names:
                pred_names = args.ab_test
            if len(pred_names) != 2:
                console.print("[red]A/B test with --from-predictions requires exactly 2 names.[/red]")
                raise SystemExit(1)

            console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
            console.print(f"[bold magenta]A/B Test (from predictions)[/bold magenta]")
            console.print(f"[bold magenta]{'='*60}[/bold magenta]")
            console.print(f"  Group A: {pred_names[0]}")
            console.print(f"  Group B: {pred_names[1]}")
            console.print(f"  Users per group: {args.n_users}")
            console.print()

            # Load both models to get test users
            models = []
            for name in pred_names:
                pred_path = PREDICTIONS_DIR / f"{name}.json"
                if not pred_path.exists():
                    console.print(f"[red]Prediction file not found: {pred_path}[/red]")
                    raise SystemExit(1)
                models.append(PredictionModel.from_prediction_file(pred_path))

            # Get test users from the first model (shared dataset)
            all_test_users = sorted(
                uid for uid in models[0]._test_dict if models[0]._test_dict[uid]
            )

            total_needed = args.n_users * 2
            rng = np.random.default_rng(42)
            if total_needed <= len(all_test_users):
                sampled_indices = rng.choice(len(all_test_users), size=total_needed, replace=False)
                sampled_indices.sort()
                sampled_users = [all_test_users[i] for i in sampled_indices]
            else:
                sampled_users = all_test_users

            mid = len(sampled_users) // 2
            group_a_users = sampled_users[:mid]
            group_b_users = sampled_users[mid:]

            console.print(f"  Group A users: {len(group_a_users)}")
            console.print(f"  Group B users: {len(group_b_users)}")
            console.print()

            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Group A → {pred_names[0]}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            name_a = _human_model_name(models[0])
            name_b = _human_model_name(models[1])
            asyncio.run(run_evaluation(
                model=models[0], n_users=args.n_users, user_ids=group_a_users,
                eval_label=f"A/B Group A: {name_a} ({len(group_a_users)} users)",
            ))

            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Group B → {pred_names[1]}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            asyncio.run(run_evaluation(
                model=models[1], n_users=args.n_users, user_ids=group_b_users,
                eval_label=f"A/B Group B: {name_b} ({len(group_b_users)} users)",
            ))

            console.print(f"\n[bold green]A/B test complete![/bold green]")
            return

        # Standard --from-predictions: evaluate each prediction file
        if not pred_names:
            console.print("[red]--from-predictions requires at least one prediction name.[/red]")
            raise SystemExit(1)
        for name in pred_names:
            pred_path = PREDICTIONS_DIR / f"{name}.json"
            if not pred_path.exists():
                console.print(f"[red]Prediction file not found: {pred_path}[/red]")
                raise SystemExit(1)

            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Evaluating predictions: {name}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            pred_model = PredictionModel.from_prediction_file(pred_path)
            asyncio.run(run_evaluation(model=pred_model, n_users=args.n_users))

        if len(pred_names) > 1:
            console.print(f"\n[bold green]All {len(pred_names)} prediction files evaluated![/bold green]")
            console.print("[bold]Compare them in the Weave leaderboard.[/bold]")
        return

    if args.ab_test:
        # A/B test: split users into non-overlapping groups
        if len(args.ab_test) != 2:
            console.print("[red]A/B test requires exactly 2 checkpoints.[/red]")
            raise SystemExit(1)

        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]A/B Test Simulation[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"  Group A: {args.ab_test[0]}")
        console.print(f"  Group B: {args.ab_test[1]}")
        console.print(f"  Users per group: {args.n_users}")
        console.print()

        # Load one checkpoint to get test users
        path_a = select_checkpoint(args.ab_test[0])
        path_b = select_checkpoint(args.ab_test[1])

        # Get all test users from checkpoint A (same data for both)
        import json as _json
        with open(path_a / "metadata.json") as f:
            meta = _json.load(f)
        all_test_users = [int(k) for k, v in meta["test_dict"].items() if v]
        all_test_users.sort()

        total_needed = args.n_users * 2
        rng = np.random.default_rng(42)
        if total_needed <= len(all_test_users):
            sampled_indices = rng.choice(len(all_test_users), size=total_needed, replace=False)
            sampled_indices.sort()
            sampled_users = [all_test_users[i] for i in sampled_indices]
        else:
            sampled_users = all_test_users

        # Split into two non-overlapping groups
        mid = len(sampled_users) // 2
        group_a_users = sampled_users[:mid]
        group_b_users = sampled_users[mid:]

        console.print(f"  Group A users: {len(group_a_users)} (IDs {group_a_users[0]}..{group_a_users[-1]})")
        console.print(f"  Group B users: {len(group_b_users)} (IDs {group_b_users[0]}..{group_b_users[-1]})")
        console.print()

        # Evaluate Group A with Model A
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Group A → Model: {args.ab_test[0]}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        asyncio.run(run_evaluation(
            path_a, args.n_users, user_ids=group_a_users,
            eval_label=f"A/B Group A: {args.ab_test[0]} ({len(group_a_users)} users)",
        ))

        # Evaluate Group B with Model B
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Group B → Model: {args.ab_test[1]}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        asyncio.run(run_evaluation(
            path_b, args.n_users, user_ids=group_b_users,
            eval_label=f"A/B Group B: {args.ab_test[1]} ({len(group_b_users)} users)",
        ))

        console.print(f"\n[bold green]A/B test complete![/bold green]")
        console.print("[bold]Compare groups in the Weave leaderboard — each group has different users, simulating real traffic splitting.[/bold]")
        return

    if args.compare:
        # Evaluate multiple checkpoints on the SAME users (controlled comparison)
        for ckpt_name in args.compare:
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Evaluating: {ckpt_name}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            checkpoint_path = select_checkpoint(ckpt_name)
            asyncio.run(run_evaluation(checkpoint_path, args.n_users))
        console.print(f"\n[bold green]All {len(args.compare)} models evaluated![/bold green]")
        console.print("[bold]Compare them in the Weave leaderboard.[/bold]")
        return

    checkpoint_path = select_checkpoint(args.run)
    asyncio.run(run_evaluation(checkpoint_path, args.n_users))


if __name__ == "__main__":
    main()
