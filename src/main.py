"""Entry point: Hydra config + W&B init + training."""

import json
import platform
from pathlib import Path

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import wandb
from data import load_dataset
from model import lightgcn_forward
from train import train

console = Console()

# Project root (flaneur/), not Hydra's output dir
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment"


def parse_config_notes(run_name: str) -> tuple[str, list[str]]:
    """Extract notes and tags from the comment block at the top of a config YAML."""
    config_file = CONFIG_DIR / f"{run_name}.yaml"
    if not config_file.exists():
        return "", []

    lines = config_file.read_text().splitlines()
    comment_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("# @package"):
            comment_lines.append(stripped.lstrip("# ").strip())
        elif stripped and not stripped.startswith("#"):
            break

    notes = " ".join(comment_lines) if comment_lines else ""

    # Auto-generate tags from config values
    tags = []
    try:
        from omegaconf import OmegaConf
        cfg_raw = OmegaConf.load(config_file)
        model = OmegaConf.to_container(cfg_raw, resolve=True).get("model", {})
        train = OmegaConf.to_container(cfg_raw, resolve=True).get("train", {})
        if model.get("embed_dim"):
            tags.append(f"dim={model['embed_dim']}")
        if model.get("n_layers"):
            tags.append(f"layers={model['n_layers']}")
        if train.get("n_negatives", 1) > 1:
            tags.append(f"neg={train['n_negatives']}")
        if model.get("embed_dropout", 0) > 0:
            tags.append(f"drop={model['embed_dropout']}")
    except Exception:
        pass

    return notes, tags


def print_device_info():
    """Print device/platform info prominently."""
    devices = jax.devices()
    backend = jax.default_backend()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Platform", f"{platform.system()} {platform.machine()}")
    table.add_row("Python", platform.python_version())
    table.add_row("JAX backend", backend.upper())
    table.add_row("Devices", ", ".join(f"{d.platform.upper()}:{d.id}" for d in devices))
    table.add_row("Device count", str(len(devices)))

    console.print(Panel(table, title="[bold]Device Info[/bold]", border_style="green"))


def _checkpoint_name(cfg) -> str:
    """Generate a descriptive checkpoint name from key config values."""
    m = cfg.model
    t = cfg.train
    return (
        f"dim{m.embed_dim}_layers{m.n_layers}"
        f"_lr{t.lr}_reg{t.reg_weight}"
        f"_neg{getattr(t, 'n_negatives', 1)}"
        f"_drop{getattr(m, 'embed_dropout', 0.0)}"
    )


def save_checkpoint(params, dataset, cfg, best_recall, best_ndcg):
    """Save embeddings and metadata for Weave evaluation."""
    run_name = cfg.wandb.run_name
    ckpt_name = _checkpoint_name(cfg)

    # Save to both the config-specific dir and the latest dir
    run_dir = CHECKPOINT_DIR / ckpt_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Also keep a "latest" symlink/copy for convenience
    latest_dir = CHECKPOINT_DIR / run_name
    latest_dir.mkdir(parents=True, exist_ok=True)

    all_embed = lightgcn_forward(params, dataset.adj_norm, cfg.model.n_layers)
    embed_data = {
        "all_embed": np.array(all_embed),
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
    }
    meta = {
        "run_name": ckpt_name,
        "best_val_recall@20": best_recall,
        "best_val_ndcg@20": best_ndcg,
        "train_dict": {str(k): v for k, v in dataset.train_dict.items()},
        "val_dict": {str(k): v for k, v in dataset.val_dict.items()},
        "test_dict": {str(k): v for k, v in dataset.test_dict.items()},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    # Save to config-specific dir
    np.savez(run_dir / "embeddings.npz", **embed_data)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    # Also save to latest dir (backwards compatibility)
    np.savez(latest_dir / "embeddings.npz", **embed_data)
    meta_latest = meta.copy()
    meta_latest["run_name"] = run_name
    with open(latest_dir / "metadata.json", "w") as f:
        json.dump(meta_latest, f)

    console.print(f"[bold green]Checkpoint saved to {run_dir}/[/bold green]")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print_device_info()
    console.print(OmegaConf.to_yaml(cfg))

    if cfg.wandb.enabled:
        notes, tags = parse_config_notes(cfg.wandb.run_name)
        run_display_name = _checkpoint_name(cfg)
        wandb.init(
            project=cfg.wandb.project,
            name=run_display_name,
            group=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            notes=notes or None,
            tags=tags or None,
        )

    dataset = load_dataset(cfg.data.data_dir)
    params, best_recall, best_ndcg = train(cfg, dataset)

    console.print(
        f"\n[bold]Training complete.[/bold] "
        f"Best Val Recall@20={best_recall:.4f}, Val NDCG@20={best_ndcg:.4f}"
    )

    save_checkpoint(params, dataset, cfg, best_recall, best_ndcg)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
