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


def save_checkpoint(params, dataset, cfg, best_recall, best_ndcg):
    """Save embeddings and metadata for Weave evaluation."""
    run_name = cfg.wandb.run_name
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    all_embed = lightgcn_forward(params, dataset.adj_norm, cfg.model.n_layers)
    np.savez(
        run_dir / "embeddings.npz",
        all_embed=np.array(all_embed),
        n_users=dataset.n_users,
        n_items=dataset.n_items,
    )

    meta = {
        "run_name": run_name,
        "best_val_recall@20": best_recall,
        "best_val_ndcg@20": best_ndcg,
        "train_dict": {str(k): v for k, v in dataset.train_dict.items()},
        "val_dict": {str(k): v for k, v in dataset.val_dict.items()},
        "test_dict": {str(k): v for k, v in dataset.test_dict.items()},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    console.print(f"[bold green]Checkpoint saved to {run_dir}/[/bold green]")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print_device_info()
    console.print(OmegaConf.to_yaml(cfg))

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
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
