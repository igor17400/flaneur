"""Entry point: Hydra config + W&B init + training."""

import platform

import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import wandb
from data import load_dataset
from train import train

console = Console()


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

    print(
        f"\nTraining complete. Best Recall@20={best_recall:.4f}, NDCG@20={best_ndcg:.4f}"
    )

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
