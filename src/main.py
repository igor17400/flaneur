"""Entry point: Hydra config + W&B init + training."""

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from data import load_dataset
from train import train


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

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
