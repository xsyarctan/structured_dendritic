from __future__ import annotations

import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from structured_dendrite.data import FlexibleSequenceDataModule
from structured_dendrite.experiment import DendriteExperiment, run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    seed_everything(cfg.run.seed, workers=True)

    datamodule = FlexibleSequenceDataModule(cfg.data)
    module = DendriteExperiment(cfg)
    run_experiment(cfg, module, datamodule)


if __name__ == "__main__":
    main()
