from __future__ import annotations

import time

import math
from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassAccuracy

from structured_dendrite.models import ClassificationModel, LanguageModel


def _build_logger(cfg):
    save_dir = cfg.logging.save_dir
    if cfg.logging.kind == "tensorboard":
        return TensorBoardLogger(save_dir=save_dir, name=cfg.logging.name)
    return CSVLogger(save_dir=save_dir, name=cfg.logging.name)


def _cosine_with_warmup_lambda(current_step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _build_optimizer_param_groups(module) -> list[dict]:
    special_groups: dict[tuple[tuple[str, float], ...], list[torch.nn.Parameter]] = defaultdict(list)
    default_parameters: list[torch.nn.Parameter] = []

    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        overrides = getattr(parameter, "_optim", None)
        if overrides:
            key = tuple(sorted((name, float(value)) for name, value in overrides.items()))
            special_groups[key].append(parameter)
        else:
            default_parameters.append(parameter)

    parameter_groups: list[dict] = []
    if default_parameters:
        parameter_groups.append({"params": default_parameters})
    for key, parameters in special_groups.items():
        parameter_groups.append({"params": parameters, **dict(key)})
    return parameter_groups


class DendriteExperiment(L.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.test_accuracy = None
        self._fit_start_time = None

    def setup(self, stage: str | None = None) -> None:
        if self.model is not None:
            return

        dataset_info = self.trainer.datamodule.info
        if self.cfg.task.name == "classification":
            self.model = ClassificationModel(self.cfg.model, self.cfg.task, dataset_info)
            self.train_accuracy = MulticlassAccuracy(num_classes=int(dataset_info.num_classes))
            self.val_accuracy = MulticlassAccuracy(num_classes=int(dataset_info.num_classes))
            self.test_accuracy = MulticlassAccuracy(num_classes=int(dataset_info.num_classes))
        elif self.cfg.task.name == "language_modeling":
            self.model = LanguageModel(self.cfg.model, self.cfg.task, dataset_info)
        else:
            raise ValueError(f"Unsupported task: {self.cfg.task.name}")

    def forward(self, batch):
        return self.model(batch)

    def on_fit_start(self) -> None:
        self._fit_start_time = time.perf_counter()
        if self.logger is None or self.model is None:
            return

        total_params = float(sum(parameter.numel() for parameter in self.model.parameters()))
        trainable_params = float(sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad))
        self.logger.log_metrics({
            "meta/param_count": total_params,
            "meta/trainable_param_count": trainable_params,
        }, step=0)

        hparams = {
            "run_name": self.cfg.run.name,
            "task_name": self.cfg.data.task_name,
            "monitor_metric": self.cfg.task.monitor_metric,
        }
        if self.device.type == "cuda":
            hparams["gpu_name"] = torch.cuda.get_device_name(self.device)
        self.logger.log_hyperparams(hparams)

    def on_fit_end(self) -> None:
        if self.logger is None:
            return
        if self._fit_start_time is not None:
            self.logger.log_metrics({"meta/fit_wall_clock_sec": time.perf_counter() - self._fit_start_time}, step=int(self.global_step))

        checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
        if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None:
            metric_name = f"best/{self.cfg.task.monitor_metric.replace('/', '_')}"
            self.logger.log_metrics({metric_name: float(checkpoint_callback.best_model_score)}, step=int(self.global_step))

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        del batch_idx
        self._shared_step(batch, stage="test")

    def _shared_step(self, batch, stage: str):
        outputs = self.model(batch)
        spike_rate = outputs["spike_rate"]

        if self.cfg.task.name == "classification":
            logits = outputs["logits"]
            labels = batch["labels"]
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=-1)

            metric = {
                "train": self.train_accuracy,
                "val": self.val_accuracy,
                "test": self.test_accuracy,
            }[stage]
            accuracy = metric(predictions, labels)

            self.log(f"{stage}/loss", loss, prog_bar=stage != "train", on_step=stage == "train", on_epoch=True)
            self.log(f"{stage}/accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        else:
            logits = outputs["logits"]
            labels = batch["labels"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=int(self.trainer.datamodule.info.pad_token_id or 0),
            )
            perplexity = torch.exp(loss.detach())
            self.log(f"{stage}/loss", loss, prog_bar=stage != "train", on_step=stage == "train", on_epoch=True)
            self.log(f"{stage}/perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True)

        self.log(f"{stage}/spike_rate", spike_rate, prog_bar=False, on_step=stage == "train", on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            _build_optimizer_param_groups(self),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
            betas=tuple(self.cfg.optimizer.betas),
        )

        if self.cfg.scheduler.name == "none":
            return optimizer

        total_steps = int(self.cfg.scheduler.total_steps) or int(self.trainer.estimated_stepping_batches)
        scheduler = LambdaLR(
            optimizer,
            lambda step: _cosine_with_warmup_lambda(
                current_step=step,
                warmup_steps=int(self.cfg.scheduler.warmup_steps),
                total_steps=total_steps,
                min_lr_ratio=float(self.cfg.scheduler.min_lr_ratio),
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def run_experiment(cfg, module: DendriteExperiment, datamodule) -> None:
    logger = _build_logger(cfg)
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.task.monitor_metric,
            mode=cfg.task.monitor_mode,
            save_top_k=cfg.logging.checkpoint.save_top_k,
            filename="{epoch:03d}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if bool(cfg.logging.rich_model_summary):
        callbacks.append(RichModelSummary(max_depth=2))
    if bool(cfg.logging.rich_progress_bar):
        callbacks.append(RichProgressBar())

    devices = cfg.trainer.devices
    if OmegaConf.is_config(devices):
        devices = OmegaConf.to_container(devices, resolve=True)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=devices,
        max_epochs=int(cfg.trainer.max_epochs),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        precision=cfg.trainer.precision,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        benchmark=bool(cfg.trainer.benchmark),
        deterministic=bool(cfg.trainer.deterministic),
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        num_sanity_val_steps=int(cfg.trainer.num_sanity_val_steps),
        fast_dev_run=bool(cfg.trainer.fast_dev_run),
        callbacks=callbacks,
        logger=logger,
    )

    if bool(cfg.run.test_only):
        trainer.test(module, datamodule=datamodule, ckpt_path=cfg.run.resume_from)
        return

    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.run.resume_from)
    if bool(cfg.run.test_after_fit):
        trainer.test(module, datamodule=datamodule, ckpt_path="best")

