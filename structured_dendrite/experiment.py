from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassAccuracy

from structured_dendrite.models import ClassificationModel, L5PCCNNBaseline, L5PCSequenceModel, LanguageModel


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


def _logger_dir(module: L.LightningModule) -> Path:
    if module.logger is not None and getattr(module.logger, "log_dir", None):
        return Path(module.logger.log_dir)
    return Path(module.trainer.default_root_dir)


def _partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    if max_fpr <= 0:
        return float("nan")
    if len(fpr) == 0:
        return float("nan")
    clipped_tpr = np.interp(max_fpr, fpr, tpr)
    mask = fpr < max_fpr
    fpr_segment = np.concatenate([fpr[mask], [max_fpr]])
    tpr_segment = np.concatenate([tpr[mask], [clipped_tpr]])
    return float(np.trapezoid(tpr_segment, fpr_segment) / max_fpr)


def _threshold_at_fpr(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, desired_fpr: float) -> tuple[float, float, float]:
    if len(thresholds) == 0:
        return float("nan"), float("nan"), float("nan")
    index = int(np.argmin(np.abs(fpr - desired_fpr)))
    if index == 0 and len(thresholds) > 1:
        index = 1
    return float(thresholds[index]), float(fpr[index]), float(tpr[index])


def _spike_alignment_curve(binary_pred: np.ndarray, label_spikes: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    pred = binary_pred.astype(np.float32)
    truth = label_spikes.astype(np.float32)
    pred_center = pred - pred.mean(axis=1, keepdims=True)
    truth_center = truth - truth.mean(axis=1, keepdims=True)
    denom = np.sqrt((pred_center * pred_center).sum(axis=1) * (truth_center * truth_center).sum(axis=1))
    valid = denom > 0
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    if not np.any(valid):
        curve = np.full_like(lags, np.nan, dtype=np.float32)
        return lags, curve, float("nan"), 0

    curve_values: list[float] = []
    for lag in lags:
        if lag < 0:
            pred_slice = pred_center[:, :lag]
            truth_slice = truth_center[:, -lag:]
        elif lag > 0:
            pred_slice = pred_center[:, lag:]
            truth_slice = truth_center[:, :-lag]
        else:
            pred_slice = pred_center
            truth_slice = truth_center
        numerator = (pred_slice * truth_slice).sum(axis=1)
        correlation = np.zeros_like(denom, dtype=np.float32)
        correlation[valid] = numerator[valid] / denom[valid]
        curve_values.append(float(np.nanmean(correlation[valid])))

    curve = np.asarray(curve_values, dtype=np.float32)
    peak_index = int(np.nanargmax(curve))
    return lags, curve, float(curve[peak_index]), int(lags[peak_index])


class BaseExperiment(L.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = None
        self._fit_start_time = None

    def on_fit_start(self) -> None:
        self._fit_start_time = time.perf_counter()
        if self.logger is None or self.model is None:
            return

        total_params = float(sum(parameter.numel() for parameter in self.model.parameters()))
        trainable_params = float(sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad))
        self.logger.log_metrics(
            {
                "meta/param_count": total_params,
                "meta/trainable_param_count": trainable_params,
            },
            step=0,
        )

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


class DendriteExperiment(BaseExperiment):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.train_accuracy = None
        self.val_accuracy = None
        self.test_accuracy = None

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
        sync_dist = bool(getattr(self.trainer, "world_size", 1) > 1)

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

            self.log(
                f"{stage}/loss",
                loss,
                prog_bar=stage != "train",
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=sync_dist,
            )
            self.log(f"{stage}/accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=sync_dist)
        else:
            logits = outputs["logits"]
            labels = batch["labels"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=int(self.trainer.datamodule.info.pad_token_id or 0),
            )
            perplexity = torch.exp(loss.detach())
            self.log(
                f"{stage}/loss",
                loss,
                prog_bar=stage != "train",
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=sync_dist,
            )
            self.log(f"{stage}/perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True, sync_dist=sync_dist)

        self.log(f"{stage}/spike_rate", spike_rate, prog_bar=False, on_step=stage == "train", on_epoch=True, sync_dist=sync_dist)
        return loss


class L5PCExperiment(BaseExperiment):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._epoch_predictions = {"val": [], "test": []}
        self._representative_dump = None

    def setup(self, stage: str | None = None) -> None:
        if self.model is not None:
            return

        dataset_info = self.trainer.datamodule.info
        family = str(OmegaConf.select(self.cfg, "model.family", default="l5pc_sequence"))
        if family == "l5pc_cnn":
            self.model = L5PCCNNBaseline(self.cfg.model, dataset_info)
        elif family == "l5pc_sequence":
            self.model = L5PCSequenceModel(self.cfg.model, dataset_info)
        else:
            raise ValueError(f"Unknown L5PC model family: {family}")

    def forward(self, batch):
        return self.model(batch)

    def on_validation_epoch_start(self) -> None:
        self._epoch_predictions["val"] = []

    def on_test_epoch_start(self) -> None:
        self._epoch_predictions["test"] = []
        self._representative_dump = None

    def _voltage_weight(self) -> float:
        initial = float(self.cfg.task.loss.voltage_weight_initial)
        decay = float(self.cfg.task.loss.voltage_weight_decay)
        return initial * (decay ** float(self.current_epoch + 1))

    def _loss_terms(self, outputs, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        spike_bce = F.binary_cross_entropy_with_logits(outputs["spike_logits"], batch["spike_targets"], reduction="none")
        gamma = float(self.cfg.task.loss.spike_reweight_gamma)
        if gamma > 0:
            spike_loss = (((1 - torch.exp(-spike_bce)).clamp_min(0.0)) ** gamma * spike_bce).mean()
        else:
            spike_loss = spike_bce.mean()
        voltage_loss = F.mse_loss(outputs["voltage"], batch["voltage_targets"])
        voltage_weight = self._voltage_weight()
        total_loss = spike_loss + voltage_weight * voltage_loss
        return total_loss, spike_loss, voltage_loss, voltage_weight

    def training_step(self, batch, batch_idx):
        del batch_idx
        outputs = self.model(batch)
        loss, spike_loss, voltage_loss, voltage_weight = self._loss_terms(outputs, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/spike_loss", spike_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/voltage_loss", voltage_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/voltage_weight", float(voltage_weight), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/spike_rate", outputs["spike_rate"], on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        del batch_idx
        self._shared_eval_step(batch, stage="test")

    def _predict_full_trace(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        window_length = int(self.cfg.task.evaluation.window_length)
        overlap = int(self.cfg.task.evaluation.overlap)
        total_length = inputs.shape[1]
        stride = window_length - overlap
        if stride <= 0:
            raise ValueError("L5PC evaluation overlap must be smaller than the evaluation window length.")
        if total_length <= window_length:
            return self.model({"inputs": inputs})

        spike_pred = inputs.new_zeros((inputs.shape[0], total_length))
        voltage_pred = inputs.new_zeros((inputs.shape[0], total_length))
        num_windows = total_length // stride + 1

        for window_index in range(num_windows):
            start = window_index * stride
            if start >= total_length:
                break
            end = start + window_length
            current_inputs = inputs[:, start:end, :]
            window_outputs = self.model({"inputs": current_inputs})
            current_spikes = window_outputs["spike_logits"]
            current_voltage = window_outputs["voltage"]
            actual_end = start + current_spikes.shape[1]

            if window_index == 0:
                spike_pred[:, :actual_end] = current_spikes
                voltage_pred[:, :actual_end] = current_voltage
            elif window_index == num_windows - 1:
                target_start = start + overlap
                duration = spike_pred.shape[1] - target_start
                spike_pred[:, target_start:] = current_spikes[:, overlap : overlap + duration]
                voltage_pred[:, target_start:] = current_voltage[:, overlap : overlap + duration]
            else:
                target_start = start + overlap
                spike_pred[:, target_start:actual_end] = current_spikes[:, overlap:]
                voltage_pred[:, target_start:actual_end] = current_voltage[:, overlap:]

        return {
            "spike_logits": spike_pred,
            "voltage": voltage_pred,
            "spike_rate": torch.sigmoid(spike_pred.detach()).mean(),
        }

    def _shared_eval_step(self, batch, stage: str) -> None:
        outputs = self._predict_full_trace(batch["inputs"])
        self._epoch_predictions[stage].append(
            {
                "example_index": batch["example_index"].detach().cpu().numpy(),
                "pred_spikes": outputs["spike_logits"].detach().cpu().numpy(),
                "label_spikes": batch["spike_targets"].detach().cpu().numpy(),
                "pred_voltage": outputs["voltage"].detach().cpu().numpy(),
                "label_voltage": batch["raw_voltage_targets"].detach().cpu().numpy(),
            }
        )
        self.log(f"{stage}/spike_rate", outputs["spike_rate"], on_step=False, on_epoch=True, prog_bar=False)

        if stage == "test" and self._representative_dump is None:
            representative_index = int(self.cfg.task.evaluation.representative_index)
            batch_indices = batch["example_index"].detach().cpu().numpy()
            matches = np.where(batch_indices == representative_index)[0]
            if len(matches) > 0:
                rep = int(matches[0])
                self._representative_dump = {
                    "example_index": representative_index,
                    "inputs": batch["inputs"][rep].detach().cpu().numpy(),
                    "pred_spikes": outputs["spike_logits"][rep].detach().cpu().numpy(),
                    "label_spikes": batch["spike_targets"][rep].detach().cpu().numpy(),
                    "pred_voltage": outputs["voltage"][rep].detach().cpu().numpy(),
                    "label_voltage": batch["raw_voltage_targets"][rep].detach().cpu().numpy(),
                }

    def on_validation_epoch_end(self) -> None:
        self._finalize_eval_stage("val")

    def on_test_epoch_end(self) -> None:
        self._finalize_eval_stage("test")

    def _finalize_eval_stage(self, stage: str) -> None:
        cached = self._epoch_predictions[stage]
        if not cached:
            return

        example_index = np.concatenate([item["example_index"] for item in cached], axis=0)
        order = np.argsort(example_index)
        pred_spikes = np.concatenate([item["pred_spikes"] for item in cached], axis=0)[order]
        label_spikes = np.concatenate([item["label_spikes"] for item in cached], axis=0)[order]
        pred_voltage = np.concatenate([item["pred_voltage"] for item in cached], axis=0)[order]
        label_voltage = np.concatenate([item["label_voltage"] for item in cached], axis=0)[order]

        burn_in = int(self.cfg.task.evaluation.burn_in)
        pred_spikes = pred_spikes[:, burn_in:]
        label_spikes = label_spikes[:, burn_in:]
        pred_voltage = pred_voltage[:, burn_in:]
        label_voltage = label_voltage[:, burn_in:]

        pred_voltage_mean = float(pred_voltage.mean())
        pred_voltage_std = float(pred_voltage.std())
        label_voltage_mean = float(label_voltage.mean())
        label_voltage_std = float(label_voltage.std())
        if bool(self.cfg.task.evaluation.rescale_voltage_to_target_stats) and pred_voltage_std > 0:
            pred_voltage = ((pred_voltage - pred_voltage_mean) / pred_voltage_std) * max(label_voltage_std, 1e-6) + label_voltage_mean

        voltage_rmse = float(np.sqrt(np.mean((label_voltage - pred_voltage) ** 2)))
        spike_auc = float(roc_auc_score(label_spikes.ravel(), pred_spikes.ravel()))
        fpr, tpr, thresholds = roc_curve(label_spikes.ravel(), pred_spikes.ravel())
        low_fpr_threshold = float(self.cfg.task.evaluation.low_fpr_threshold)
        spike_pauc = _partial_auc(fpr, tpr, max_fpr=low_fpr_threshold)
        chosen_threshold, actual_fpr, actual_tpr = _threshold_at_fpr(fpr, tpr, thresholds, desired_fpr=low_fpr_threshold)
        binary_pred_spikes = pred_spikes > chosen_threshold
        lags, cross_corr_curve, peak_cross_corr, peak_lag = _spike_alignment_curve(
            binary_pred_spikes,
            label_spikes,
            max_lag=int(self.cfg.task.evaluation.max_correlation_lag),
        )

        metrics = {
            f"{stage}/voltage_rmse": voltage_rmse,
            f"{stage}/spike_auc": spike_auc,
            f"{stage}/spike_pauc": spike_pauc,
            f"{stage}/spike_cross_corr": peak_cross_corr,
            f"{stage}/spike_cross_corr_lag": float(peak_lag),
            f"{stage}/spike_threshold": chosen_threshold,
            f"{stage}/threshold_fpr": actual_fpr,
            f"{stage}/threshold_tpr": actual_tpr,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=(stage == "val"))

        if stage == "test" and bool(self.cfg.task.evaluation.save_prediction_dump) and not bool(self.trainer.fast_dev_run):
            output_dir = _logger_dir(self)
            output_dir.mkdir(parents=True, exist_ok=True)
            dump_path = output_dir / "l5pc_predictions.npz"
            np.savez_compressed(
                dump_path,
                example_index=example_index.astype(np.int32),
                label_mems=label_voltage.astype(np.float32),
                pred_mems=pred_voltage.astype(np.float32),
                label_spikes=label_spikes.astype(np.float32),
                binary_pred_spikes=binary_pred_spikes.astype(bool),
                pred_spikes=pred_spikes.astype(np.float32),
                roc_fpr=fpr.astype(np.float32),
                roc_tpr=tpr.astype(np.float32),
                roc_thresholds=thresholds.astype(np.float32),
                cross_corr_lags=lags.astype(np.int32),
                cross_corr_curve=cross_corr_curve.astype(np.float32),
                representative_input=self._representative_dump["inputs"].astype(np.float32) if self._representative_dump is not None else np.array([]),
                representative_pred_voltage=self._representative_dump["pred_voltage"].astype(np.float32) if self._representative_dump is not None else np.array([]),
                representative_label_voltage=self._representative_dump["label_voltage"].astype(np.float32) if self._representative_dump is not None else np.array([]),
                representative_pred_spikes=self._representative_dump["pred_spikes"].astype(np.float32) if self._representative_dump is not None else np.array([]),
                representative_label_spikes=self._representative_dump["label_spikes"].astype(np.float32) if self._representative_dump is not None else np.array([]),
            )
            summary_path = output_dir / "l5pc_summary.json"
            summary = {
                "run_name": self.cfg.run.name,
                "seed": int(self.cfg.run.seed),
                "model_family": str(OmegaConf.select(self.cfg, "model.family", default="l5pc_sequence")),
                "n_layers": int(self.cfg.model.n_layers),
                "d_model": int(self.cfg.model.d_model) if OmegaConf.select(self.cfg, "model.d_model", default=None) is not None else None,
                "param_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
                "voltage_rmse": voltage_rmse,
                "spike_auc": spike_auc,
                "spike_pauc": spike_pauc,
                "peak_cross_corr": peak_cross_corr,
                "peak_cross_corr_lag": peak_lag,
                "threshold": chosen_threshold,
                "threshold_fpr": actual_fpr,
                "threshold_tpr": actual_tpr,
                "prediction_dump": str(dump_path),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self._epoch_predictions[stage] = []


def build_experiment(cfg):
    if cfg.task.name == "l5pc_emulation":
        return L5PCExperiment(cfg)
    return DendriteExperiment(cfg)


def run_experiment(cfg, module, datamodule) -> None:
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
        ckpt_path = None if bool(cfg.trainer.fast_dev_run) else "best"
        trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_path)




