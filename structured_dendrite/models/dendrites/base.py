from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from structured_dendrite.models.dendrites.optim import build_optim_settings, mark_module_optim, mark_parameter_optim


def cfg_value(cfg, key: str, default):
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


class IdentityDendrite(nn.Module):
    output_ready = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class PointwiseMLPDendrite(nn.Module):
    output_ready = False

    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        hidden_multiplier = float(cfg_value(cfg, "mlp_hidden_multiplier", 2.0))
        hidden_dim = max(1, int(round(d_model * hidden_multiplier)))
        bias = bool(cfg_value(cfg, "mlp_bias", True))

        freeze_all = bool(cfg_value(cfg, "freeze_all", False))
        freeze_processor = freeze_all or bool(cfg_value(cfg, "freeze_processor", False))
        freeze_skip = freeze_all or bool(cfg_value(cfg, "freeze_skip", False))

        self.processor = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=bias),
        )
        self.input_scale = nn.Parameter(torch.ones(d_model) * float(cfg_value(cfg, "input_scale_init", 1.0)))

        if freeze_processor:
            for parameter in self.processor.parameters():
                parameter.requires_grad = False
        if freeze_skip:
            self.input_scale.requires_grad = False

        processor_optim = build_optim_settings(cfg, "processor")
        skip_optim = build_optim_settings(cfg, "skip")
        mark_module_optim(self.processor, processor_optim)
        mark_parameter_optim(self.input_scale, skip_optim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.processor(inputs) + inputs * self.input_scale
