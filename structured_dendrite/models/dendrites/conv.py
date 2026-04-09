from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from structured_dendrite.models.dendrites.base import cfg_value
from structured_dendrite.models.dendrites.optim import build_optim_settings, mark_module_optim, mark_parameter_optim


class Conv1dDendrite(nn.Module):
    output_ready = False

    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        self.direction = cfg.direction
        depthwise_groups = int(cfg_value(cfg, "groups", -1))
        depthwise_groups = d_model if depthwise_groups <= 0 else depthwise_groups

        use_branch_mixer = bool(cfg_value(cfg, "use_branch_mixer", True))
        branch_mixer_groups = int(cfg_value(cfg, "branch_mixer_groups", 1))
        mixer_bias = bool(cfg_value(cfg, "mixer_bias", True))

        freeze_all = bool(cfg_value(cfg, "freeze_all", False))
        freeze_processor = freeze_all or bool(cfg_value(cfg, "freeze_processor", False))
        freeze_skip = freeze_all or bool(cfg_value(cfg, "freeze_skip", False))

        self.input_scale = nn.Parameter(torch.ones(d_model) * float(cfg_value(cfg, "input_scale_init", 1.0)))
        self.forward_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=int(cfg.kernel_size),
            groups=depthwise_groups,
            bias=bool(cfg.bias),
        )
        self.forward_mixer = nn.Conv1d(d_model, d_model, kernel_size=1, groups=branch_mixer_groups, bias=mixer_bias) if use_branch_mixer else nn.Identity()

        self.backward_conv = None
        self.backward_mixer = None
        if self.direction == "bidir":
            self.backward_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=int(cfg.kernel_size),
                groups=depthwise_groups,
                bias=bool(cfg.bias),
            )
            self.backward_mixer = nn.Conv1d(d_model, d_model, kernel_size=1, groups=branch_mixer_groups, bias=mixer_bias) if use_branch_mixer else nn.Identity()

        if freeze_processor:
            for module in [self.forward_conv, self.forward_mixer, self.backward_conv, self.backward_mixer]:
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if freeze_skip:
            self.input_scale.requires_grad = False

        processor_optim = build_optim_settings(cfg, "processor")
        skip_optim = build_optim_settings(cfg, "skip")
        for module in [self.forward_conv, self.forward_mixer, self.backward_conv, self.backward_mixer]:
            mark_module_optim(module, processor_optim)
        mark_parameter_optim(self.input_scale, skip_optim)

    def _causal_conv(self, conv: nn.Conv1d, mixer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        padding = conv.kernel_size[0] - 1
        padded = F.pad(inputs, (padding, 0))
        return mixer(conv(padded))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        outputs = self._causal_conv(self.forward_conv, self.forward_mixer, sequence)
        if self.backward_conv is not None and self.backward_mixer is not None:
            reversed_sequence = torch.flip(sequence, dims=[-1])
            backward = self._causal_conv(self.backward_conv, self.backward_mixer, reversed_sequence)
            outputs = outputs + torch.flip(backward, dims=[-1])
        outputs = outputs.transpose(1, 2)
        return outputs + inputs * self.input_scale
