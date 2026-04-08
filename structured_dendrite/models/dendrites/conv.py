from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from structured_dendrite.models.dendrites.optim import build_optim_settings, mark_module_optim, mark_parameter_optim


class Conv1dDendrite(nn.Module):
    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        self.output_ready = False
        self.direction = cfg.direction
        self.input_scale = nn.Parameter(torch.ones(d_model) * float(cfg.input_scale_init))
        groups = int(cfg.groups) if int(cfg.groups) > 0 else d_model

        self.forward_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=int(cfg.kernel_size),
            groups=groups,
            bias=bool(cfg.bias),
        )
        self.backward_conv = None
        if self.direction == "bidir":
            self.backward_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=int(cfg.kernel_size),
                groups=groups,
                bias=bool(cfg.bias),
            )

        if bool(cfg.freeze_processor):
            for parameter in self.forward_conv.parameters():
                parameter.requires_grad = False
            if self.backward_conv is not None:
                for parameter in self.backward_conv.parameters():
                    parameter.requires_grad = False

        processor_optim = build_optim_settings(cfg, "processor")
        skip_optim = build_optim_settings(cfg, "skip")
        mark_module_optim(self.forward_conv, processor_optim)
        mark_module_optim(self.backward_conv, processor_optim)
        mark_parameter_optim(self.input_scale, skip_optim)

    def _causal_conv(self, conv: nn.Conv1d, inputs: torch.Tensor) -> torch.Tensor:
        padding = conv.kernel_size[0] - 1
        padded = F.pad(inputs, (padding, 0))
        return conv(padded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        outputs = self._causal_conv(self.forward_conv, sequence)
        if self.backward_conv is not None:
            reversed_sequence = torch.flip(sequence, dims=[-1])
            backward = self._causal_conv(self.backward_conv, reversed_sequence)
            outputs = outputs + torch.flip(backward, dims=[-1])
        outputs = outputs.transpose(1, 2)
        return outputs + inputs * self.input_scale
