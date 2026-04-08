from __future__ import annotations

import torch
import torch.nn as nn

from structured_dendrite.models.dendrites.optim import mark_parameter_optim


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_gap: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(-membrane_gap)
        return 1.0 - membrane_gap.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (saved,) = ctx.saved_tensors
        inside = (1 - torch.abs(saved)).clamp(min=0.0)
        support = (torch.abs(saved) <= 1).float()
        return -grad_output * inside * support * 0.5


@torch.jit.script
def _fused_forward(prefix: torch.Tensor, drive: torch.Tensor, truncation: int):
    running = drive.clone()
    output = drive.clone()
    for offset in range(1, truncation + 1):
        running[..., :-offset] *= prefix[..., offset:]
        output[..., offset:] += running[..., :-offset]
    return output


@torch.jit.script
def _fused_prefix_grad(prefix: torch.Tensor, drive: torch.Tensor, output: torch.Tensor, grad_output: torch.Tensor, truncation: int):
    residual = output - drive
    grad_prefix = residual * grad_output
    running = drive
    current = residual
    for offset in range(1, truncation):
        running[..., :-offset] *= prefix[..., offset:]
        current[..., offset:] -= running[..., :-offset]
        grad_prefix[..., :-offset] += current[..., offset:] * grad_output[..., offset:]
    return torch.where(prefix != 0, grad_prefix / prefix, torch.zeros_like(prefix))


@torch.jit.script
def _fused_drive_grad(prefix: torch.Tensor, grad_output: torch.Tensor, truncation: int):
    grad_drive = grad_output.clone()
    running = torch.ones_like(grad_drive)
    for offset in range(1, truncation + 1):
        running[..., :-offset] *= prefix[..., offset:]
        grad_drive[..., :-offset] += running[..., :-offset] * grad_output[..., offset:]
    return grad_drive


class _TruncatedParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix: torch.Tensor, drive: torch.Tensor, truncation: int):
        output = _fused_forward(prefix, drive, truncation)
        ctx.save_for_backward(prefix, drive, output)
        ctx.truncation = truncation
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        prefix, drive, output = ctx.saved_tensors
        truncation = ctx.truncation
        grad_prefix = _fused_prefix_grad(prefix, drive.clone(), output, grad_output, truncation)
        grad_drive = _fused_drive_grad(prefix, grad_output, truncation)
        return grad_prefix, grad_drive, None


class TruncatedLIF(nn.Module):
    def __init__(self, d_model: int, truncation_steps: int = 4, optim_settings: dict[str, float] | None = None) -> None:
        super().__init__()
        self.truncation_steps = truncation_steps
        self.rou = nn.Parameter(torch.rand(d_model))
        self.a = nn.Parameter(torch.rand(d_model) * 0.8)
        self.b = nn.Parameter(torch.rand(d_model) * 0.3)

        mark_parameter_optim(self.rou, optim_settings)
        mark_parameter_optim(self.a, optim_settings)
        mark_parameter_optim(self.b, optim_settings)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        rou = self.rou.view(1, -1, 1)
        a = self.a.view(1, -1, 1)
        b = self.b.view(1, -1, 1)

        prefix = rou * (1 - b * sequence)
        drive = (1 - rou) * a * sequence
        membrane = _TruncatedParallel.apply(prefix, drive, self.truncation_steps)
        return membrane.transpose(1, 2)
