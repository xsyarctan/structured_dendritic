from __future__ import annotations

import math

import torch
import torch.nn as nn

from structured_dendrite.models.dendrites.optim import build_optim_settings, mark_module_optim, mark_parameter_optim


def _repeat_rows(values: torch.Tensor, repeats: int) -> torch.Tensor:
    return values.unsqueeze(0).repeat(repeats, 1)


class DiagonalSSMKernel(nn.Module):
    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        directions = 2 if cfg.direction == "bidir" else 1
        d_state = int(cfg.d_state)

        log_dt = torch.rand(d_model) * (math.log(float(cfg.dt_max)) - math.log(float(cfg.dt_min))) + math.log(float(cfg.dt_min))
        log_a_real = torch.log(0.5 * torch.ones(d_model, d_state // 2))
        a_imag = math.pi * _repeat_rows(torch.arange(d_state // 2), d_model)
        coeffs = torch.randn(directions, d_model, d_state // 2, dtype=torch.cfloat)

        train_dynamics = not bool(cfg.freeze_dynamics)
        self._register_tensor("log_dt", log_dt, trainable=train_dynamics)
        self._register_tensor("log_a_real", log_a_real, trainable=train_dynamics)
        self._register_tensor("a_imag", a_imag, trainable=train_dynamics)
        self.coeffs = nn.Parameter(torch.view_as_real(coeffs))

        dynamics_optim = build_optim_settings(cfg, "dynamics")
        processor_optim = build_optim_settings(cfg, "processor")
        mark_parameter_optim(getattr(self, "log_dt", None), dynamics_optim)
        mark_parameter_optim(getattr(self, "log_a_real", None), dynamics_optim)
        mark_parameter_optim(getattr(self, "a_imag", None), dynamics_optim)
        mark_parameter_optim(self.coeffs, processor_optim)

    def _register_tensor(self, name: str, value: torch.Tensor, trainable: bool) -> None:
        if trainable:
            self.register_parameter(name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def forward(self, sequence_length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        coeffs = torch.view_as_complex(self.coeffs)
        a_matrix = -torch.exp(self.log_a_real) + 1j * self.a_imag
        timeline = torch.arange(sequence_length, device=dt.device)
        delta_a = a_matrix * dt.unsqueeze(-1)
        exponent = delta_a.unsqueeze(-1) * timeline
        scaled_coeffs = coeffs * (torch.exp(delta_a) - 1.0) / a_matrix
        kernels = 2 * torch.einsum("dhn,hnl->dhl", scaled_coeffs, torch.exp(exponent)).real
        return kernels


class S4DDendrite(nn.Module):
    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        self.output_ready = False
        self.direction = cfg.direction
        self.kernel = DiagonalSSMKernel(d_model=d_model, cfg=cfg)
        self.input_scale = nn.Parameter(torch.ones(d_model) * float(cfg.input_scale_init))
        skip_optim = build_optim_settings(cfg, "skip")
        mark_parameter_optim(self.input_scale, skip_optim)

    def _fft_convolution(self, inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[-1]
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * sequence_length)
        input_fft = torch.fft.rfft(inputs.float(), n=2 * sequence_length)
        outputs = torch.fft.irfft(input_fft * kernel_fft.unsqueeze(0), n=2 * sequence_length)[..., :sequence_length]
        return outputs.to(dtype=inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        kernels = self.kernel(sequence.shape[-1])
        outputs = self._fft_convolution(sequence, kernels[0])

        if self.direction == "bidir":
            reversed_sequence = torch.flip(sequence, dims=[-1])
            backward = self._fft_convolution(reversed_sequence, kernels[1])
            outputs = outputs + torch.flip(backward, dims=[-1])

        outputs = outputs.transpose(1, 2)
        return outputs + inputs * self.input_scale


class StandardS4DDendrite(nn.Module):
    def __init__(self, d_model: int, cfg, dropout: float) -> None:
        super().__init__()
        self.output_ready = True
        self.direction = cfg.direction
        self.kernel = DiagonalSSMKernel(d_model=d_model, cfg=cfg)
        self.D = nn.Parameter(torch.ones(d_model))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=-2),
        )

        skip_optim = build_optim_settings(cfg, "skip")
        processor_optim = build_optim_settings(cfg, "processor")
        mark_parameter_optim(self.D, skip_optim)
        mark_module_optim(self.output_linear, processor_optim)

    def _fft_convolution(self, inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[-1]
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * sequence_length)
        input_fft = torch.fft.rfft(inputs.float(), n=2 * sequence_length)
        outputs = torch.fft.irfft(input_fft * kernel_fft.unsqueeze(0), n=2 * sequence_length)[..., :sequence_length]
        return outputs.to(dtype=inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        kernels = self.kernel(sequence.shape[-1])
        outputs = self._fft_convolution(sequence, kernels[0])

        if self.direction == "bidir":
            reversed_sequence = torch.flip(sequence, dims=[-1])
            backward = self._fft_convolution(reversed_sequence, kernels[1])
            outputs = outputs + torch.flip(backward, dims=[-1])

        outputs = outputs + sequence * self.D.unsqueeze(-1)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.output_linear(outputs)
        return outputs.transpose(1, 2)
