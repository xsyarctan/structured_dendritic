from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from structured_dendrite.models.dendrites.optim import build_optim_settings, mark_module_optim, mark_parameter_optim


class _SingleDirectionGLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0 or (d_model // 2) % n_heads != 0:
            raise ValueError("d_model and d_model//2 must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.value_head_dim = d_model // n_heads
        self.key_head_dim = (d_model // 2) // n_heads
        self.scaling = self.key_head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model // 2, bias=False)
        self.k_proj = nn.Linear(d_model, d_model // 2, bias=False)
        self.k_gate = nn.Sequential(
            nn.Linear(d_model, 16, bias=False),
            nn.Linear(16, d_model // 2, bias=False),
        )
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.LayerNorm(self.value_head_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        queries = self.q_proj(inputs)
        keys = self.k_proj(inputs) * self.scaling
        key_gate = self.k_gate(inputs)
        values = self.v_proj(inputs)
        gate = F.silu(self.g_proj(inputs))

        outputs = self._gated_linear_attention(queries, keys, values, key_gate)
        return self.out_proj(gate * outputs)

    def _gated_linear_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_gate: torch.Tensor,
    ) -> torch.Tensor:
        queries = rearrange(queries, "b l (h d) -> b h l d", h=self.n_heads)
        keys = rearrange(keys, "b l (h d) -> b h l d", h=self.n_heads)
        values = rearrange(values, "b l (h d) -> b h l d", h=self.n_heads)
        key_gate = rearrange(key_gate, "b l (h d) -> b h l d", h=self.n_heads)

        gate = torch.exp(F.logsigmoid(key_gate).mean(dim=-1, keepdim=True) / 16.0)
        prefix_gate = torch.cumprod(gate + 1e-6, dim=2)

        kv = torch.einsum("bhld,bhle->bhlde", keys, values)
        kv_scaled = kv / prefix_gate.unsqueeze(-1)
        state = torch.cumsum(kv_scaled, dim=2) * prefix_gate.unsqueeze(-1)
        outputs = torch.einsum("bhld,bhlde->bhle", queries, state)
        outputs = self.group_norm(outputs)
        return rearrange(outputs, "b h l d -> b l (h d)")


class GLADendrite(nn.Module):
    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        self.output_ready = False
        self.direction = cfg.direction
        self.input_scale = nn.Parameter(torch.ones(d_model) * float(cfg.input_scale_init))
        self.forward_gla = _SingleDirectionGLA(d_model=d_model, n_heads=int(cfg.n_heads))
        self.backward_gla = _SingleDirectionGLA(d_model=d_model, n_heads=int(cfg.n_heads)) if cfg.direction == "bidir" else None

        if bool(cfg.freeze_processor):
            for parameter in self.forward_gla.parameters():
                parameter.requires_grad = False
            if self.backward_gla is not None:
                for parameter in self.backward_gla.parameters():
                    parameter.requires_grad = False

        processor_optim = build_optim_settings(cfg, "processor")
        skip_optim = build_optim_settings(cfg, "skip")
        mark_module_optim(self.forward_gla, processor_optim)
        mark_module_optim(self.backward_gla, processor_optim)
        mark_parameter_optim(self.input_scale, skip_optim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_gla(inputs)
        if self.backward_gla is not None:
            reversed_inputs = torch.flip(inputs, dims=[1])
            backward = self.backward_gla(reversed_inputs)
            outputs = outputs + torch.flip(backward, dims=[1])
        return outputs + inputs * self.input_scale
