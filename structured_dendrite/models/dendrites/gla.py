from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from structured_dendrite.models.dendrites.base import cfg_value
from structured_dendrite.models.dendrites.optim import build_optim_settings, mark_module_optim, mark_parameter_optim

try:
    from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla
except Exception:
    fused_chunk_gla = None
    fused_recurrent_gla = None


class _SingleDirectionGLA(nn.Module):
    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        n_heads = int(cfg.n_heads)
        if d_model % n_heads != 0 or (d_model // 2) % n_heads != 0:
            raise ValueError("d_model and d_model//2 must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.value_head_dim = d_model // n_heads
        self.key_head_dim = (d_model // 2) // n_heads
        self.scaling = self.key_head_dim ** -0.5
        self.gate_normalizer = float(cfg_value(cfg, "gla_gate_normalizer", 16.0))

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

        self._post_init()

    def _post_init(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)

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
        log_gate = F.logsigmoid(key_gate) / self.gate_normalizer

        if fused_chunk_gla is not None and queries.is_cuda:
            if self.training:
                outputs, _ = fused_chunk_gla(queries, keys, values, log_gate, initial_state=None, output_final_state=True)
            else:
                outputs, _ = fused_recurrent_gla(queries, keys, values, log_gate, initial_state=None, output_final_state=True)
        else:
            outputs = self._recurrent_fallback(queries, keys, values, log_gate)

        outputs = self.group_norm(outputs)
        return rearrange(outputs, "b h l d -> b l (h d)")

    def _recurrent_fallback(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        log_gate: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_heads, sequence_length, key_dim = queries.shape
        value_dim = values.shape[-1]
        state = queries.new_zeros((batch_size, n_heads, key_dim, value_dim))
        outputs = []
        gates = log_gate.exp().unsqueeze(-1)

        for step in range(sequence_length):
            state = state * gates[:, :, step] + torch.einsum("bhd,bhe->bhde", keys[:, :, step], values[:, :, step])
            outputs.append(torch.einsum("bhd,bhde->bhe", queries[:, :, step], state))

        return torch.stack(outputs, dim=2)


class GLADendrite(nn.Module):
    output_ready = False

    def __init__(self, d_model: int, cfg) -> None:
        super().__init__()
        self.direction = cfg.direction
        freeze_all = bool(cfg_value(cfg, "freeze_all", False))
        freeze_processor = freeze_all or bool(cfg_value(cfg, "freeze_processor", False))
        freeze_skip = freeze_all or bool(cfg_value(cfg, "freeze_skip", False))

        self.input_scale = nn.Parameter(torch.ones(d_model) * float(cfg_value(cfg, "input_scale_init", 1.0)))
        self.forward_gla = _SingleDirectionGLA(d_model=d_model, cfg=cfg)
        self.backward_gla = _SingleDirectionGLA(d_model=d_model, cfg=cfg) if cfg.direction == "bidir" else None

        if freeze_processor:
            for module in [self.forward_gla, self.backward_gla]:
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if freeze_skip:
            self.input_scale.requires_grad = False

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
