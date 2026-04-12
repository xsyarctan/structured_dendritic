from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from structured_dendrite.models.dendrites import build_dendrite
from structured_dendrite.models.spiking import SpikeFunction, TruncatedLIF


def cfg_select(cfg, key: str, default):
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def build_optim_settings(lr, weight_decay) -> dict[str, float] | None:
    settings: dict[str, float] = {}
    if lr is not None:
        settings["lr"] = float(lr)
    if weight_decay is not None:
        settings["weight_decay"] = float(weight_decay)
    return settings or None


class ExponentialPreprocessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernel_size: int = 20,
        init_tau: float = 5.0,
        learnable_tau: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.kernel_size = int(kernel_size)
        time_index = torch.arange(self.kernel_size, dtype=torch.float32)
        self.register_buffer("time_index", time_index)
        tau = torch.full((self.input_dim,), float(init_tau), dtype=torch.float32)
        if learnable_tau:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        tau = self.tau.clamp_min(1e-3)
        kernel = torch.exp(-self.time_index.unsqueeze(0) / tau.unsqueeze(-1))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        kernel = kernel.unsqueeze(1).to(dtype=sequence.dtype)
        padded = F.pad(sequence, (self.kernel_size - 1, 0))
        outputs = F.conv1d(padded, kernel, groups=self.input_dim)
        return outputs.transpose(1, 2)


class L5PCFactorizedLayer(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        d_model = int(model_cfg.d_model)
        self.prenorm = bool(model_cfg.prenorm)
        self.norm = nn.LayerNorm(d_model)
        self.dendrite = build_dendrite(d_model, model_cfg)
        soma_optim = build_optim_settings(
            cfg_select(model_cfg, "soma.optim.lr", None),
            cfg_select(model_cfg, "soma.optim.weight_decay", None),
        )
        self.soma = TruncatedLIF(
            d_model=d_model,
            truncation_steps=int(model_cfg.soma.truncation_steps),
            optim_settings=soma_optim,
        )
        self.threshold = float(model_cfg.soma.threshold)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
        )
        self.dropout = nn.Dropout(float(model_cfg.dropout))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        residual = hidden
        working = self.norm(hidden) if self.prenorm else hidden
        membrane = self.soma(self.dendrite(working))
        spikes = SpikeFunction.apply(self.threshold - membrane)
        outputs = residual + self.dropout(self.output_projection(spikes))
        if not self.prenorm:
            outputs = self.norm(outputs)
        return outputs, {
            "membrane": membrane,
            "spikes": spikes,
            "spike_rate": spikes.detach().float().mean(),
        }


class L5PCResidualLayer(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        d_model = int(model_cfg.d_model)
        self.prenorm = bool(model_cfg.prenorm)
        self.norm = nn.LayerNorm(d_model)
        self.dendrite = build_dendrite(d_model, model_cfg)
        self.dropout = nn.Dropout(float(model_cfg.dropout))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        residual = hidden
        working = self.norm(hidden) if self.prenorm else hidden
        processed = self.dendrite(working)
        if not getattr(self.dendrite, "output_ready", False):
            processed = self.dropout(processed)
        outputs = residual + processed
        if not self.prenorm:
            outputs = self.norm(outputs)
        return outputs, {
            "membrane": None,
            "spikes": None,
            "spike_rate": outputs.new_zeros(()),
        }


class L5PCSequenceModel(nn.Module):
    def __init__(self, model_cfg, dataset_info) -> None:
        super().__init__()
        self.variant = str(cfg_select(model_cfg, "variant", "factorized"))
        input_channels = int(dataset_info.input_channels)
        preprocess_kind = str(cfg_select(model_cfg, "preprocess.kind", "none"))
        if preprocess_kind == "exp_conv":
            self.preprocessor = ExponentialPreprocessor(
                input_dim=input_channels,
                kernel_size=int(cfg_select(model_cfg, "preprocess.kernel_size", 20)),
                init_tau=float(cfg_select(model_cfg, "preprocess.init_tau", 5.0)),
                learnable_tau=bool(cfg_select(model_cfg, "preprocess.learnable_tau", True)),
            )
        elif preprocess_kind == "none":
            self.preprocessor = nn.Identity()
        else:
            raise ValueError(f"Unknown L5PC preprocess kind: {preprocess_kind}")

        self.encoder = nn.Linear(input_channels, int(model_cfg.d_model))
        self.encoder_dropout = nn.Dropout(float(cfg_select(model_cfg, "encoder.dropout", 0.0)))
        block_cls = L5PCFactorizedLayer if self.variant == "factorized" else L5PCResidualLayer
        self.layers = nn.ModuleList([block_cls(model_cfg) for _ in range(int(model_cfg.n_layers))])
        self.final_norm = nn.LayerNorm(int(model_cfg.d_model)) if bool(cfg_select(model_cfg, "final_norm", True)) else nn.Identity()
        self.head_dropout = nn.Dropout(float(cfg_select(model_cfg, "heads.dropout", 0.0)))
        self.voltage_source = str(cfg_select(model_cfg, "heads.voltage_source", "membrane" if self.variant == "factorized" else "hidden"))
        self.spike_source = str(cfg_select(model_cfg, "heads.spike_source", "hidden"))
        self.voltage_head = nn.Linear(int(model_cfg.d_model), 1)
        self.spike_head = nn.Linear(int(model_cfg.d_model), 1)
        nn.init.constant_(self.spike_head.bias, -2.0)
        nn.init.normal_(self.spike_head.weight, mean=0.0, std=0.001)

    def _head_input(self, hidden: torch.Tensor, membrane: torch.Tensor | None, source: str) -> torch.Tensor:
        if source == "hidden" or membrane is None:
            return hidden
        if source == "membrane":
            return membrane
        raise ValueError(f"Unknown L5PC head source: {source}")

    def forward(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = batch["inputs"] if isinstance(batch, dict) else batch
        hidden = self.encoder_dropout(self.encoder(self.preprocessor(inputs)))

        last_membrane = None
        spike_rates: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, state = layer(hidden)
            if state["membrane"] is not None:
                last_membrane = state["membrane"]
            spike_rates.append(state["spike_rate"])

        hidden = self.final_norm(hidden)
        voltage_features = self._head_input(hidden, last_membrane, self.voltage_source)
        spike_features = self._head_input(hidden, last_membrane, self.spike_source)
        voltage = self.voltage_head(self.head_dropout(voltage_features)).squeeze(-1)
        spike_logits = self.spike_head(self.head_dropout(spike_features)).squeeze(-1)
        spike_rate = torch.stack(spike_rates).mean() if spike_rates else hidden.new_zeros(())
        return {
            "voltage": voltage,
            "spike_logits": spike_logits,
            "spike_rate": spike_rate,
        }


class L5PCCNNBaseline(nn.Module):
    def __init__(self, model_cfg, dataset_info) -> None:
        super().__init__()
        input_channels = int(dataset_info.input_channels)
        hidden_channels = int(cfg_select(model_cfg, "hidden_channels", 128))

        def conv_block(cin: int, cout: int, kernel_size: int) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            )

        blocks = [conv_block(input_channels, hidden_channels, kernel_size=45)]
        for _ in range(6):
            blocks.append(conv_block(hidden_channels, hidden_channels, kernel_size=19))
        self.backbone = nn.Sequential(*blocks)
        self.voltage_head = nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True)
        self.spike_head = nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.spike_head.bias, -2.0)
        nn.init.normal_(self.spike_head.weight, mean=0.0, std=0.001)

    def forward(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = batch["inputs"] if isinstance(batch, dict) else batch
        sequence = inputs.transpose(1, 2)
        hidden = self.backbone(sequence)
        voltage = self.voltage_head(hidden).squeeze(1)
        spike_logits = self.spike_head(hidden).squeeze(1)
        return {
            "voltage": voltage,
            "spike_logits": spike_logits,
            "spike_rate": torch.sigmoid(spike_logits.detach()).mean(),
        }
