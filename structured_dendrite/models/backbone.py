from __future__ import annotations

import torch
import torch.nn as nn
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


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.unsqueeze(-1).float()
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_pool(sequence: torch.Tensor, mask: torch.Tensor | None, mode: str) -> torch.Tensor:
    if mode == "mean":
        return masked_mean(sequence, mask)
    if mode == "first":
        return sequence[:, 0]
    if mode == "last":
        if mask is None:
            return sequence[:, -1]
        positions = mask.long().sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(sequence.size(0), device=sequence.device)
        return sequence[batch_indices, positions]
    raise ValueError(f"Unsupported pooling mode: {mode}")


class SequenceEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
        max_positions: int,
        position_embedding: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_positions, d_model) if position_embedding == "learned" else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            batch_size, sequence_length = input_ids.shape
            positions = torch.arange(sequence_length, device=input_ids.device)
            position_ids = positions.unsqueeze(0).expand(batch_size, sequence_length)
            hidden = hidden + self.position_embedding(position_ids)
        return self.dropout(hidden)


class ImageProjection(nn.Module):
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        max_positions: int,
        position_embedding: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_channels, d_model)
        self.position_embedding = nn.Embedding(max_positions, d_model) if position_embedding == "learned" else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(sequence)
        if self.position_embedding is not None:
            batch_size, sequence_length, _ = sequence.shape
            positions = torch.arange(sequence_length, device=sequence.device)
            position_ids = positions.unsqueeze(0).expand(batch_size, sequence_length)
            hidden = hidden + self.position_embedding(position_ids)
        return self.dropout(hidden)


class DendriteBlock(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.prenorm = bool(model_cfg.prenorm)
        self.block_mode = cfg_select(model_cfg, "block.mode", "spiking")
        self.norm = nn.LayerNorm(model_cfg.d_model)
        self.dendrite = build_dendrite(model_cfg.d_model, model_cfg)
        self.soma = None
        self.threshold = None
        self.output_projection = None
        if self.block_mode == "spiking":
            soma_optim = build_optim_settings(
                cfg_select(model_cfg, "soma.optim.lr", None),
                cfg_select(model_cfg, "soma.optim.weight_decay", None),
            )
            self.soma = TruncatedLIF(
                d_model=model_cfg.d_model,
                truncation_steps=model_cfg.soma.truncation_steps,
                optim_settings=soma_optim,
            )
            self.threshold = float(model_cfg.soma.threshold)
            self.output_projection = nn.Sequential(
                nn.Linear(model_cfg.d_model, model_cfg.d_model * 2),
                nn.GLU(dim=-1),
            )
        elif self.block_mode != "residual":
            raise ValueError(f"Unsupported block mode: {self.block_mode}")
        self.dropout = nn.Dropout(model_cfg.dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = inputs
        hidden = self.norm(inputs) if self.prenorm else inputs
        if self.block_mode == "spiking":
            membrane = self.soma(self.dendrite(hidden))
            spikes = SpikeFunction.apply(self.threshold - membrane)
            outputs = residual + self.dropout(self.output_projection(spikes))
            spike_rate = spikes.detach().float().mean()
        else:
            processed = self.dendrite(hidden)
            if not getattr(self.dendrite, "output_ready", False):
                processed = self.dropout(processed)
            outputs = residual + processed
            spike_rate = hidden.new_zeros(())
        if not self.prenorm:
            outputs = self.norm(outputs)
        return outputs, spike_rate


class DendriticBackbone(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DendriteBlock(model_cfg) for _ in range(model_cfg.n_layers)])
        self.final_norm = nn.LayerNorm(model_cfg.d_model) if bool(cfg_select(model_cfg, "final_norm", True)) else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spike_rates = []
        hidden = inputs
        for layer in self.layers:
            hidden, spike_rate = layer(hidden)
            spike_rates.append(spike_rate)
        return self.final_norm(hidden), torch.stack(spike_rates).mean()


class ClassificationModel(nn.Module):
    def __init__(self, model_cfg, task_cfg, dataset_info) -> None:
        super().__init__()
        max_positions = max(int(dataset_info.sequence_length or 1) + 8, int(model_cfg.max_positions))
        self.input_kind = dataset_info.input_kind
        self.decoder_mode = cfg_select(task_cfg, "decoder_mode", "pool")
        self.pooling = cfg_select(task_cfg, "pooling", "mean")
        self.pair_mode = cfg_select(task_cfg, "pair_mode", "concat_abs_prod")
        encoder_position_embedding = cfg_select(model_cfg, "encoder.position_embedding", "none")
        encoder_dropout = float(cfg_select(model_cfg, "encoder.dropout", 0.0))
        self.backbone = DendriticBackbone(model_cfg)

        if self.input_kind in {"text", "pair_text"}:
            self.encoder = SequenceEmbedding(
                vocab_size=int(dataset_info.vocab_size),
                d_model=int(model_cfg.d_model),
                padding_idx=int(dataset_info.pad_token_id or 0),
                max_positions=max_positions,
                position_embedding=encoder_position_embedding,
                dropout=encoder_dropout,
            )
        elif self.input_kind == "image_sequence":
            self.encoder = ImageProjection(
                input_channels=int(dataset_info.image_channels or 1),
                d_model=int(model_cfg.d_model),
                max_positions=max_positions,
                position_embedding=encoder_position_embedding,
                dropout=encoder_dropout,
            )
        else:
            raise ValueError(f"Unsupported classification input kind: {self.input_kind}")

        if self.decoder_mode != "pool":
            raise ValueError(f"Unsupported classification decoder mode: {self.decoder_mode}")

        head_multiplier = 4 if dataset_info.pair_inputs else 1
        head_dim = model_cfg.d_model * head_multiplier
        self.head_norm = nn.LayerNorm(head_dim) if bool(cfg_select(task_cfg, "head_norm", False)) else nn.Identity()
        self.head_dropout = nn.Dropout(float(cfg_select(task_cfg, "head_dropout", 0.0)))
        self.head = nn.Linear(head_dim, dataset_info.num_classes)

    def _encode_sequence(self, inputs: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        hidden, spike_rate = self.backbone(hidden)
        pooled = masked_pool(hidden, attention_mask, self.pooling)
        return pooled, spike_rate

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.input_kind == "pair_text":
            if self.pair_mode != "concat_abs_prod":
                raise ValueError(f"Unsupported pair mode: {self.pair_mode}")
            pooled_a, spike_rate_a = self._encode_sequence(batch["inputs_a"], batch["mask_a"])
            pooled_b, spike_rate_b = self._encode_sequence(batch["inputs_b"], batch["mask_b"])
            features = torch.cat(
                [pooled_a, pooled_b, torch.abs(pooled_a - pooled_b), pooled_a * pooled_b],
                dim=-1,
            )
            spike_rate = 0.5 * (spike_rate_a + spike_rate_b)
        else:
            features, spike_rate = self._encode_sequence(batch["inputs"], batch.get("attention_mask"))

        logits = self.head(self.head_dropout(self.head_norm(features)))
        return {"logits": logits, "spike_rate": spike_rate}


class LanguageModel(nn.Module):
    def __init__(self, model_cfg, task_cfg, dataset_info) -> None:
        super().__init__()
        max_positions = max(int(dataset_info.sequence_length or 1) + 8, int(model_cfg.max_positions))
        decoder_mode = cfg_select(task_cfg, "decoder_mode", "sequence")
        if decoder_mode != "sequence":
            raise ValueError(f"Unsupported language-model decoder mode: {decoder_mode}")

        encoder_position_embedding = cfg_select(model_cfg, "encoder.position_embedding", "none")
        embedding_dropout = float(cfg_select(task_cfg, "embedding_dropout", cfg_select(model_cfg, "encoder.dropout", 0.0)))
        self.embedding = SequenceEmbedding(
            vocab_size=int(dataset_info.vocab_size),
            d_model=int(model_cfg.d_model),
            padding_idx=int(dataset_info.pad_token_id or 0),
            max_positions=max_positions,
            position_embedding=encoder_position_embedding,
            dropout=embedding_dropout,
        )
        self.backbone = DendriticBackbone(model_cfg)
        self.output_norm = nn.LayerNorm(model_cfg.d_model) if bool(cfg_select(task_cfg, "output_norm", True)) else nn.Identity()
        self.lm_head = nn.Linear(model_cfg.d_model, dataset_info.vocab_size, bias=False)
        if bool(model_cfg.tie_embeddings):
            self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.embedding(batch["inputs"])
        hidden, spike_rate = self.backbone(hidden)
        logits = self.lm_head(self.output_norm(hidden))
        return {"logits": logits, "spike_rate": spike_rate}
