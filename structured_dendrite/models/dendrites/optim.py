from __future__ import annotations

from omegaconf import OmegaConf


def build_optim_settings(cfg, prefix: str) -> dict[str, float] | None:
    lr = OmegaConf.select(cfg, f"optim.{prefix}_lr", default=None)
    weight_decay = OmegaConf.select(cfg, f"optim.{prefix}_weight_decay", default=None)

    settings: dict[str, float] = {}
    if lr is not None:
        settings["lr"] = float(lr)
    if weight_decay is not None:
        settings["weight_decay"] = float(weight_decay)
    return settings or None


def mark_parameter_optim(parameter, settings: dict[str, float] | None) -> None:
    if settings is None or parameter is None or not parameter.requires_grad:
        return
    parameter._optim = dict(settings)


def mark_module_optim(module, settings: dict[str, float] | None) -> None:
    if settings is None or module is None:
        return
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter._optim = dict(settings)
