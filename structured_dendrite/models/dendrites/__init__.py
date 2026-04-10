from structured_dendrite.models.dendrites.base import IdentityDendrite, PointwiseMLPDendrite
from structured_dendrite.models.dendrites.conv import Conv1dDendrite
from structured_dendrite.models.dendrites.s4d import S4DDendrite, StandardS4DDendrite


def build_dendrite(d_model: int, model_cfg):
    cfg = model_cfg.dendrite
    if cfg.kind == "identity":
        return IdentityDendrite()
    if cfg.kind == "pointwise_mlp":
        return PointwiseMLPDendrite(d_model=d_model, cfg=cfg)
    if cfg.kind == "conv1d":
        return Conv1dDendrite(d_model=d_model, cfg=cfg)
    if cfg.kind == "gla":
        from structured_dendrite.models.dendrites.gla import GLADendrite

        return GLADendrite(d_model=d_model, cfg=cfg)
    if cfg.kind == "s4d":
        return S4DDendrite(d_model=d_model, cfg=cfg)
    if cfg.kind == "s4d_standard":
        return StandardS4DDendrite(d_model=d_model, cfg=cfg, dropout=float(model_cfg.dropout))
    raise ValueError(f"Unknown dendrite kind: {cfg.kind}")


__all__ = ["build_dendrite"]
