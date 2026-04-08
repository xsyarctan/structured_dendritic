from __future__ import annotations

import torch
import torch.nn as nn


class IdentityDendrite(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
