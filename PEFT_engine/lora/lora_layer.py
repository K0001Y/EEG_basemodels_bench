"""LoRA base classes: LoRALayer and LoRALinear.

LoRALayer contains the low-rank decomposition (lora_A, lora_B) with scaling.
LoRALinear wraps nn.Linear and adds the LoRA delta on top of the frozen original weight.
"""

import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-rank adaptation layer: delta_W = (alpha / r) * B @ A.

    Args:
        in_features:  input dimension of the original weight.
        out_features: output dimension of the original weight.
        r:            LoRA rank.
        alpha:        scaling numerator (actual scale = alpha / r).
        dropout:      dropout probability on the LoRA path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        # A: [r, in_features] — initialized with Kaiming uniform
        # B: [out_features, r] — initialized with zeros (so delta_W = 0 at start)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA delta: (alpha / r) * B @ (A @ x^T)^T.

        Args:
            x: [*, in_features] input tensor.

        Returns:
            [*, out_features] LoRA delta contribution.
        """
        # x: [..., in_features]
        # A: [r, in_features]  ->  A @ x^T: [r, ...]  ->  need B @ A applied to x
        # Simpler: delta = x @ A^T @ B^T * scaling
        z = self.lora_dropout(x)
        delta = z @ self.lora_A.t()  # [..., r]
        delta = delta @ self.lora_B.t()  # [..., out_features]
        return delta * self.scaling


class LoRALinear(nn.Module):
    """nn.Linear wrapper with LoRA delta added on top of frozen weight.

    forward(x) = original_linear(x) + lora_delta(x)

    The original weight and bias are frozen (requires_grad=False).
    Only lora_A and lora_B are trainable.

    Args:
        original: the nn.Linear to wrap (its weight is frozen in-place).
        r:       LoRA rank.
        alpha:   scaling numerator.
        dropout: dropout probability.
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.in_features = original.in_features
        self.out_features = original.out_features

        # Freeze original parameters
        for p in self.original.parameters():
            p.requires_grad = False

        self.lora = LoRALayer(
            in_features=original.in_features,
            out_features=original.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora(x)
