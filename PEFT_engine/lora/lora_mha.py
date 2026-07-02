"""LoRA wrapper for nn.MultiheadAttention.

nn.MultiheadAttention uses a fused in_proj_weight of shape [3*d, d] (Q/K/V stacked).
PEFT library cannot directly match this fused weight, so we wrap the MHA and add
LoRA deltas to both the in_proj and out_proj.

This is the core component for CBraMod Scheme A (time-path enhanced custom LoRA).

The wrapper reimplements the attention forward with LoRA deltas:
    qkv = F.linear(x, in_proj_weight, in_proj_bias) + lora_in_delta(x)
    out = F.linear(attn_output, out_proj_weight, out_proj_bias) + lora_out_delta(attn_output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_layer import LoRALayer


class LoRAMultiheadAttention(nn.Module):
    """Wraps nn.MultiheadAttention with LoRA on in_proj_weight and out_proj.

    Original weights are frozen; only lora_A and lora_B parameters are trainable.

    Args:
        original_mha: the nn.MultiheadAttention module to wrap.
        r:      LoRA rank.
        alpha:  scaling numerator (scale = alpha / r).
        dropout: LoRA dropout probability.
    """

    def __init__(
        self,
        original_mha: nn.MultiheadAttention,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_mha
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Freeze original parameters
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA on in_proj_weight: [3*embed_dim, embed_dim]
        self.lora_in = LoRALayer(
            in_features=self.embed_dim,
            out_features=3 * self.embed_dim,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        # LoRA on out_proj: nn.Linear(embed_dim, embed_dim)
        self.lora_out = LoRALayer(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    def _compute_qkv_with_lora(self, x: torch.Tensor) -> tuple:
        """Compute Q, K, V projections with LoRA delta.

        Args:
            x: [batch, seq_len, embed_dim] input (self-attention: query=key=value).

        Returns:
            (q, k, v) each of shape [batch, num_heads, seq_len, head_dim].
        """
        # Original QKV projection
        weight = self.original.in_proj_weight  # [3*embed_dim, embed_dim]
        bias = self.original.in_proj_bias      # [3*embed_dim] or None

        qkv = F.linear(x, weight, bias)  # [batch, seq_len, 3*embed_dim]

        # Add LoRA delta on QKV
        qkv = qkv + self.lora_in(x)

        # Split into Q, K, V
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        # Reshape for multi-head: [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        return q, k, v

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention.

        Args:
            q, k, v: [batch, num_heads, seq_len, head_dim]
            attn_mask: attention mask [seq_len, seq_len] or broadcastable.
            key_padding_mask: [batch, seq_len] boolean mask (True = padding).

        Returns:
            [batch, seq_len, embed_dim] attention output.
        """
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, seq_len]

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.original.dropout, training=self.training)

        out = attn @ v  # [batch, num_heads, seq_len, head_dim]
        batch_size, _, seq_len, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass with LoRA deltas.

        Matches the nn.MultiheadAttention interface.
        Assumes self-attention (query == key == value) and batch_first=True.

        Returns:
            (attn_output, attn_weights) — attn_weights is None when need_weights=False.
        """
        # Self-attention: use query as input for QKV projection
        q, k, v = self._compute_qkv_with_lora(query)

        # Attention
        attn_output = self._scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # Output projection with LoRA delta
        out_weight = self.original.out_proj.weight  # [embed_dim, embed_dim]
        out_bias = self.original.out_proj.bias     # [embed_dim] or None

        output = F.linear(attn_output, out_weight, out_bias)
        output = output + self.lora_out(attn_output)  # Add LoRA delta

        if need_weights:
            return output, None
        return output, None
