"""CBraMod model adapter.

Builds the CBraMod backbone + seizure classification head, loads pretrained weights,
and applies LoRA (custom Scheme A or PEFT Scheme B/C).

Architecture:
    CBraMod backbone (12-layer criss-cross transformer with dual-path attention)
    + classification head (all_patch_reps 3-layer MLP)

Input: [B, 16, 10, 200] (channels × patches × patch_size)
Output: [B] logits (binary: 0=normal, 1=seizure)
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .base_model import BaseModelAdapter
from ..lora.inject import inject_lora, count_lora_parameters


class CBraModAdapter(BaseModelAdapter):
    """Adapter for CBraMod model with LoRA support."""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self._ensure_import_path()

    def _ensure_import_path(self):
        """Add CBraMod source to sys.path so we can import its modules."""
        cbramod_dir = os.path.join(self.project_root, "external", "models", "CBraMod-main")
        if cbramod_dir not in sys.path:
            sys.path.insert(0, cbramod_dir)

    def build_model(self, config: dict) -> nn.Module:
        """Build CBraMod backbone + classifier and load pretrained weights.

        Args:
            config: model section of YAML config, containing:
                pretrained_weights: path to .pth file
                classifier: classifier type (default: 'all_patch_reps')
                d_model, dim_feedforward, n_layer, nhead: backbone hyperparams
                dropout: classifier dropout

        Returns:
            CBraModModel with backbone + classifier.
        """
        from models.cbramod import CBraMod

        d_model = config.get("d_model", 200)
        dim_feedforward = config.get("dim_feedforward", 800)
        n_layer = config.get("n_layer", 12)
        nhead = config.get("nhead", 8)
        classifier_type = config.get("classifier", "all_patch_reps")
        dropout = config.get("dropout", 0.1)

        backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=d_model,
            dim_feedforward=dim_feedforward, seq_len=30,
            n_layer=n_layer, nhead=nhead,
        )

        # Load pretrained weights
        pretrained_path = config.get("pretrained_weights")
        if pretrained_path:
            if not os.path.isabs(pretrained_path):
                pretrained_path = os.path.join(self.project_root, pretrained_path)
            state_dict = torch.load(pretrained_path, map_location="cpu")
            backbone.load_state_dict(state_dict)
            print(f"Loaded CBraMod pretrained weights from {pretrained_path}")

        # Replace proj_out with Identity (we use our own classifier)
        backbone.proj_out = nn.Identity()

        # Build classification head
        if classifier_type == "all_patch_reps":
            classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(16 * 10 * d_model, 10 * d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(10 * d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier_type == "avgpooling_patch_reps":
            classifier = nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(d_model, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        elif classifier_type == "all_patch_reps_twolayer":
            classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(16 * 10 * d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                Rearrange("b 1 -> (b 1)"),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        model = CBraModModel(backbone, classifier)
        return model

    def apply_lora(self, model: nn.Module, lora_config: dict) -> nn.Module:
        """Apply LoRA to CBraMod model.

        Handles three modes:
            - None: full fine-tuning (all params trainable)
            - {'frozen': true}: linear probing (freeze backbone, train classifier only)
            - {'type': 'custom'}: Scheme A — custom MHA wrapper with per-module r
            - {'type': 'peft'}: Scheme B/C — PEFT library
        """
        if lora_config is None:
            # Full fine-tuning: all parameters trainable
            for p in model.parameters():
                p.requires_grad = True
            return model

        if lora_config.get("frozen", False):
            # Linear probing: freeze backbone, train classifier only
            for name, p in model.named_parameters():
                p.requires_grad = "classifier" in name
            return model

        lora_type = lora_config.get("type", "custom")

        if lora_type == "custom":
            # Scheme A: custom LoRA injection with per-module r
            r_temporal = lora_config.get("r_temporal", 16)
            r_spatial = lora_config.get("r_spatial", 8)
            r_ffn = lora_config.get("r_ffn", 8)
            alpha_ratio = lora_config.get("lora_alpha_ratio", 2.0)
            lora_dropout = lora_config.get("lora_dropout", 0.1)
            modules_to_save = lora_config.get("modules_to_save", ["classifier"])

            model = inject_lora(
                model=model,
                r_temporal=r_temporal,
                r_spatial=r_spatial,
                r_ffn=r_ffn,
                alpha_ratio=alpha_ratio,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save,
            )
            return model

        elif lora_type == "peft":
            # Scheme B or C: use PEFT library
            from peft import LoraConfig, get_peft_model

            scheme = lora_config.get("scheme", "C")

            if scheme == "C":
                # FFN-only: target linear1, linear2
                peft_config = LoraConfig(
                    r=lora_config.get("r", 16),
                    lora_alpha=lora_config.get("lora_alpha", 32),
                    lora_dropout=lora_config.get("lora_dropout", 0.1),
                    target_modules=lora_config.get("target_modules", ["linear1", "linear2"]),
                    bias=lora_config.get("bias", "none"),
                    modules_to_save=lora_config.get("modules_to_save", ["classifier"]),
                )
                model = get_peft_model(model, peft_config)

            elif scheme == "B":
                # QKV split + PEFT: requires model surgery first
                model = self._apply_scheme_b(model, lora_config)

            return model

        else:
            raise ValueError(f"Unknown lora type: {lora_type}")

    def _apply_scheme_b(self, model: nn.Module, lora_config: dict) -> nn.Module:
        """Scheme B: Split MHA in_proj into q/k/v_proj, then apply PEFT.

        This requires modifying the model to replace nn.MultiheadAttention
        with SplitMHA, remapping pretrained weights, then using PEFT.
        """
        from peft import LoraConfig, get_peft_model

        r_t = lora_config.get("r_temporal", 16)
        r_s = lora_config.get("r_spatial", 8)
        alpha_t = r_t * lora_config.get("lora_alpha_ratio", 2.0)
        alpha_s = r_s * lora_config.get("lora_alpha_ratio", 2.0)
        lora_dropout = lora_config.get("lora_dropout", 0.1)
        modules_to_save = lora_config.get("modules_to_save", ["classifier"])

        # Step 1: Replace MHA with SplitMHA and remap weights
        layers = model.backbone.encoder.layers
        for layer in layers:
            layer.self_attn_t = self._mha_to_splitmha(layer.self_attn_t)
            layer.self_attn_s = self._mha_to_splitmha(layer.self_attn_s)

        # Step 2: Apply PEFT with per-module r (two passes)
        # First pass: temporal modules (r_t)
        config_t = LoraConfig(
            r=r_t, lora_alpha=int(alpha_t), lora_dropout=lora_dropout,
            target_modules=lora_config.get("target_modules_t", ["q_proj", "k_proj", "v_proj", "out_proj"]),
            bias="none",
        )
        model = get_peft_model(model, config_t)

        # Merge first LoRA, then apply second
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

        config_s = LoraConfig(
            r=r_s, lora_alpha=int(alpha_s), lora_dropout=lora_dropout,
            target_modules=lora_config.get("target_modules_s", ["q_proj", "k_proj", "v_proj", "out_proj"])
                    + lora_config.get("target_modules_ffn", ["linear1", "linear2"]),
            bias="none",
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, config_s)
        return model

    def _mha_to_splitmha(self, mha: nn.MultiheadAttention) -> nn.Module:
        """Convert nn.MultiheadAttention to SplitMHA with separate q/k/v_proj."""
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        split = SplitMHA(embed_dim, num_heads)

        # Remap in_proj_weight [3*d, d] -> q/k/v_proj.weight [d, d] each
        if mha.in_proj_weight is not None:
            d = embed_dim
            w = mha.in_proj_weight
            split.q_proj.weight.data = w[:d].clone()
            split.k_proj.weight.data = w[d:2*d].clone()
            split.v_proj.weight.data = w[2*d:].clone()

        # Remap in_proj_bias
        if mha.in_proj_bias is not None:
            b = mha.in_proj_bias
            split.q_proj.bias.data = b[:d].clone()
            split.k_proj.bias.data = b[d:2*d].clone()
            split.v_proj.bias.data = b[2*d:].clone()

        # Remap out_proj
        split.out_proj.weight.data = mha.out_proj.weight.data.clone()
        if mha.out_proj.bias is not None:
            split.out_proj.bias.data = mha.out_proj.bias.data.clone()

        split.dropout = mha.dropout
        return split

    def get_trainable_param_info(self, model: nn.Module) -> dict:
        """Return parameter statistics."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable / total if total > 0 else 0.0
        return {"total": total, "trainable": trainable, "ratio": ratio}


class CBraModModel(nn.Module):
    """CBraMod backbone + classification head wrapper."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


class SplitMHA(nn.Module):
    """Multi-head attention with separate q_proj/k_proj/v_proj (for PEFT compatibility).

    Used in CBraMod Scheme B to make the fused in_proj_weight accessible
    as individual Linear modules that PEFT can target.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = 0.0

    def forward(self, query, key, value, need_weights=False, attn_mask=None,
                key_padding_mask=None, **kwargs):
        import torch.nn.functional as F

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        B, N, C = q.shape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        return out, None
