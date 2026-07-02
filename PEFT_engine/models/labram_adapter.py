"""LaBraM model adapter.

Builds the LaBraM (NeuralTransformer) backbone + seizure classification head,
loads pretrained weights, and applies LoRA (custom Scheme B or PEFT Scheme A/C).

Architecture:
    NeuralTransformer backbone (12-layer standard transformer)
    + binary classification head (nn.Linear)

Input: [B, 16, 10, 200] (channels × patches × patch_size)
Output: [B] logits (binary: 0=normal, 1=seizure)

Channel alignment:
    LaBraM pre-trained with 128 electrodes. For 16 bipolar channels,
    we can either map to nearest electrode indices or disable pos_embed.
"""

import os
import sys
from functools import partial

import torch
import torch.nn as nn

from .base_model import BaseModelAdapter
from ..lora.inject import inject_lora_layerwise, count_lora_parameters


class LaBraMAdapter(BaseModelAdapter):
    """Adapter for LaBraM model with LoRA support."""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self._ensure_import_path()

    def _ensure_import_path(self):
        """Add LaBraM source to sys.path."""
        labram_dir = os.path.join(self.project_root, "external", "models", "LaBraM-main")
        if labram_dir not in sys.path:
            sys.path.insert(0, labram_dir)

    def build_model(self, config: dict) -> nn.Module:
        """Build LaBraM backbone + classifier and load pretrained weights.

        Args:
            config: model section of YAML config, containing:
                pretrained_weights: path to checkpoint
                embed_dim, depth, num_heads, mlp_ratio, patch_size: backbone hyperparams
                use_abs_pos_emb: whether to use absolute position embedding
                num_classes: number of output classes (1 for binary)

        Returns:
            LaBraMModel with backbone + head.
        """
        # Import LaBraM modules
        from modeling_finetune import NeuralTransformer, labram_base_patch200_200

        embed_dim = config.get("embed_dim", 200)
        depth = config.get("depth", 12)
        num_heads = config.get("num_heads", 10)
        mlp_ratio = config.get("mlp_ratio", 4)
        patch_size = config.get("patch_size", 200)
        use_abs_pos_emb = config.get("use_abs_pos_emb", False)
        init_values = config.get("init_values", 0.1)
        num_classes = config.get("num_classes", 1)

        # Build the model
        backbone = NeuralTransformer(
            EEG_size=2000,
            patch_size=patch_size,
            in_chans=1,
            out_chans=8,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            init_values=init_values,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=True,
            init_scale=0.001,
        )

        # Load pretrained weights
        pretrained_path = config.get("pretrained_weights")
        if pretrained_path:
            if not os.path.isabs(pretrained_path):
                pretrained_path = os.path.join(self.project_root, pretrained_path)
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Remove head weights (we'll replace the head)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded LaBraM pretrained weights from {pretrained_path}")
                print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            else:
                print(f"Warning: pretrained weights not found at {pretrained_path}")

        # Replace head for binary classification
        backbone.head = nn.Linear(embed_dim, num_classes)

        model = LaBraMModel(backbone, num_classes=num_classes)
        return model

    def apply_lora(self, model: nn.Module, lora_config: dict) -> nn.Module:
        """Apply LoRA to LaBraM model.

        Handles three modes:
            - None: full fine-tuning
            - {'frozen': true}: linear probing
            - {'type': 'custom', 'scheme': 'B'}: deep enhanced layerwise LoRA
            - {'type': 'peft', 'scheme': 'A'}: standard PEFT
            - {'type': 'peft', 'scheme': 'C'}: attention + LayerScale
        """
        if lora_config is None:
            # Full fine-tuning
            for p in model.parameters():
                p.requires_grad = True
            return model

        if lora_config.get("frozen", False):
            # Linear probing: freeze backbone, train head only
            for name, p in model.named_parameters():
                p.requires_grad = "head" in name
            return model

        lora_type = lora_config.get("type", "peft")
        scheme = lora_config.get("scheme", "A")

        if lora_type == "custom" and scheme == "B":
            # Scheme B: deep enhanced layerwise LoRA
            layer_r_config = lora_config.get("layer_r_config", {})
            alpha_ratio = lora_config.get("lora_alpha_ratio", 2.0)
            lora_dropout = lora_config.get("lora_dropout", 0.1)
            target_modules = lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"])
            modules_to_save = lora_config.get("modules_to_save", ["head"])

            # inject_lora_layerwise modifies in-place: finds model.backbone.blocks,
            # replaces target linears with LoRALinear, freezes all, unfreezes LoRA + head
            inject_lora_layerwise(
                model=model,
                layer_r_config=layer_r_config,
                alpha_ratio=alpha_ratio,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )
            return model

        elif lora_type == "peft":
            from peft import LoraConfig, get_peft_model

            if scheme == "A":
                # Standard PEFT: all 4 modules, uniform r=8
                peft_config = LoraConfig(
                    r=lora_config.get("r", 8),
                    lora_alpha=lora_config.get("lora_alpha", 16),
                    lora_dropout=lora_config.get("lora_dropout", 0.1),
                    target_modules=lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"]),
                    bias=lora_config.get("bias", "none"),
                    modules_to_save=lora_config.get("modules_to_save", ["head"]),
                )

            elif scheme == "C":
                # Attention + LayerScale: only qkv/proj, r=16, unfreeze gamma
                peft_config = LoraConfig(
                    r=lora_config.get("r", 16),
                    lora_alpha=lora_config.get("lora_alpha", 32),
                    lora_dropout=lora_config.get("lora_dropout", 0.1),
                    target_modules=lora_config.get("target_modules", ["qkv", "proj"]),
                    bias=lora_config.get("bias", "none"),
                    modules_to_save=lora_config.get("modules_to_save", ["head", "gamma_1", "gamma_2"]),
                )

            else:
                raise ValueError(f"Unknown LaBraM scheme: {scheme}")

            # Apply PEFT to the backbone
            model.backbone = get_peft_model(model.backbone, peft_config)
            return model

        else:
            raise ValueError(f"Unknown lora type/scheme: {lora_type}/{scheme}")

    def get_trainable_param_info(self, model: nn.Module) -> dict:
        """Return parameter statistics."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable / total if total > 0 else 0.0
        return {"total": total, "trainable": trainable, "ratio": ratio}


class LaBraMModel(nn.Module):
    """LaBraM backbone + classification head wrapper.

    Handles the input_chans parameter for channel alignment.
    """

    # Default input_chans: map 16 bipolar channels to first 16 electrode indices
    # This is a placeholder; in practice the mapping should be configured
    DEFAULT_INPUT_CHANS = list(range(16))

    def __init__(self, backbone: nn.Module, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, input_chans: list = None) -> torch.Tensor:
        """
        Args:
            x: [B, 16, 10, 200] (batch × channels × patches × patch_size)
            input_chans: channel indices for pos_embed selection.
                        If None and backbone has pos_embed, uses default mapping.

        Returns:
            [B] or [B, num_classes] logits.
        """
        if input_chans is None:
            input_chans = self.DEFAULT_INPUT_CHANS

        # Pass through backbone with input_chans
        out = self.backbone(x, input_chans=input_chans)

        # For binary classification (num_classes=1), squeeze last dim
        if self.num_classes == 1 and out.dim() > 1:
            out = out.squeeze(-1)

        return out
