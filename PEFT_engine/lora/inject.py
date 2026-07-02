"""LoRA injection logic: inject_lora() and inject_lora_layerwise().

inject_lora():          CBraMod Scheme A — per-module r (temporal/spatial/ffn)
inject_lora_layerwise(): LaBraM Scheme B — per-layer r (shallow/middle/deep)

Both functions:
1. Traverse model layers and replace target modules with LoRA wrappers
2. Freeze all backbone parameters
3. Leave LoRA params and modules_to_save trainable
"""

import re
import torch
import torch.nn as nn

from .lora_layer import LoRALinear
from .lora_mha import LoRAMultiheadAttention


def _freeze_all(model: nn.Module):
    """Freeze all parameters in the model."""
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_modules(model: nn.Module, module_names: list):
    """Unfreeze parameters whose parent module name matches any in module_names.

    Args:
        model: the model.
        module_names: list of module name substrings to unfreeze (e.g. ['classifier', 'head']).
    """
    for name, p in model.named_parameters():
        for target in module_names:
            if target in name:
                p.requires_grad = True
                break


def _unfreeze_lora_params(model: nn.Module):
    """Unfreeze all lora_A and lora_B parameters."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True


def inject_lora(
    model: nn.Module,
    r_temporal: int = 16,
    r_spatial: int = 8,
    r_ffn: int = 8,
    alpha_ratio: float = 2.0,
    lora_dropout: float = 0.1,
    modules_to_save: list = None,
) -> nn.Module:
    """Inject custom LoRA into CBraMod (Scheme A).

    Replaces self_attn_t/self_attn_s with LoRAMultiheadAttention (different r),
    and linear1/linear2 with LoRALinear.

    Expects model to have: model.backbone.encoder.layers (ModuleList of TransformerEncoderLayer).

    Args:
        model: the full model (backbone + classifier).
        r_temporal: LoRA rank for self_attn_t.
        r_spatial:  LoRA rank for self_attn_s.
        r_ffn:      LoRA rank for linear1/linear2.
        alpha_ratio: alpha = r * alpha_ratio.
        lora_dropout: dropout for LoRA layers.
        modules_to_save: list of module name substrings to keep trainable (e.g. ['classifier']).

    Returns:
        The same model with LoRA injected (in-place modification).
    """
    if modules_to_save is None:
        modules_to_save = ["classifier"]

    # Find encoder layers — supports both model.backbone.encoder and model.encoder
    layers = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):
        layers = model.backbone.encoder.layers
    elif hasattr(model, "encoder"):
        layers = model.encoder.layers
    else:
        raise ValueError("Cannot find encoder.layers in model. Expected model.backbone.encoder.layers or model.encoder.layers")

    alpha_t = r_temporal * alpha_ratio
    alpha_s = r_spatial * alpha_ratio
    alpha_ffn = r_ffn * alpha_ratio

    for i, layer in enumerate(layers):
        # Replace self_attn_t with LoRAMultiheadAttention (r_temporal)
        if hasattr(layer, "self_attn_t"):
            layer.self_attn_t = LoRAMultiheadAttention(
                original_mha=layer.self_attn_t,
                r=r_temporal,
                alpha=alpha_t,
                dropout=lora_dropout,
            )

        # Replace self_attn_s with LoRAMultiheadAttention (r_spatial)
        if hasattr(layer, "self_attn_s"):
            layer.self_attn_s = LoRAMultiheadAttention(
                original_mha=layer.self_attn_s,
                r=r_spatial,
                alpha=alpha_s,
                dropout=lora_dropout,
            )

        # Replace linear1 with LoRALinear (r_ffn)
        if hasattr(layer, "linear1"):
            layer.linear1 = LoRALinear(
                original=layer.linear1,
                r=r_ffn,
                alpha=alpha_ffn,
                dropout=lora_dropout,
            )

        # Replace linear2 with LoRALinear (r_ffn)
        if hasattr(layer, "linear2"):
            layer.linear2 = LoRALinear(
                original=layer.linear2,
                r=r_ffn,
                alpha=alpha_ffn,
                dropout=lora_dropout,
            )

    # Freeze all, then unfreeze LoRA and modules_to_save
    _freeze_all(model)
    _unfreeze_lora_params(model)
    _unfreeze_modules(model, modules_to_save)

    return model


def inject_lora_layerwise(
    model: nn.Module,
    layer_r_config: dict,
    alpha_ratio: float = 2.0,
    lora_dropout: float = 0.1,
    target_modules: list = None,
    modules_to_save: list = None,
) -> nn.Module:
    """Inject LoRA with per-layer r (LaBraM Scheme B — deep enhanced layerwise).

    Replaces qkv/proj/fc1/fc2 with LoRALinear, where each layer uses a different r
    based on its depth.

    Expects model to have: model.blocks (ModuleList of Block).

    Args:
        model: the full model.
        layer_r_config: dict like:
            {'shallow': {'layers': [0,1,2,3], 'r': 4},
             'middle':  {'layers': [4,5,6,7], 'r': 8},
             'deep':    {'layers': [8,9,10,11], 'r': 16}}
            Or a simpler dict: {0: 4, 1: 4, ..., 8: 16, ...}
        alpha_ratio: alpha = r * alpha_ratio.
        lora_dropout: dropout for LoRA layers.
        target_modules: list of attribute names to replace within each block
            (default: ['qkv', 'proj', 'fc1', 'fc2']).
        modules_to_save: list of module name substrings to keep trainable.

    Returns:
        The same model with LoRA injected.
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]
    if modules_to_save is None:
        modules_to_save = ["head"]

    # Build layer_idx -> r mapping
    layer_r_map = {}
    if isinstance(layer_r_config, dict):
        # Check if it's the {'shallow': {'layers': [...], 'r': ...}} format
        first_val = next(iter(layer_r_config.values()))
        if isinstance(first_val, dict):
            for group_name, group_config in layer_r_config.items():
                r = group_config["r"]
                for idx in group_config["layers"]:
                    layer_r_map[idx] = r
        else:
            # Already {0: 4, 1: 4, ...} format
            layer_r_map = dict(layer_r_config)
    else:
        raise ValueError(f"Unsupported layer_r_config format: {type(layer_r_config)}")

    # Find blocks — supports model.blocks, model.backbone.blocks, etc.
    blocks = None
    if hasattr(model, "blocks"):
        blocks = model.blocks
    elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        blocks = model.backbone.blocks
    else:
        raise ValueError("Cannot find blocks in model. Expected model.blocks or model.backbone.blocks")

    default_r = 8  # fallback if layer not in config

    for i, block in enumerate(blocks):
        r = layer_r_map.get(i, default_r)
        alpha = r * alpha_ratio

        # Find target linears within the block
        # The block structure varies: attn.qkv, attn.proj, mlp.fc1, mlp.fc2
        _replace_linears_in_block(block, target_modules, r, alpha, lora_dropout, prefix=f"block_{i}")

    # Freeze all, then unfreeze LoRA and modules_to_save
    _freeze_all(model)
    _unfreeze_lora_params(model)
    _unfreeze_modules(model, modules_to_save)

    return model


def _replace_linears_in_block(
    block: nn.Module,
    target_modules: list,
    r: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
):
    """Recursively search a block and replace target nn.Linear modules with LoRALinear.

    Handles nested structures like attn.qkv, attn.proj, mlp.fc1, mlp.fc2.

    Args:
        block: the block module to search.
        target_modules: list of target attribute names (e.g. ['qkv', 'proj', 'fc1', 'fc2']).
        r: LoRA rank.
        alpha: scaling numerator.
        dropout: LoRA dropout.
        prefix: debugging prefix.
    """
    replacements = []

    for name, child in block.named_modules():
        # Skip the block itself and LoRA modules
        if child is block:
            continue
        if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
            continue

        # Check if this module's attribute name (last component) is a target
        attr_name = name.split(".")[-1]
        if attr_name in target_modules and isinstance(child, nn.Linear):
            # Find the parent module and the attribute
            parent = block
            parts = name.split(".")[:-1]
            for part in parts:
                parent = getattr(parent, part)
            replacements.append((parent, attr_name, child))

    # Perform replacements
    for parent, attr_name, original_linear in replacements:
        lora_linear = LoRALinear(
            original=original_linear,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, attr_name, lora_linear)


def count_lora_parameters(model: nn.Module) -> dict:
    """Count LoRA parameters specifically (lora_A + lora_B)."""
    lora_params = 0
    total_params = 0
    trainable_params = 0
    for name, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
        if "lora_A" in name or "lora_B" in name:
            lora_params += p.numel()
    return {
        "total": total_params,
        "trainable": trainable_params,
        "lora": lora_params,
        "ratio": trainable_params / total_params if total_params > 0 else 0,
    }
