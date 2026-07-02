"""Abstract model adapter interface.

All model adapters (CBraMod, LaBraM) inherit from BaseModelAdapter and implement:
    build_model():        construct backbone + classification head, load pretrained weights
    apply_lora():         apply LoRA (custom or PEFT) to the model
    get_trainable_param_info(): return parameter statistics
"""

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    def build_model(self, config: dict) -> nn.Module:
        """Build the complete model (backbone + classification head) and load pretrained weights.

        Args:
            config: model section of the YAML config.

        Returns:
            nn.Module: the model ready for LoRA application or full fine-tuning.
        """

    @abstractmethod
    def apply_lora(self, model: nn.Module, lora_config: dict) -> nn.Module:
        """Apply LoRA to the model.

        lora_config['type'] determines the injection method:
            'custom': use custom LoRA injectors (CBraMod Scheme A, LaBraM Scheme B)
            'peft':  use HuggingFace PEFT library (CBraMod Scheme B/C, LaBraM Scheme A/C)

        If lora_config is None or {'frozen': true}, skip LoRA:
            None:        all parameters trainable (full fine-tuning)
            frozen:true: only classifier trainable (linear probing)

        Args:
            model: the base model.
            lora_config: lora section of YAML config (or None).

        Returns:
            The model with LoRA applied (or frozen/fully trainable).
        """

    @abstractmethod
    def get_trainable_param_info(self, model: nn.Module) -> dict:
        """Return parameter statistics.

        Returns:
            {'total': int, 'trainable': int, 'ratio': float}
        """

    def save_adapter(self, model, path: str):
        """Save LoRA adapter weights.

        PEFT models: save_pretrained() → directory format.
        Custom LoRA: save only LoRA state_dict → single file.
        """
        if hasattr(model, "save_pretrained"):
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
        else:
            # Save only LoRA parameters (lora_A, lora_B) and modules_to_save
            lora_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_state[name] = param.data
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(lora_state, path)

    def load_adapter(self, model, path: str):
        """Load LoRA adapter weights."""
        if hasattr(model, "load_adapter"):
            model.load_adapter(path, adapter_name="default")
        else:
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
