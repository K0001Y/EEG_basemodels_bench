"""Model adapter package.

Provides BaseModelAdapter and concrete implementations for CBraMod and LaBraM.
"""

from .base_model import BaseModelAdapter
from .cbramod_adapter import CBraModAdapter
from .labram_adapter import LaBraMAdapter

__all__ = ["BaseModelAdapter", "CBraModAdapter", "LaBraMAdapter"]
