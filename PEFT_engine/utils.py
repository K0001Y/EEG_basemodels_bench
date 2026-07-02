"""Utility functions: logging, seed, IO."""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir: str, name: str = "train") -> str:
    """Create output directory structure and return log file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tensorboard").mkdir(exist_ok=True)
    (output_dir / "best_adapter").mkdir(exist_ok=True)
    (output_dir / "latest_adapter").mkdir(exist_ok=True)
    log_path = str(output_dir / f"{name}.log")
    return log_path


def log_print(msg: str, log_path: str = None):
    """Print to console and optionally append to log file."""
    print(msg)
    sys.stdout.flush()
    if log_path:
        with open(log_path, "a") as f:
            f.write(msg + "\n")


def save_jsonl(data: dict, path: str):
    """Append one JSON record to a .jsonl file."""
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total / trainable / ratio statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return {"total": total, "trainable": trainable, "ratio": ratio}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"
