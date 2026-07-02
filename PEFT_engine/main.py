"""Unified entry point for LoRA fine-tuning experiments.

Usage:
    # Training
    python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml

    # Resume from checkpoint
    python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml \\
        --resume results/cbramod_chbmit_schemeA/latest.pt
"""

import argparse
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PEFT_engine.utils import set_seed, load_config, count_parameters, log_print, setup_logger
from PEFT_engine.models import CBraModAdapter, LaBraMAdapter
from PEFT_engine.datasets import CHBMITDataset, SienaDataset, TUSZDataset
from PEFT_engine.trainer import Trainer


def get_dataset(config: dict, project_root: str):
    """Build dataset loader based on config.

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataset_cfg = config["dataset"]
    model_name = config["model"]["name"]
    train_cfg = config.get("train", {})
    dataset_name = dataset_cfg["name"]

    if dataset_name == "chbmit":
        dataset = CHBMITDataset(dataset_cfg, model_name, project_root, train_cfg)
    elif dataset_name == "siena":
        dataset = SienaDataset(dataset_cfg, model_name, project_root, train_cfg)
    elif dataset_name == "tusz":
        dataset = TUSZDataset(dataset_cfg, model_name, project_root, train_cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset.get_data_loader()


def get_model_adapter(config: dict, project_root: str):
    """Build model adapter based on config.

    Returns:
        BaseModelAdapter instance
    """
    model_name = config["model"]["name"]
    if model_name == "cbramod":
        return CBraModAdapter(project_root=project_root)
    elif model_name == "labram":
        return LaBraMAdapter(project_root=project_root)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="PEFT Engine: LoRA fine-tuning for seizure detection")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resume training")
    args = parser.parse_args()

    # Resolve paths
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    # Load config
    config = load_config(config_path)
    project_root = config.get("project_root", PROJECT_ROOT)

    # Set seed
    seed = config.get("train", {}).get("seed", 3407)
    set_seed(seed)

    # Setup output directory
    output_dir = config.get("output", {}).get("dir", "results/default")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    config["output"]["dir"] = output_dir
    log_path = setup_logger(output_dir)

    log_print("=" * 70, log_path)
    log_print(f"PEFT Engine — LoRA Fine-tuning for Seizure Detection", log_path)
    log_print(f"Config: {config_path}", log_path)
    log_print(f"Output: {output_dir}", log_path)
    log_print(f"Seed: {seed}", log_path)

    # Build dataset
    log_print("\n--- Building Dataset ---", log_path)
    data_loader = get_dataset(config, project_root)

    # Build model adapter and model
    log_print("\n--- Building Model ---", log_path)
    adapter = get_model_adapter(config, project_root)
    model = adapter.build_model(config["model"])

    # Apply LoRA
    lora_config = config.get("lora")
    if lora_config is None:
        log_print("Mode: Full Fine-tuning (all parameters trainable)", log_path)
    elif lora_config.get("frozen", False):
        log_print("Mode: Linear Probing (frozen backbone, classifier only)", log_path)
    else:
        log_print(f"Mode: LoRA (type={lora_config.get('type')}, "
                  f"scheme={lora_config.get('scheme', 'N/A')})", log_path)

    model = adapter.apply_lora(model, lora_config)

    # Print parameter info
    param_info = adapter.get_trainable_param_info(model)
    log_print(f"Parameters: total={param_info['total']:,}, "
              f"trainable={param_info['trainable']:,}, "
              f"ratio={param_info['ratio']:.4%}", log_path)

    # Build trainer
    log_print("\n--- Building Trainer ---", log_path)
    trainer = Trainer(config, model, data_loader)

    # Resume if specified
    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(project_root, resume_path)
        log_print(f"Resuming from: {resume_path}", log_path)
        trainer.resume(resume_path)

    # Start training
    log_print("\n--- Starting Training ---", log_path)
    log_print("=" * 70, log_path)
    trainer.train()


if __name__ == "__main__":
    main()
