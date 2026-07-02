"""Trainer: training loop with checkpoint, resume, and threshold optimization.

Features:
    - Grouped learning rates (LoRA params + classifier/head params)
    - AdamW + CosineAnnealingLR with warmup (step-level)
    - Focal Loss for class imbalance
    - Early stopping on PR AUC
    - Checkpoint management (best/latest/periodic)
    - Resume from checkpoint
    - Threshold optimization on validation set
    - Logging (console + TensorBoard + JSON)
"""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import build_loss
from .evaluator import Evaluator
from .utils import count_parameters, log_print, save_jsonl


class Trainer:
    """Training loop for LoRA fine-tuning of seizure detection models.

    Args:
        config: full YAML config dict.
        model: the model (with LoRA applied or full/frozen).
        data_loader: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
        evaluator: Evaluator instance for validation/test.
    """

    def __init__(self, config: dict, model: nn.Module, data_loader: dict, evaluator=None):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = config.get("train", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.batch_size = train_cfg.get("batch_size", 64)
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.head_lr = train_cfg.get("head_learning_rate", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 0.05)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.save_ckpt_freq = train_cfg.get("save_ckpt_freq", 10)
        self.patience = train_cfg.get("early_stopping_patience", 10)
        self.seed = train_cfg.get("seed", 3407)
        self.num_workers = train_cfg.get("num_workers", 8)
        self.threshold_optimization = train_cfg.get("threshold_optimization", True)

        # Loss function
        loss_type = train_cfg.get("loss_type", "focal")
        self.criterion = build_loss(
            loss_type=loss_type,
            focal_alpha=train_cfg.get("focal_alpha", 0.25),
            focal_gamma=train_cfg.get("focal_gamma", 2.0),
        ).to(self.device)

        # Output directory
        self.output_dir = Path(config.get("output", {}).get("dir", "results/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tensorboard").mkdir(exist_ok=True)
        (self.output_dir / "best_adapter").mkdir(exist_ok=True)
        (self.output_dir / "latest_adapter").mkdir(exist_ok=True)

        # Evaluators
        self.val_evaluator = evaluator or Evaluator(data_loader["val"], self.device)
        self.test_evaluator = Evaluator(data_loader["test"], self.device)

        # TensorBoard
        self.tb_writer = SummaryWriter(str(self.output_dir / "tensorboard"))

        # JSON log path
        self.jsonl_path = str(self.output_dir / "log.jsonl")

        # Move model to device
        self.model = self.model.to(self.device)

        # Build optimizer with grouped learning rates
        self._build_optimizer()

        # Training state
        self.start_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _build_optimizer(self):
        """Build AdamW optimizer with grouped learning rates.

        LoRA params: learning_rate (1e-4)
        Classifier/head params: head_learning_rate (1e-3)
        """
        lora_params = []
        head_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            elif any(k in name for k in ["classifier", "head", "gamma"]):
                head_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.lr, "weight_decay": self.weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr, "weight_decay": self.weight_decay})
        if other_params:
            # For full fine-tuning, backbone params use base lr
            param_groups.append({"params": other_params, "lr": self.lr, "weight_decay": self.weight_decay})

        self.optimizer = torch.optim.AdamW(param_groups)

        # CosineAnnealingLR with warmup (step-level)
        steps_per_epoch = len(self.data_loader["train"])
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Full training loop with early stopping, checkpointing, and threshold optimization."""
        log_path = str(self.output_dir / "train.log")

        # Print training start info
        log_print("=" * 70, log_path)
        log_print(f"Training started. Output: {self.output_dir}", log_path)
        param_info = count_parameters(self.model)
        log_print(f"Parameters: total={param_info['total']:,}, "
                  f"trainable={param_info['trainable']:,}, "
                  f"ratio={param_info['ratio']:.4%}", log_path)
        self.tb_writer.add_scalar("model/trainable_params", param_info["trainable"], 0)
        self.tb_writer.add_scalar("model/total_params", param_info["total"], 0)
        log_print(f"Dataset: train={len(self.data_loader['train'].dataset)}, "
                  f"val={len(self.data_loader['val'].dataset)}, "
                  f"test={len(self.data_loader['test'].dataset)}", log_path)
        log_print("=" * 70, log_path)

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # Training phase
            train_loss = self._train_one_epoch(epoch)

            # Validation
            val_metrics = self.val_evaluator.evaluate(self.model)
            val_pr_auc = val_metrics["pr_auc"]

            # Test (for monitoring)
            test_metrics = self.test_evaluator.evaluate(self.model)

            # Scheduler step is done per-batch in _train_one_epoch
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log epoch results
            epoch_time = (time.time() - epoch_start) / 60
            self._log_epoch(epoch, train_loss, val_metrics, test_metrics, current_lr, epoch_time)

            # Early stopping
            if val_pr_auc > self.best_metric:
                self.best_metric = val_pr_auc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_best()
                log_print(f"  >> New best PR AUC: {val_pr_auc:.4f} (epoch {epoch})", log_path)
            else:
                self.patience_counter += 1
                log_print(f"  >> No improvement. Patience: {self.patience_counter}/{self.patience}", log_path)

            # Save latest checkpoint
            self._save_latest(epoch)

            # Periodic checkpoint
            if (epoch + 1) % self.save_ckpt_freq == 0:
                self._save_periodic(epoch)

            # Early stopping check
            if self.patience_counter >= self.patience:
                log_print(f"Early stopping at epoch {epoch} (patience {self.patience})", log_path)
                break

        # Post-training: threshold optimization + final test
        log_print("=" * 70, log_path)
        log_print("Training complete. Loading best model for final evaluation.", log_path)
        self._load_best()

        if self.threshold_optimization:
            best_threshold = self.val_evaluator.optimize_threshold(self.model)
            log_print(f"Optimal threshold (val): {best_threshold:.4f}", log_path)
        else:
            best_threshold = 0.5

        # Final test evaluation
        final_test = self.test_evaluator.evaluate(self.model, threshold=best_threshold)
        log_print(f"Final test results (threshold={best_threshold:.4f}):", log_path)
        for k, v in final_test.items():
            if k != "confusion_matrix":
                log_print(f"  {k}: {v:.4f}", log_path)
        log_print(f"  confusion_matrix: {final_test['confusion_matrix']}", log_path)

        # Save final results
        results = {
            "best_epoch": self.best_epoch,
            "best_val_pr_auc": self.best_metric,
            "optimal_threshold": best_threshold,
            "test_metrics": {k: v for k, v in final_test.items() if k != "confusion_matrix"},
        }
        with open(self.output_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)

        self.tb_writer.close()
        log_print("Done!", log_path)

    def _train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average training loss."""
        self.model.train()
        losses = []
        pbar = tqdm(self.data_loader["train"], desc=f"Epoch {epoch+1}/{self.epochs}", mininterval=5)
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.5f}")

        return float(np.mean(losses))

    def _log_epoch(self, epoch, train_loss, val_metrics, test_metrics, lr, time_min):
        """Log epoch results to console, TensorBoard, and JSON."""
        log_msg = (
            f"[Epoch {epoch+1}/{self.epochs}] Loss: {train_loss:.5f} | "
            f"Val: pr_auc={val_metrics['pr_auc']:.4f}, sens={val_metrics['sensitivity']:.4f}, "
            f"spec={val_metrics['specificity']:.4f} | "
            f"Test: pr_auc={test_metrics['pr_auc']:.4f} | "
            f"LR: {lr:.2e} | Time: {time_min:.1f} min | "
            f"EarlyStopping: {self.patience_counter}/{self.patience}"
        )
        log_print(log_msg, str(self.output_dir / "train.log"))

        # TensorBoard
        self.tb_writer.add_scalar("train/loss", train_loss, epoch)
        self.tb_writer.add_scalar("train/learning_rate", lr, epoch)
        for k, v in val_metrics.items():
            if k != "confusion_matrix":
                self.tb_writer.add_scalar(f"val/{k}", v, epoch)
        for k, v in test_metrics.items():
            if k != "confusion_matrix":
                self.tb_writer.add_scalar(f"test/{k}", v, epoch)

        # JSON
        json_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "lr": lr,
            "time_min": time_min,
            "patience": self.patience_counter,
        }
        save_jsonl(json_record, self.jsonl_path)

    def _save_best(self):
        """Save best model adapter weights."""
        best_dir = self.output_dir / "best_adapter"
        best_dir.mkdir(exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(best_dir))
        else:
            # Save trainable parameters only
            state = {k: v.data for k, v in self.model.named_parameters() if v.requires_grad}
            torch.save(state, str(best_dir / "adapter_weights.pt"))

    def _save_latest(self, epoch: int):
        """Save latest adapter weights and training state."""
        latest_dir = self.output_dir / "latest_adapter"
        latest_dir.mkdir(exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(latest_dir))
        else:
            state = {k: v.data for k, v in self.model.named_parameters() if v.requires_grad}
            torch.save(state, str(latest_dir / "adapter_weights.pt"))

        # Training state
        checkpoint = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, str(self.output_dir / "latest.pt"))

    def _save_periodic(self, epoch: int):
        """Save periodic checkpoint."""
        ckpt_dir = self.output_dir / f"checkpoint_epoch{epoch+1}"
        ckpt_dir.mkdir(exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(ckpt_dir))
        else:
            state = {k: v.data for k, v in self.model.named_parameters() if v.requires_grad}
            torch.save(state, str(ckpt_dir / "adapter_weights.pt"))

    def _load_best(self):
        """Load best adapter weights."""
        best_dir = self.output_dir / "best_adapter"
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(str(best_dir), adapter_name="default")
        elif (best_dir / "adapter_weights.pt").exists():
            state = torch.load(str(best_dir / "adapter_weights.pt"), map_location=self.device)
            self.model.load_state_dict(state, strict=False)

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: path to latest.pt file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        self.patience_counter = checkpoint["patience_counter"]

        # Restore optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # Load adapter weights
        latest_dir = self.output_dir / "latest_adapter"
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(str(latest_dir), adapter_name="default")
        elif (latest_dir / "adapter_weights.pt").exists():
            state = torch.load(str(latest_dir / "adapter_weights.pt"), map_location=self.device)
            self.model.load_state_dict(state, strict=False)

        log_print(f"Resumed from epoch {self.start_epoch}. "
                  f"Best metric: {self.best_metric:.4f} (epoch {self.best_epoch})",
                  str(self.output_dir / "train.log"))
