"""Evaluator: segment-level metrics for seizure binary classification.

Metrics:
    PR AUC (main selection metric), ROC AUC, Sensitivity, Specificity,
    Balanced Accuracy, F1, Cohen's Kappa, False Alarm Rate.

Also provides threshold optimization on validation set.
"""

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    auc,
)
from tqdm import tqdm


class Evaluator:
    """Evaluate model on a data split using segment-level metrics."""

    def __init__(self, data_loader, device=None):
        self.data_loader = data_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def collect_predictions(self, model) -> tuple:
        """Run inference and collect ground truth labels and prediction scores.

        Returns:
            (truths: np.ndarray, scores: np.ndarray)
        """
        model.eval()
        truths = []
        scores = []
        for x, y in tqdm(self.data_loader, desc="Evaluating", mininterval=2):
            x = x.to(self.device)
            y = y.to(self.device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            truths.extend(y.cpu().numpy().tolist())
            scores.extend(prob.cpu().numpy().tolist())
        return np.array(truths), np.array(scores)

    def compute_metrics(self, truths: np.ndarray, scores: np.ndarray,
                        threshold: float = 0.5) -> dict:
        """Compute all segment-level metrics.

        Args:
            truths: ground truth labels [N].
            scores: predicted probability scores [N].
            threshold: decision threshold for binary predictions.

        Returns:
            Dict of metric name → value.
        """
        preds = (scores > threshold).astype(int)
        truths_int = truths.astype(int)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(truths_int, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Sensitivity (recall for positive class)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity (recall for negative class)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # False alarm rate
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # PR AUC
        precision_arr, recall_arr, _ = precision_recall_curve(truths_int, scores)
        pr_auc = auc(recall_arr, precision_arr)

        # ROC AUC
        try:
            roc_auc = roc_auc_score(truths_int, scores)
        except ValueError:
            roc_auc = 0.0

        # Other metrics
        balanced_acc = balanced_accuracy_score(truths_int, preds)
        f1 = f1_score(truths_int, preds, zero_division=0)
        kappa = cohen_kappa_score(truths_int, preds)

        return {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_acc),
            "f1": float(f1),
            "kappa": float(kappa),
            "false_alarm_rate": float(false_alarm_rate),
            "threshold": float(threshold),
            "confusion_matrix": cm.tolist(),
        }

    def evaluate(self, model, threshold: float = 0.5) -> dict:
        """Full evaluation: collect predictions + compute metrics.

        Args:
            model: the model to evaluate.
            threshold: decision threshold.

        Returns:
            Dict of metrics.
        """
        truths, scores = self.collect_predictions(model)
        return self.compute_metrics(truths, scores, threshold)

    def optimize_threshold(self, model, min_sensitivity: float = None) -> float:
        """Find the optimal decision threshold on validation set.

        Strategy: maximize F1, or find threshold where sensitivity >= min_sensitivity.

        Args:
            model: the model.
            min_sensitivity: if set, find threshold achieving at least this sensitivity
                             with best F1. If None, simply maximize F1.

        Returns:
            Optimal threshold value.
        """
        truths, scores = self.collect_predictions(model)

        precision_arr, recall_arr, thresholds = precision_recall_curve(truths, scores)

        # F1 = 2 * P * R / (P + R)
        f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)

        if min_sensitivity is not None:
            # Find thresholds where sensitivity >= min_sensitivity
            valid_mask = recall_arr >= min_sensitivity
            if valid_mask.any():
                valid_f1 = np.where(valid_mask, f1_scores, -1)
                best_idx = np.argmax(valid_f1)
            else:
                best_idx = np.argmax(f1_scores)
        else:
            best_idx = np.argmax(f1_scores)

        # precision_recall_curve returns thresholds of length len(precision) - 1
        if best_idx < len(thresholds):
            best_threshold = float(thresholds[best_idx])
        else:
            best_threshold = 0.5

        return best_threshold

    @torch.no_grad()
    def evaluate_with_threshold(self, model, threshold: float) -> dict:
        """Evaluate with a specific decision threshold.

        Args:
            model: the model.
            threshold: decision threshold.

        Returns:
            Dict of metrics.
        """
        return self.evaluate(model, threshold)
