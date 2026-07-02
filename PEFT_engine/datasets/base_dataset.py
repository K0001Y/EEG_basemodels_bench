"""Abstract dataset interface for seizure detection.

All dataset loaders (CHB-MIT, Siena) inherit from BaseDataset.

The intermediate format (produced by preprocessing scripts):
    Each pickle file: {'X': ndarray [n_channels, n_samples], 'y': int, 'patient': str}

The Dataset.__getitem__ pipeline:
    1. Load pickle → X: [16, 2560], y: int
    2. Resample [16, 2560] → [16, 2000] (256Hz → 200Hz × 10s)
    3. Model-specific reshape:
       - CBraMod: reshape(16, 10, 200) → /100 (normalize)
       - LaBraM:  reshape(16, 10, 200)
    4. Return (tensor, label)
"""

import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy import signal as scipy_signal
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class BaseDataset(ABC):
    """Abstract base class for dataset loaders.

    Subclasses must implement get_data_loader().
    """

    def __init__(self, config: dict, model_name: str, project_root: str = None):
        """
        Args:
            config: dataset section of YAML config.
            model_name: 'cbramod' or 'labram', controls reshape logic.
            project_root: project root directory for relative paths.
        """
        self.config = config
        self.model_name = model_name
        self.project_root = project_root or os.getcwd()
        self.processed_data_dir = config.get("processed_data_dir", "")
        if not os.path.isabs(self.processed_data_dir):
            self.processed_data_dir = os.path.join(self.project_root, self.processed_data_dir)

    @abstractmethod
    def get_data_loader(self) -> dict:
        """Return {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes (1 for binary with BCEWithLogitsLoss)."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """'binary' or 'multiclass'."""

    def _load_split(self, split: str) -> list:
        """Load all pickle files for a data split.

        Args:
            split: 'train', 'val', or 'test'.

        Returns:
            List of (X, y) tuples.
        """
        split_dir = os.path.join(self.processed_data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples = []
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".pkl"):
                continue
            with open(os.path.join(split_dir, fname), "rb") as f:
                data = pickle.load(f)
            samples.append(data)
        return samples

    def _preprocess_sample(self, X: np.ndarray) -> torch.Tensor:
        """Resample and reshape a raw segment for model input.

        Args:
            X: [n_channels, n_samples] raw EEG segment (e.g. [16, 2560]).

        Returns:
            [n_channels, n_patches, patch_size] tensor (e.g. [16, 10, 200]).
        """
        # Resample: 256Hz × 10s = 2560 → 200Hz × 10s = 2000
        n_samples = X.shape[1]
        if n_samples != 2000:
            X = scipy_signal.resample(X, 2000, axis=1)

        # Reshape to [16, 10, 200]
        X = X.reshape(X.shape[0], 10, 200)

        # Model-specific normalization
        if self.model_name == "cbramod":
            X = X / 100.0

        return torch.tensor(X, dtype=torch.float32)

    def _get_sample_weights(self, samples: list) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler.

        Weight = 1 / class_count[label]

        Returns:
            [N] tensor of sample weights.
        """
        labels = [s["y"] for s in samples]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float64)
        return sample_weights

    def _build_loader(
        self,
        samples: list,
        batch_size: int,
        shuffle: bool = True,
        sampler_type: str = "weighted",
        num_workers: int = 8,
        augment_config: dict = None,
        is_train: bool = False,
    ) -> DataLoader:
        """Build a DataLoader with optional sampling and augmentation.

        Args:
            samples: list of {'X': array, 'y': int} dicts.
            batch_size: batch size.
            shuffle: whether to shuffle (ignored if sampler is used).
            sampler_type: 'weighted', 'random', or 'oversample'.
            num_workers: DataLoader workers.
            augment_config: augmentation parameters dict.
            is_train: whether this is the training split.

        Returns:
            DataLoader instance.
        """
        dataset = _EEGSegmentDataset(
            samples=samples,
            preprocess_fn=self._preprocess_sample,
            augment_config=augment_config if is_train else None,
        )

        sampler = None
        if is_train:
            if sampler_type == "weighted":
                sample_weights = self._get_sample_weights(samples)
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False  # sampler overrides shuffle
            elif sampler_type == "oversample":
                # Oversample positive class by factor
                sample_weights = self._get_sample_weights(samples)
                oversample_factor = augment_config.get("oversample_factor", 1.0) if augment_config else 1.0
                if oversample_factor > 1.0:
                    sample_weights = sample_weights * oversample_factor
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False
            # else: random (default shuffle=True)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )


class _EEGSegmentDataset(Dataset):
    """PyTorch Dataset for EEG segments from preprocessed pickles.

    Applies preprocessing (resample, reshape, normalize) and optional augmentation.
    """

    def __init__(self, samples: list, preprocess_fn=None, augment_config: dict = None):
        self.samples = samples
        self.preprocess_fn = preprocess_fn
        self.augment_config = augment_config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        X = data["X"]
        y = data["y"]

        # Apply augmentation on raw signal before preprocessing
        if self.augment_config:
            from ..augmentation import apply_augmentation
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_aug = apply_augmentation(
                X_tensor,
                time_shift_sec=self.augment_config.get("augment_time_shift", 1.0),
                channel_dropout_p=self.augment_config.get("augment_channel_dropout", 0.1),
                noise_std=self.augment_config.get("augment_noise_std", 0.01),
                sampling_rate=256,  # raw signal is 256Hz
            )
            X = X_aug.numpy()

        # Preprocess (resample, reshape, normalize)
        if self.preprocess_fn:
            X_tensor = self.preprocess_fn(X)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)

        return X_tensor, torch.tensor(y, dtype=torch.float32)
