"""Siena Scalp EEG Dataset loader.

Data is preprocessed into pickles by preprocessing/preprocess_siena.py:
    Each pickle: {'X': [16, 2560], 'y': int(0|1), 'patient': str}

Train/Val/Test split by patient:
    Train: pn01–pn10
    Val:   pn11–pn12
    Test:  pn13–pn14
"""

import os
import pickle

import numpy as np
import torch

from .base_dataset import BaseDataset


class SienaDataset(BaseDataset):
    """Siena scalp EEG seizure detection dataset.

    Expected directory structure:
        {processed_data_dir}/
        ├── train/  (pn01-pn10)
        ├── val/    (pn11-pn12)
        └── test/   (pn13-pn14)
    """

    def __init__(self, config: dict, model_name: str, project_root: str = None,
                 train_config: dict = None):
        super().__init__(config, model_name, project_root)
        self.train_config = train_config or {}
        self._num_classes = config.get("num_classes", 1)
        self._task_type = config.get("task_type", "binary")

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def task_type(self) -> str:
        return self._task_type

    def get_data_loader(self) -> dict:
        """Build train/val/test DataLoaders.

        Returns:
            {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        """
        batch_size = self.train_config.get("batch_size", 64)
        num_workers = self.train_config.get("num_workers", 8)
        sampler_type = self.train_config.get("sampler", "weighted")
        augment_config = {
            "augment_time_shift": self.train_config.get("augment_time_shift", 1.0),
            "augment_channel_dropout": self.train_config.get("augment_channel_dropout", 0.1),
            "augment_noise_std": self.train_config.get("augment_noise_std", 0.01),
        }

        # Load splits
        train_samples = self._load_split("train")
        val_samples = self._load_split("val")
        test_samples = self._load_split("test")

        print(f"Siena dataset sizes: train={len(train_samples)}, "
              f"val={len(val_samples)}, test={len(test_samples)}")

        # Print class distribution
        for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            labels = [s["y"] for s in samples]
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            print(f"  {name}: positive={n_pos} ({100*n_pos/len(labels):.1f}%), "
                  f"negative={n_neg} ({100*n_neg/len(labels):.1f}%)")

        data_loaders = {
            "train": self._build_loader(
                train_samples, batch_size, sampler_type=sampler_type,
                num_workers=num_workers, augment_config=augment_config, is_train=True,
            ),
            "val": self._build_loader(
                val_samples, batch_size, shuffle=False,
                num_workers=num_workers, is_train=False,
            ),
            "test": self._build_loader(
                test_samples, batch_size, shuffle=False,
                num_workers=num_workers, is_train=False,
            ),
        }
        return data_loaders
