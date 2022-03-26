import pickle
from pathlib import Path
from typing import List, Any, Union, Optional

import torch
from torch import Tensor
from torch import randperm
from torch.utils.data import Dataset

TRAINING_X = "images_l.pkl"
TRAINING_Y = "labels_l.pkl"
TRAINING_UL = "images_ul.pkl"
TEST = "images_test.pkl"


class LabeledDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, name: Optional[str] = "Training"):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.x)}---"


class UnlabeledDataset(Dataset):
    def __init__(self, x: Tensor, x_processed: Optional[Any] = None, name: Optional[str] = "Unlabeled data"):
        self.x = x
        self.x_processed = x_processed
        self.name = name

    def __getitem__(self, index):
        if self.x_processed:
            return self.x[index], self.x_processed[index]
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.x)}---"


class WrapperDataset(Dataset):
    def __init__(self, datasets: List[Dataset], length: int):
        self.datasets = datasets
        self.len = length

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]

    def __len__(self):
        return self.len


class MixedDataset(Dataset):
    def __init__(self, labeled: LabeledDataset,
                 unlabeled: UnlabeledDataset,
                 epoch_ratio_over_unlabeled: int = 3,
                 name: Optional[str] = "data for semisupervised learning"):
        self.labeled_x = labeled.x
        self.labeled_y = labeled.y
        self.unlabeled_x = unlabeled
        self.name = name
        self.epoch_ratio_over_unlabeled = epoch_ratio_over_unlabeled

    def __getitem__(self, index):
        unlabeled_x_multiple = [
            self.unlabeled_x[(index * self.epoch_ratio_over_unlabeled + i) % len(self.unlabeled_x)][0] for i in
            range(self.epoch_ratio_over_unlabeled)]
        unlabeled_x_processed_multiple = [
            self.unlabeled_x[(index * self.epoch_ratio_over_unlabeled + i) % len(self.unlabeled_x)][1] for i in
            range(self.epoch_ratio_over_unlabeled)]

        return self.labeled_x[index], self.labeled_y[index], \
               unlabeled_x_multiple, unlabeled_x_processed_multiple

    def __len__(self):
        return len(self.labeled_x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.labeled_x)}---"