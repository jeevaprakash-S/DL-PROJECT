"""Saved split loading and class-aware, bounded training sampling."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import config

class SequenceDataset(Dataset):
    def __init__(self, split: str):
        root = config.PROCESSED_DATA_DIR
        self.x = np.load(root / f"X_{split}.npy", mmap_mode="r")
        self.next = np.load(root / f"y_next_{split}.npy", mmap_mode="r")
        self.stage = np.load(root / f"y_stage_{split}.npy", mmap_mode="r")
    def __len__(self): return len(self.stage)
    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.x[index], copy=True)).float(), torch.from_numpy(np.array(self.next[index], copy=True)).float(), torch.tensor(int(self.stage[index]), dtype=torch.long))

def counts(dataset): return np.bincount(np.asarray(dataset.stage), minlength=config.NUM_STAGES)

def make_loaders(batch_size=config.BATCH_SIZE, train_steps=None):
    datasets = {name: SequenceDataset(name) for name in ("train", "val", "test")}
    train_counts = counts(datasets["train"])
    # sqrt inverse frequency prevents benign dominance without repeating the
    # 433 exploitation windows hundreds of times in a single epoch.
    weights = 1.0 / np.sqrt(np.maximum(train_counts[np.asarray(datasets["train"].stage)], 1))
    n = len(datasets["train"]) if train_steps is None else min(len(datasets["train"]), train_steps * batch_size)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=n, replacement=True)
    loaders = {"train": DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=0),
               "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
               "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0)}
    for name, ds in datasets.items():
        print(f"{name}: {len(ds):,}; " + str(dict(zip(config.STAGES, counts(ds).tolist()))))
    return loaders, datasets["train"].x.shape[-1], train_counts
