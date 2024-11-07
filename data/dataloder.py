import torch
import numpy as np
from torch.utils.data import Subset, random_split, DistributedSampler, Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

def logitudinal_ds(log_dataset, TRAIN_RATIO: float, TEST_RATIO: float, stratify: bool = False):
    allsubj = [subj.split('/')[-1].replace("_normalised.nii.gz","") for subj, i, _, _ in log_dataset.samples]
    cleaned_subjects = ['_'.join(subj.split('_')[:-1]) for subj in allsubj]
    unique_subjects = np.unique(cleaned_subjects)
    test_size = int(TEST_RATIO * len(unique_subjects))
    test_subjects = np.random.choice(unique_subjects, size=test_size, replace=False)
    test_indices = [i for i, subj in enumerate(cleaned_subjects) if subj in test_subjects]
    train_indices = [i for i, subj in enumerate(cleaned_subjects) if subj not in test_subjects]
    train_dataset = Subset(log_dataset, train_indices)
    test_dataset = Subset(log_dataset, test_indices)
    return train_dataset, test_dataset 

def stratified_split_classification(dataset, test_size):
    labels = [dataset[i][1][0] for i in range(len(dataset))]  # Assuming label is the last element
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=test_size, stratify=labels
    )
    train_ds = Subset(dataset, train_indices)
    test_ds = Subset(dataset, test_indices)
    return train_ds, test_ds


def stratified_split_regression(dataset, test_ratio):
    labels = np.array([dataset[i][-1].item() for i in range(len(dataset))])  # Adjust this based on your dataset structure
    bins = pd.qcut(labels, q=10, labels=False)  # You can adjust the number of bins
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_ratio,
        stratify=bins,
        random_state=42
    )

    train_ds = [dataset[i] for i in train_idx]
    test_ds = [dataset[i] for i in test_idx]

    return train_ds, test_ds

def splitting_data(dataset, TEST_RATIO: float, stratify: bool=False):
    if stratify:
        train_ds, test_ds = stratified_split_regression(dataset, TEST_RATIO)
    else:
        train_ds, test_ds = random_split(dataset, [1-TEST_RATIO, TEST_RATIO])
    return train_ds, test_ds


def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, #shuffle is mutually exclusive with distributed sampler
        num_workers=num_workers,
        sampler=DistributedSampler(dataset)
    )