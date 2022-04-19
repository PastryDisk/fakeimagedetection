import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder


def get_dataset(classes, dataroot, mode, isTrain, no_crop, cropSize, no_flip, no_resize, rz_interp, loadSize, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual):
    dset_lst = []
    for cls in classes:
        print(cls)
        root = dataroot + '/' + cls
        dset = dataset_folder(mode, isTrain, no_crop, cropSize, no_flip, no_resize, root, rz_interp, loadSize, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(classes, dataroot, batch_size, num_threads, mode, isTrain, no_crop, cropSize, no_flip, no_resize, rz_interp, loadSize, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual, serial_batches):
    shuffle = not serial_batches
    dataset = get_dataset(classes, dataroot, mode, isTrain, no_crop, cropSize, no_flip, no_resize, rz_interp, loadSize, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual)
    sampler = None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(num_threads))
    return data_loader
