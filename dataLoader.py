import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms

import pandas as pd
import numpy as np
import random as rd

import matplotlib.pyplot as plt

class KaggleMNIST(data.Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.dataset.iloc[idx, 1:]
        label = self.dataset.iloc[idx, 0]
        
        image = np.array([image])
        label = np.array([label])
                
#         PIL_image = Image.fromarray(image.astype(np.uint8))
        
        # image = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def get_train_valid_loader(csv_path,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MINST dataset for Kaggle challange. 
    A sample digit can be optionally dislplayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - train_path: path to the dataset.
    - batch_size: how many samples per batch to load.
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot a sample digit from the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # TODO correct the mean and std
    normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    # (self, csv_file, transform=None):
    train_dataset = KaggleMNIST(
        csv_path=csv_path, transform=None,
    )

    valid_dataset = KaggleMNIST(
        csv_path=csv_path, transform=None,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data.SubsetRandomSampler(train_idx)
    valid_sampler = data.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        data_iter = iter(sample_loader)
        
        result = data_iter.next()
        image = result['image']
        label = result['label']

        image = np.array(image)
        image = image.reshape(28, 28)
        
        plt.figure(1, figsize=(3, 3))
        plt.title(label[0][0])
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()

    return (train_loader, valid_loader)