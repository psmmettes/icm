#
# Helper functions.
#

import os
import sys
import re
import datetime

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.io import ImageReadMode, read_image

from resnet import resnet18, resnet34, resnet50, resnet101

#
#
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#
# Load specified model.
#
def get_model(network, dims):
    if network == "resnet18":
        model = resnet18(dims)
    elif network == "resnet34":
        model = resnet34(dims)
    elif network == "resnet50":
        model = resnet50(dims)
    elif network == "resnet101":
        model = resnet101(dims)
    elif network == "wrn168":
        model = wrn168(dims)
    elif network == "resnet32_im":
        model = resnet_im(32, dims)
    return model.cuda()

#
# Load CIFAR100.
#
def get_cifar100(batch_size, ex_class=-1):
    # Mean and std pixel values.
    cmean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cstd  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # Transforms.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    
    # Get train loader.
    train_data = torchvision.datasets.CIFAR100(root="../../data/", train=True, transform=transform_train)
    if ex_class > 0:
        t = np.array(train_data.targets).astype(int)
        indices = []
        for i in range(100):
            ci = np.where(t == i)[0]
            ci = np.random.choice(ci, ex_class, replace=False)
            indices.append(ci)
        indices = np.concatenate(indices)
        train_data = torch.utils.data.Subset(train_data, indices.tolist())
    train_loader = DataLoader(train_data, shuffle=True, num_workers=32, batch_size=batch_size)
    
    # Get test loader.
    test_data = torchvision.datasets.CIFAR100(root="../../data/", train=False, transform=transform_test)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=32, batch_size=batch_size)
    
    return train_loader, test_loader
    
#
# Load CIFAR10.
#
def get_cifar10(batch_size, ex_class=-1):
    # Mean and std pixel values.
    cmean = (0.4914, 0.4822, 0.4465)
    cstd  = (0.2023, 0.1994, 0.2010)

    # Transforms.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cmean, cstd)
    ])
    
    # Get train loader.
    train_data = torchvision.datasets.CIFAR10(root="../../data/", train=True, transform=transform_train, download=False)
    if ex_class > 0:
        t = np.array(train_data.targets).astype(int)
        indices = []
        for i in range(10):
            ci = np.where(t == i)[0]
            ci = np.random.choice(ci, ex_class, replace=False)
            indices.append(ci)
        indices = np.concatenate(indices)
        train_data = torch.utils.data.Subset(train_data, indices.tolist())
    train_loader = DataLoader(train_data, shuffle=True, num_workers=32, batch_size=batch_size)
    
    # Get test loader.
    test_data = torchvision.datasets.CIFAR10(root="../../data/", train=False, transform=transform_test, download=False)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=32, batch_size=batch_size)
    
    return train_loader, test_loader

#
# Load MNIST.
#
def get_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])
    train_data = torchvision.datasets.MNIST("../../data/", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=32, batch_size=batch_size)
    test_data = torchvision.datasets.MNIST("../../data/", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=32, batch_size=batch_size)
    return train_loader, test_loader

#
# CUB Birds dataset class.
#
class CubDataset(Dataset):
    def __init__(self, split: str, transforms=None):
        self.split = split
        self.transforms = transforms

        self.data_dir = os.path.join("../../data/CUB_200_2011/", "CUB_200_2011")
        self.image_dir = os.path.join(self.data_dir, "images")
        self.split_dir = os.path.join(self.data_dir, "splits")

        if split not in ["train", "val", "trainval", "test"]:
            raise ValueError("Split must be either train, val, trainval or test.")

        self.image_label_list = []

        with open(os.path.join(self.split_dir, f"{split}.txt"), "r") as f:
            for l in f:
                self.image_label_list.append(l.split())

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx):
        file_name, label = self.image_label_list[idx]
        file_name = f"{file_name}.jpg"
        label = torch.tensor(int(label))

        image = read_image(os.path.join(self.image_dir, file_name), ImageReadMode.RGB) / 255
        if image.size(0) == 1:
            print(file_name)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

#
# Load CUB Birds dataset.
#
def get_birds(batch_size):
    # Constants.
    cmean = (0.4856, 0.4994, 0.4324)
    cstd  = (0.2322, 0.2277, 0.2659)
    res   = (224, 224)

    # Transforms.
    transform_train = transforms.Compose([
        transforms.Resize(res),
        transforms.Normalize(cmean, cstd),
        transforms.RandomResizedCrop(
                size=224,
                scale=(4/10, 1),
                ratio=(3/4, 4/3),
            )
    ])
    transform_test = transforms.Compose([
        transforms.Resize(res),
        transforms.Normalize(cmean, cstd)
    ])
    
    # Datasets.
    train_dataset = CubDataset(split="trainval", transforms=transform_train)
    test_dataset = CubDataset(split="test", transforms=transform_test)
    
    # Dataloaders.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32
    )
    
    return train_loader, test_loader
