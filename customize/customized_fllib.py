from fedscale.core.arg_parser import args
from fedscale.dataloaders.utils_data import get_data_transform
from torchvision import datasets
import torch
def init_dataset():
    # not support detection task or rl
    # only openimg dataset
    if args.data_set == 'openImg':
        from fedscale.dataloaders.openimage import OpenImage

        train_transform, test_transform = get_data_transform('openImg')
        train_dataset = OpenImage(args.data_dir, dataset='train', transform=train_transform)
        test_dataset = OpenImage(args.data_dir, dataset='test', transform=test_transform)
        val_dataset = OpenImage(args.data_dir, dataset='validation', transform=test_transform)
    if args.data_set == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                        transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                        transform=test_transform)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                        transform=test_transform)

    return train_dataset, test_dataset, val_dataset

