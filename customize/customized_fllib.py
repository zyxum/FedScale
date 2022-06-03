from fedscale.core.arg_parser import args
from fedscale.core.utils.utils_data import get_data_transform
def init_dataset():
    # not support detection task or rl
    # only openimg dataset
    if args.data_set == 'openImg':
        from fedscale.core.utils.openimage import OpenImage

        train_transform, test_transform = get_data_transform('openImg')
        train_dataset = OpenImage(args.data_dir, dataset='train', transform=train_transform)
        test_dataset = OpenImage(args.data_dir, dataset='test', transform=test_transform)
        val_dataset = OpenImage(args.data_dir, dataset='validation', transform=test_transform)
    return train_dataset, test_dataset, val_dataset

