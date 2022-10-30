import torch
from torchvision import transforms

from datasets.sdedit_dataset import SDEditDataset


def get_dataset(args, config):
    image_size = config.data.image_size
    dataset = SDEditDataset(
        args,
        config,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        ),
    )
    return dataset


def data_transform(config, X: torch.Tensor):
    if config.data.rescaled:
        X = 2 * X - 1.0
    return X


def inverse_data_transform(config, X: torch.Tensor):
    if config.data.rescaled:
        X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)
