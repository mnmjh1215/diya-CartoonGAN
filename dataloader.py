# data loader for our CartoonGAN implementation
# we assume that all images are already preprocessed using preprocessing.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config

# transforms that will be applied to all datasets
transform = transforms.Compose([
    # resizing and center cropping is not needed since we already did those using preprocessing.py
    transforms.ToTensor(),
    # normalize tensor that each element is in range [-1, 1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def load_image_dataloader(root_dir, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=True):
    """
    :param root_dir: directory that contains another directory of images. All images should be under root_dir/<some_dir>/
    :param batch_size: batch size
    :param num_workers: number of workers for torch.utils.data.DataLoader
    :return: torch.utils.Dataloader object
    """
    assert os.path.isdir(root_dir)

    image_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    dataloader = DataLoader(image_dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return dataloader

# use as following
# photo_images = load_image_dataloader(root_dir=Config.photo_image_dir)
# animation_images = load_image_dataloader(root_dir=Config.animation_image_dir)
# edge_smoothed_images = load_image_dataloader(root_dir=Config.edge_smoothed_image_dir)
# test_photo_images = load_image_dataloader(root_dir=Config.test_photo_image_dir, shuffle=False)
