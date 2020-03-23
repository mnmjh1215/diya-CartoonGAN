# data loader for our CartoonGAN implementation
# we assume that all images are already preprocessed using preprocessing.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from config import CartoonGANConfig as Config
from torchvision.datasets.folder import pil_loader
import glob

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
    :param shuffle: use shuffle
    :return: torch.utils.Dataloader object
    """
    assert os.path.isdir(root_dir)

    image_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    dataloader = DataLoader(image_dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return dataloader


class ImageDataset(Dataset):
    def __init__(self, root_dir, loader=pil_loader, transform=transform):
        self.loader = loader
        self.transform = transform
        self.path = os.path.join(root_dir, 'images/*')
        self.images = []
        for image_path in glob.glob(self.path):
            image = self.loader(image_path)
            image_tensor = self.transform(image)
            self.images.append(image_tensor)
        self.images = torch.stack(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], 0


def load_image_dataloader_on_RAM(root_dir, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=True):
    # load images on ram
    # only use when ram is significantly larger than data size
    assert os.path.isdir(root_dir)
    image_dataloader = DataLoader(ImageDataset(root_dir),
                                  shuffle=shuffle,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    return image_dataloader
